from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .calibration import apply_whitener, fit_whitener


DEFAULT_RIDGE_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)


@dataclass
class AttentionCASTOperator:
    mean: torch.Tensor
    transform: torch.Tensor
    map_query: torch.Tensor
    map_value: torch.Tensor
    ridge: float
    loo_r2: float
    supported_count: int
    rank: int
    weighting: str

    @property
    def summary(self):
        return {
            "ridge": float(self.ridge),
            "loo_r2": float(self.loo_r2),
            "supported_count": int(self.supported_count),
            "rank": int(self.rank),
            "weighting": self.weighting,
            "query_map_frobenius": float(self.map_query.norm().item()),
            "value_map_frobenius": float(self.map_value.norm().item()),
        }


def _truncate_rank(matrix, rank):
    if rank <= 0 or rank >= min(matrix.shape):
        return matrix
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    return (u[:, :rank] * s[:rank]) @ vh[:rank, :]


def _source_weights(counts, weighting, device):
    if weighting == "none":
        return torch.ones_like(counts, dtype=torch.float64, device=device)
    if weighting == "log":
        weights = torch.log1p(counts.double())
    elif weighting == "sqrt":
        weights = counts.double().sqrt()
    else:
        raise ValueError(f"Unknown Attention-CAST weighting: {weighting}")
    return weights / weights.mean().clamp_min(1e-12)


def _select_ridge_loo(u, s, uty, yw, grid):
    s2 = s * s
    u2 = u * u
    y_energy = (yw * yw).sum().item()
    best_ridge, best_mse = float(grid[0]), float("inf")
    for lam in grid:
        shrink = s2 / (s2 + lam)
        y_hat = (u * shrink[None, :]) @ uty
        hat_diag = (u2 * shrink[None, :]).sum(dim=-1)
        denom = (1.0 - hat_diag).clamp_min(1e-6)
        resid = (yw - y_hat) / denom[:, None]
        mse = float((resid * resid).sum().item())
        if mse < best_mse:
            best_mse = mse
            best_ridge = float(lam)
    loo_r2 = 1.0 - best_mse / max(y_energy, 1e-12)
    return best_ridge, float(loo_r2)


def fit_attention_cast(
    source_vectors,
    delta_query,
    delta_value,
    support_counts=None,
    ridge=None,
    whiten_eps=1e-4,
    weighting="none",
    rank=0,
    ridge_grid=DEFAULT_RIDGE_GRID,
):
    """Fit closed-form ZCA-whitened Attention-CAST operator strictly from source codebook A."""
    if source_vectors.ndim != 2 or delta_query.ndim != 2 or delta_value.ndim != 2:
        raise ValueError("Attention-CAST expects 2D matrices")
    if len(source_vectors) != len(delta_query) or len(source_vectors) != len(delta_value):
        raise ValueError("Source vectors and residual banks must match in length")

    device = source_vectors.device
    with torch.autocast(device_type=device.type, enabled=False):
        mean, transform = fit_whitener(source_vectors, eps=whiten_eps)
        z_src = apply_whitener(source_vectors, mean, transform).double()
        dq = delta_query.to(device=device, dtype=torch.float64)
        dv = delta_value.to(device=device, dtype=torch.float64)

        if support_counts is not None:
            counts = torch.as_tensor(support_counts, device=device, dtype=torch.float64)
            mask = counts > 0
            if not mask.any():
                raise ValueError("Attention-CAST requires at least one source-supported code")
            counts = counts[mask]
        else:
            norms = dq.norm(dim=-1) + dv.norm(dim=-1)
            mask = norms > 0
            if not mask.any():
                mask = torch.ones(len(source_vectors), dtype=torch.bool, device=device)
            counts = torch.ones(int(mask.sum().item()), dtype=torch.float64, device=device)

        z_sub = z_src[mask]
        y_sub = torch.cat((dq[mask], dv[mask]), dim=-1)
        weights = _source_weights(counts, weighting, device)
        sqrt_w = weights.sqrt()
        x = z_sub * sqrt_w[:, None]
        yw = y_sub * sqrt_w[:, None]

        u, s, vh = torch.linalg.svd(x, full_matrices=False)
        uty = u.T @ yw
        if ridge is None or ridge <= 0.0:
            chosen_ridge, loo_r2 = _select_ridge_loo(u, s, uty, yw, ridge_grid)
        else:
            chosen_ridge = float(ridge)
            _, loo_r2 = _select_ridge_loo(u, s, uty, yw, (chosen_ridge,))

        coef = vh.T @ ((s / (s * s + chosen_ridge))[:, None] * uty)
        dim = dq.shape[1]
        map_q = _truncate_rank(coef[:, :dim], rank)
        map_v = _truncate_rank(coef[:, dim:], rank)

    return AttentionCASTOperator(
        mean=mean,
        transform=transform,
        map_query=map_q,
        map_value=map_v,
        ridge=chosen_ridge,
        loo_r2=loo_r2,
        supported_count=int(mask.sum().item()),
        rank=int(rank),
        weighting=weighting,
    )


def compile_attention_cast(operator, target_vectors, scale=1.0):
    """Compile target per-code query and value residuals in one matrix multiply per matrix."""
    device = target_vectors.device
    with torch.autocast(device_type=device.type, enabled=False):
        z_tgt = apply_whitener(
            target_vectors,
            operator.mean.to(device=device),
            operator.transform.to(device=device),
        ).double()
        dq = float(scale) * (z_tgt @ operator.map_query.to(device=device))
        dv = float(scale) * (z_tgt @ operator.map_value.to(device=device))
        return dq.float(), dv.float()
