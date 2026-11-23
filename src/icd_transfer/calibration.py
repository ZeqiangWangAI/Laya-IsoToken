import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F


CODEBOOK_ALIGN_MODES = ("none", "mean", "bures_wasserstein")


@dataclass
class CalibrationProfile:
    logit_bias: torch.Tensor
    crowding: torch.Tensor | None
    residual: torch.Tensor | None
    summary: dict


@dataclass
class CodebookAlignment:
    mode: str
    source_mean: torch.Tensor
    target_mean: torch.Tensor
    linear_map: torch.Tensor | None
    shift_unit: torch.Tensor
    shift_norm: float
    summary: dict


def _sym_matrix_power(matrix, power):
    sym = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = torch.linalg.eigh(sym)
    eigvals = eigvals.clamp_min(1e-12)
    return (eigvecs * eigvals.pow(power).unsqueeze(0)) @ eigvecs.T


def fit_codebook_alignment(source_vectors, target_vectors, mode="none", reg=0.05):
    """Fit zero-label offline domain alignment from target codebook Z_B to source codebook Z_A.

    Modes:
    - none: identity.
    - mean: first-order centering z_b -> mu_A + (z_b - mu_B).
    - bures_wasserstein: unique minimum-distortion SPD optimal transport map matching both
      mean (mu_B -> mu_A) and shrunk covariance (Sigma_B -> Sigma_A) without gratuitous rotation.
    """
    if mode not in CODEBOOK_ALIGN_MODES:
        raise ValueError(f"Unknown codebook alignment mode: {mode}")
    if not (0.0 < reg < 1.0):
        raise ValueError("Covariance shrinkage reg must lie in (0, 1)")
    device = target_vectors.device
    with torch.autocast(device_type=device.type, enabled=False):
        src = source_vectors.double()
        tgt = target_vectors.double()
        mu_a = src.mean(0)
        mu_b = tgt.mean(0)
        delta = mu_b - mu_a
        shift_norm = float(delta.norm().item())
        shift_unit = (delta / max(shift_norm, 1e-12)).float()
        if mode == "none":
            summary = {"mode": "none", "shift_norm": shift_norm}
            return CodebookAlignment(mode, mu_a.float(), mu_b.float(), None, shift_unit, shift_norm, summary)
        if mode == "mean":
            summary = {"mode": "mean", "shift_norm": shift_norm}
            return CodebookAlignment(mode, mu_a.float(), mu_b.float(), None, shift_unit, shift_norm, summary)
        dim = src.shape[-1]
        ca = src - mu_a
        cb = tgt - mu_b
        cov_a = (ca.T @ ca) / max(len(src) - 1, 1)
        cov_b = (cb.T @ cb) / max(len(tgt) - 1, 1)
        eye = torch.eye(dim, device=device, dtype=torch.float64)
        mean_var_a = float(torch.trace(cov_a).item()) / dim
        mean_var_b = float(torch.trace(cov_b).item()) / dim
        sig_a = (1.0 - reg) * cov_a + reg * max(mean_var_a, 1e-8) * eye
        sig_b = (1.0 - reg) * cov_b + reg * max(mean_var_b, 1e-8) * eye
        sig_b_half = _sym_matrix_power(sig_b, 0.5)
        sig_b_inv_half = _sym_matrix_power(sig_b, -0.5)
        middle = sig_b_half @ sig_a @ sig_b_half
        middle_half = _sym_matrix_power(middle, 0.5)
        t_bw = sig_b_inv_half @ middle_half @ sig_b_inv_half
        t_bw = 0.5 * (t_bw + t_bw.T)
        bures_sq = (
            delta.pow(2).sum()
            + torch.trace(sig_a)
            + torch.trace(sig_b)
            - 2.0 * torch.trace(middle_half)
        ).clamp_min(0.0)
        mapped = cb @ t_bw
        mean_cos = float(F.cosine_similarity(cb, mapped, dim=-1, eps=1e-12).mean().item())
        summary = {
            "mode": "bures_wasserstein",
            "reg": float(reg),
            "shift_norm": shift_norm,
            "bures_wasserstein_distance": float(bures_sq.sqrt().item()),
            "mean_map_cosine": mean_cos,
        }
        return CodebookAlignment(mode, mu_a.float(), mu_b.float(), t_bw.float(), shift_unit, shift_norm, summary)


def apply_codebook_alignment(labels, alignment):
    """Apply offline codebook alignment to all available views of LabelViews."""
    if alignment is None or alignment.mode == "none":
        return labels
    vectors = labels.vectors.clone()
    mu_a = alignment.source_mean.to(device=vectors.device, dtype=torch.float32)
    mu_b = alignment.target_mean.to(device=vectors.device, dtype=torch.float32)
    t_map = None if alignment.linear_map is None else alignment.linear_map.to(device=vectors.device, dtype=torch.float32)
    with torch.autocast(device_type=vectors.device.type, enabled=False):
        for view in range(vectors.shape[1]):
            mask = labels.available[:, view]
            if not mask.any():
                continue
            centered = vectors[mask, view].float() - mu_b
            transformed = centered if t_map is None else centered @ t_map
            vectors[mask, view] = (mu_a + transformed).to(vectors.dtype)
    return type(labels)(vectors, labels.available, labels.operators, labels.delta_query, labels.delta_value)


def project_out_note_shift(hidden, alignment):
    """Single-note zero-corpus projection removing positive shared language/domain shift along mu_B - mu_A."""
    if alignment is None or alignment.shift_norm <= 1e-8:
        return hidden
    mu_a = alignment.source_mean.to(device=hidden.device, dtype=torch.float32)
    unit = alignment.shift_unit.to(device=hidden.device, dtype=torch.float32)
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        h = hidden.float()
        proj = ((h - mu_a) @ unit).clamp_min(0.0)
        return (h - proj.unsqueeze(-1) * unit).to(hidden.dtype)


def primary_semantic_vectors(labels):
    """Extract one atomic-or-composed semantic vector per label from LabelViews."""
    return torch.where(labels.available[:, 0:1], labels.vectors[:, 0], labels.vectors[:, 2])


def fit_whitener(source_vectors, eps=1e-4):
    if eps <= 0:
        raise ValueError("Whitening eps must be positive")
    with torch.autocast(device_type=source_vectors.device.type, enabled=False):
        data = source_vectors.double()
        mean = data.mean(0)
        centered = data - mean
        cov = (centered.T @ centered) / max(len(data) - 1, 1)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        inv_sqrt = eigvals.clamp_min(0.0).add(eps).rsqrt()
        transform = (eigvecs * inv_sqrt.unsqueeze(0)) @ eigvecs.T
    return mean, transform


def apply_whitener(vectors, mean, transform):
    with torch.autocast(device_type=vectors.device.type, enabled=False):
        whitened = (vectors.double() - mean) @ transform
        return F.normalize(whitened, dim=-1).float()


def codebook_crowding(unit_vectors, k=10, batch_size=1024):
    count = len(unit_vectors)
    if k <= 0:
        raise ValueError("Crowding neighborhood size k must be positive")
    if count <= 1:
        return torch.zeros(count, device=unit_vectors.device, dtype=torch.float32)
    neighbors = min(k, count - 1)
    vectors = F.normalize(unit_vectors.float(), dim=-1)
    out = torch.empty(count, device=vectors.device, dtype=torch.float32)
    with torch.autocast(device_type=vectors.device.type, enabled=False):
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            sim = vectors[start:stop] @ vectors.T
            rows = torch.arange(stop - start, device=vectors.device)
            sim[rows, rows + start] = -torch.inf
            out[start:stop] = sim.topk(neighbors, dim=-1).values.mean(-1)
    return out


def extrapolation_residual(source_unit, target_unit, ridge=1.0):
    """Scale-invariant subspace extrapolation residual rho_b in [0, 1].

    Uses normalized Gram matrix G_A = (d / N_A) Z_A^T Z_A (mean eigenvalue = 1)
    so rho_b = sqrt(ridge * z_b^T (G_A + ridge I)^{-1} z_b) is invariant to N_A and d
    and preserves the anisotropic clinical subspace (must be fed unwhitened embeddings).
    """
    if ridge <= 0:
        raise ValueError("Extrapolation ridge must be positive")
    with torch.autocast(device_type=source_unit.device.type, enabled=False):
        src = F.normalize(source_unit.double(), dim=-1)
        tgt = F.normalize(target_unit.double(), dim=-1)
        n_src, dim = src.shape
        scale = float(dim) / max(n_src, 1)
        gram = scale * (src.T @ src) + ridge * torch.eye(dim, device=src.device, dtype=torch.float64)
        solved = torch.linalg.solve(gram, tgt.T).T
        quad = (ridge * (tgt * solved).sum(-1)).clamp(0.0, 1.0)
        return quad.sqrt().float()


def cardinality_prior_shift(source_count, target_count, mode, scale=1.0):
    if source_count <= 0 or target_count <= 0:
        raise ValueError("Codebook sizes must be positive")
    if scale == 0.0 or mode.startswith("single_label"):
        return 0.0
    return -float(scale) * math.log(target_count / source_count)


def compile_calibration(
    source_vectors,
    target_vectors,
    mode,
    source_count=None,
    target_count=None,
    device=None,
    prior_shift_scale=0.0,
    crowding_alpha=0.0,
    crowding_k=10,
    whiten_eps=1e-4,
    residual_ridge=1.0,
    record_geometry=False,
    crowding_batch_size=1024,
    residual_target_vectors=None,
):
    resolved_source = len(source_vectors) if source_vectors is not None else (source_count or 0)
    resolved_target = len(target_vectors) if target_vectors is not None else (target_count or 0)
    resolved_device = target_vectors.device if target_vectors is not None else torch.device(device or "cpu")
    prior_shift = cardinality_prior_shift(resolved_source, resolved_target, mode, prior_shift_scale)
    need_geometry = crowding_alpha != 0.0 or record_geometry
    if not need_geometry:
        bias = torch.full((resolved_target,), prior_shift, device=resolved_device, dtype=torch.float32)
        summary = {
            "source_count": resolved_source,
            "target_count": resolved_target,
            "prior_shift_scale": float(prior_shift_scale),
            "prior_shift": prior_shift,
            "crowding_alpha": 0.0,
            "crowding_k": int(crowding_k),
            "geometry_computed": False,
        }
        return CalibrationProfile(bias, None, None, summary)
    if source_vectors is None or target_vectors is None:
        raise ValueError("Semantic crowding/geometry calibration requires label embeddings")
    mean, transform = fit_whitener(source_vectors, eps=whiten_eps)
    src_unit = apply_whitener(source_vectors, mean, transform)
    tgt_unit = apply_whitener(target_vectors, mean, transform)
    src_crowd = codebook_crowding(src_unit, k=crowding_k, batch_size=crowding_batch_size)
    tgt_crowd = codebook_crowding(tgt_unit, k=crowding_k, batch_size=crowding_batch_size)
    src_crowd_mean = float(src_crowd.mean().item())
    # Compute extrapolation residual in mean-centered RAW semantic space so clinical anisotropy is preserved
    raw_tgt_source = target_vectors if residual_target_vectors is None else residual_target_vectors
    with torch.autocast(device_type=source_vectors.device.type, enabled=False):
        raw_src = F.normalize(source_vectors.double() - mean, dim=-1).float()
        raw_tgt = F.normalize(raw_tgt_source.double() - mean, dim=-1).float()
    residual = extrapolation_residual(raw_src, raw_tgt, ridge=residual_ridge)
    bias = torch.full((resolved_target,), prior_shift, device=resolved_device, dtype=torch.float32)
    if crowding_alpha != 0.0:
        bias = bias - float(crowding_alpha) * (tgt_crowd - src_crowd_mean)
    summary = {
        "source_count": resolved_source,
        "target_count": resolved_target,
        "prior_shift_scale": float(prior_shift_scale),
        "prior_shift": prior_shift,
        "crowding_alpha": float(crowding_alpha),
        "crowding_k": int(crowding_k),
        "whiten_eps": float(whiten_eps),
        "residual_ridge": float(residual_ridge),
        "geometry_computed": True,
        "source_crowding_mean": src_crowd_mean,
        "target_crowding_mean": float(tgt_crowd.mean().item()),
        "residual_mean": float(residual.mean().item()),
        "residual_p50": float(torch.quantile(residual, 0.5).item()),
        "residual_p90": float(torch.quantile(residual, 0.9).item()),
    }
    return CalibrationProfile(bias, tgt_crowd, residual, summary)


def residual_buckets(counts, keys, residual, edges=(0.25, 0.5, 0.75)):
    """Stratify multilabel transfer metrics by zero-label extrapolation residual rho_b."""
    values = residual.detach().cpu().tolist()
    bounds = [0.0, *edges, 1.0001]
    buckets = {}
    for low, high in zip(bounds[:-1], bounds[1:]):
        label = f"rho_{low:.2f}_to_{min(high, 1.0):.2f}"
        selected = [key for key, value in zip(keys, values, strict=True) if low <= value < high]
        buckets[label] = {
            "candidate_labels": len(selected),
            **counts.label_result(selected),
        }
    return buckets


def _recover_source_probs(source_prior_delta, source_counts):
    """Exact algebraic inversion of IsoTokenHead.set_source_prior(counts, N, smoothing=0.5)."""
    delta = source_prior_delta.double()
    if source_counts is not None and len(source_counts) == delta.numel():
        counts = torch.as_tensor(source_counts, device=delta.device, dtype=torch.float64)
        hi = int(torch.argmax(counts).item())
        lo = int(torch.argmin(counts).item())
        c_hi, c_lo = float(counts[hi].item()), float(counts[lo].item())
        if c_hi > c_lo:
            ratio = ((c_lo + 0.5) / (c_hi + 0.5)) * math.exp(float(delta[hi].item() - delta[lo].item()))
            if ratio > 1.0 + 1e-9:
                n_est = (c_hi * ratio - c_lo) / (ratio - 1.0) - 0.5
                if n_est >= c_hi:
                    pi = ((counts + 0.5) / (n_est + 1.0)).clamp(min=1e-7, max=1.0 - 1e-7)
                    bar_pi = pi.mean().clamp(min=1e-7, max=1.0 - 1e-7)
                    bar_ell = float(torch.log(bar_pi / (1.0 - bar_pi)).item())
                    return pi, bar_ell
    # Fallback when counts are absent: calibrate bar_ell so max(pi) ~= 0.40
    bar_ell = -0.4055 - float(delta.max().item())
    pi = torch.sigmoid(delta + bar_ell).clamp(min=1e-7, max=1.0 - 1e-7)
    return pi, bar_ell


def wiener_spherical_gain(source_vecs, target_vecs, wiener_lambda=0.0, chunk_size=4096):
    """Closed-form MMSE Wiener spherical attenuation compensation g_lam(s_b) = s_b / (s_b^2 + lam * (1 - s_b^2))."""
    lam = float(wiener_lambda)
    if lam <= 0.0:
        return None, None
    with torch.autocast(device_type=target_vecs.device.type, enabled=False):
        mu = source_vecs.float().mean(dim=0, keepdim=True)
        z_a = F.normalize(source_vecs.float() - mu, dim=-1)
        z_b = F.normalize(target_vecs.float() - mu, dim=-1)
        gains = []
        sims = []
        for start in range(0, z_b.size(0), chunk_size):
            sim = z_b[start:start + chunk_size] @ z_a.T
            s = sim.max(dim=-1).values.clamp(min=0.0, max=1.0)
            g = s / (s * s + lam * (1.0 - s * s)).clamp(min=1e-8)
            gains.append(g)
            sims.append(s)
        gain = torch.cat(gains, dim=0).float()
        all_s = torch.cat(sims, dim=0)
        summary = {
            "wiener_lambda": lam,
            "crossover_cos": lam / (1.0 - lam) if lam < 1.0 else float("inf"),
            "mean_affinity": float(all_s.mean().item()),
            "mean_gain": float(gain.mean().item()),
            "p10_gain": float(torch.quantile(gain, 0.1).item()),
            "p50_gain": float(torch.quantile(gain, 0.5).item()),
            "p90_gain": float(torch.quantile(gain, 0.9).item()),
            "max_gain": float(gain.max().item()),
        }
        return gain, summary


def semantic_prior_projection(
    source_vecs,
    target_vecs,
    source_prior_delta,
    tau=0.0,
    top_k=1,
    temp=0.05,
    source_counts=None,
    chunk_size=4096,
):
    """Transfer source empirical prior onto target codebook B via exact top-1 or Bayesian mixture marginalization."""
    if tau == 0.0:
        return torch.zeros(target_vecs.size(0), device=target_vecs.device, dtype=torch.float32), None
    k = max(1, min(int(top_k), int(source_vecs.size(0))))
    with torch.autocast(device_type=target_vecs.device.type, enabled=False):
        mu = source_vecs.float().mean(dim=0, keepdim=True)
        z_a = F.normalize(source_vecs.float() - mu, dim=-1)
        z_b = F.normalize(target_vecs.float() - mu, dim=-1)
        delta_a = source_prior_delta.to(device=target_vecs.device, dtype=torch.float32)
        pi_a = None
        bar_ell = 0.0
        if k > 1:
            pi_a, bar_ell = _recover_source_probs(delta_a, source_counts)
        proj = []
        sims = []
        for start in range(0, z_b.size(0), chunk_size):
            sim = z_b[start:start + chunk_size] @ z_a.T
            if k == 1:
                top_sim, top_idx = sim.max(dim=-1)
                aff = top_sim.clamp(min=0.0, max=1.0)
                proj.append((aff * aff) * delta_a[top_idx])
                sims.append(aff)
            else:
                top_vals, top_idx = torch.topk(sim, k=k, dim=-1)
                aff = top_vals[:, 0].clamp(min=0.0, max=1.0)
                weights = torch.softmax(top_vals.double() / max(float(temp), 1e-4), dim=-1)
                pi_mix = (weights * pi_a[top_idx]).sum(dim=-1).clamp(min=1e-7, max=1.0 - 1e-7)
                ell_mix = torch.log(pi_mix / (1.0 - pi_mix)) - bar_ell
                proj.append((aff * aff) * ell_mix.float())
                sims.append(aff)
        raw = torch.cat(proj, dim=0)
        raw_mean = float(raw.mean().item())
        out = float(tau) * (raw - raw_mean)
        all_sims = torch.cat(sims, dim=0)
        summary = {
            "prior_proj_tau": float(tau),
            "prior_proj_k": k,
            "prior_proj_temp": float(temp),
            "mean_affinity": float(all_sims.mean().item()),
            "raw_mean_shift": float(tau) * raw_mean,
            "mean_shift": float(out.mean().item()),
            "std_shift": float(out.std().item()),
        }
        return out, summary


def resolve_zero_target_policy(mode, codes, source_count, has_groups=False, source_prior_tau=0.0):
    """Deterministic zero-target-data operator dispatch from public codebook metadata alone.

    Uses fixed universal constants (crowding_alpha=0.5, wiener_lambda=1/3, span_nms_cos=0.80)
    and inspects only public codebook descriptions, cardinality |B|, task output_mode, and
    composition groups—never reading target clinical notes or labels.
    """
    sample_descs = [c.description for c in codes if c.description]
    alpha_chars = 0
    upper_chars = 0
    non_latin_chars = 0
    for desc in sample_descs:
        for ch in desc:
            if ch.isalpha():
                alpha_chars += 1
                if ch.isupper():
                    upper_chars += 1
                if ord(ch) > 0x024F:
                    non_latin_chars += 1
    all_caps_ratio = upper_chars / max(alpha_chars, 1)
    non_latin_ratio = non_latin_chars / max(alpha_chars, 1)
    target_count = len(codes)
    compact_catalog = target_count < max(int(source_count) // 10, 1)

    if str(mode).startswith("single_label"):
        policy = {
            "codebook_align": "mean" if all_caps_ratio >= 0.70 else "none",
            "crowding_alpha": 0.5,
            "wiener_lambda": 1.0 / 3.0,
            "prior_proj_tau": 0.0,
            "prior_shift_scale": 0.0,
            "span_nms_cos": 0.0,
            "group_algebra": "none",
        }
    else:
        policy = {
            "codebook_align": "bures_wasserstein" if (non_latin_ratio >= 0.30 or compact_catalog) else "none",
            "crowding_alpha": 0.5,
            "wiener_lambda": 1.0 / 3.0,
            "prior_proj_tau": float(source_prior_tau) if (float(source_prior_tau) > 0.0 and not compact_catalog) else 0.0,
            "prior_shift_scale": 1.0 if compact_catalog else 0.0,
            "span_nms_cos": 0.80,
            "group_algebra": "frechet_clamp" if bool(has_groups) else "none",
        }
    policy["public_metadata"] = {
        "output_mode": str(mode),
        "source_count": int(source_count),
        "target_count": int(target_count),
        "compact_catalog": bool(compact_catalog),
        "all_caps_ratio": round(all_caps_ratio, 4),
        "non_latin_ratio": round(non_latin_ratio, 4),
        "has_groups": bool(has_groups),
    }
    return policy


def compile_cms_gem_crosswalk(source_codes, target_codes, gem_dir, direction="both", device=None):
    """Compile official CMS 2018 General Equivalence Mapping (GEM) matrix from ICD-9 to ICD-10.

    Supports `forward` (I9->I10 only) and `both` (union of forward I9->I10 and backward I10->I9,
    matching the standard AHRQ MapIT / Sentinel forward-backward GEM crosswalk). Returns a sparse
    or padded index/weight table mapping each target code to its CMS-equivalent source codes.
    """
    from pathlib import Path

    gdir = Path(gem_dir)
    src_index = {c.key: i for i, c in enumerate(source_codes)}
    tgt_index = {c.key: i for i, c in enumerate(target_codes)}
    edges = {i: [] for i in range(len(target_codes))}

    specs = [
        (gdir / "2018_I9gem.txt", "icd9_diag", "icd10_diag", False),
        (gdir / "gem_i9pcs.txt", "icd9_proc", "icd10_proc", False),
    ]
    if direction == "both":
        specs.extend([
            (gdir / "2018_I10gem.txt", "icd9_diag", "icd10_diag", True),
            (gdir / "gem_pcsi9.txt", "icd9_proc", "icd10_proc", True),
        ])

    seen_pairs = set()
    for path, src_sys, tgt_sys, reverse in specs:
        if not path.exists():
            raise FileNotFoundError(f"Missing CMS GEM file: {path}")
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            c1, c2, flag = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if len(flag) >= 2 and flag[1] == "1":
                continue
            s_code, t_code = (c2, c1) if reverse else (c1, c2)
            s_idx = src_index.get(f"{src_sys}:{s_code}")
            t_idx = tgt_index.get(f"{tgt_sys}:{t_code}")
            if s_idx is not None and t_idx is not None and (t_idx, s_idx) not in seen_pairs:
                seen_pairs.add((t_idx, s_idx))
                edges[t_idx].append(s_idx)

    max_deg = max((len(v) for v in edges.values()), default=0)
    k = max(1, max_deg)
    idx_tensor = torch.zeros((len(target_codes), k), dtype=torch.long)
    w_tensor = torch.zeros((len(target_codes), k), dtype=torch.float32)
    mapped = 0
    for t_idx, s_list in edges.items():
        if s_list:
            mapped += 1
            w = 1.0 / len(s_list)
            for j, s_idx in enumerate(s_list):
                idx_tensor[t_idx, j] = s_idx
                w_tensor[t_idx, j] = w
    if device is not None:
        idx_tensor = idx_tensor.to(device)
        w_tensor = w_tensor.to(device)
    summary = {
        "mode": f"cms_gem_{direction}",
        "gem_dir": str(gdir),
        "total_edges": len(seen_pairs),
        "mapped_targets": mapped,
        "total_targets": len(target_codes),
        "coverage": mapped / max(len(target_codes), 1),
        "max_in_degree": max_deg,
    }
    return idx_tensor, w_tensor, summary


def apply_span_semantic_nms(row_probs, row_spans, prototypes, codes, threshold, cos_threshold=0.80, span_window=1):
    """One-Mention-One-Code mutual exclusion: suppress lower-scoring semantic siblings grounded on the same clinical span."""
    active = torch.nonzero(row_probs >= threshold, as_tuple=False).flatten()
    n_active = int(active.numel())
    if n_active == 0:
        return []
    if n_active == 1:
        return [codes[int(active[0].item())].key]
    scores = row_probs[active]
    order = torch.argsort(scores, descending=True)
    ordered = active[order]
    spans = row_spans[ordered]
    vecs = F.normalize(prototypes[ordered].float(), dim=-1)
    # Compute conflict adjacency strictly on the tiny [n_active, n_active] subset on-device.
    span_diff = (spans.unsqueeze(1) - spans.unsqueeze(0)).abs() <= int(span_window)
    sim_conflict = (vecs @ vecs.T) >= float(cos_threshold)
    conflict = (span_diff & sim_conflict).cpu().numpy()
    ordered_list = ordered.cpu().tolist()
    kept_positions = []
    for pos in range(n_active):
        row_c = conflict[pos]
        if not any(row_c[kpos] for kpos in kept_positions):
            kept_positions.append(pos)
    return [codes[ordered_list[pos]].key for pos in kept_positions]
