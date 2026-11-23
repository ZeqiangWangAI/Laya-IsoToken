import json
import math
from pathlib import Path

import torch


def load_eval_artifacts(eval_dir):
    root = Path(eval_dir)
    geometry = torch.load(root / "geometry.pt", map_location="cpu", weights_only=False)
    predictions = []
    with (root / "predictions.jsonl").open() as stream:
        for line in stream:
            if line.strip():
                predictions.append(json.loads(line))
    metrics = json.loads((root / "metrics.json").read_text())
    return metrics, geometry, predictions


def _f1(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return (2.0 * tp / denom) if denom > 0 else 0.0


def _spearman_rank_corr(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    def ranks(vals):
        order = sorted(range(n), key=lambda i: vals[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i + 1
            while j < n and vals[order[j]] == vals[order[i]]:
                j += 1
            avg = 0.5 * (i + j - 1)
            for k in range(i, j):
                out[order[k]] = avg
            i = j
        return out
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den_x = math.sqrt(sum((a - mx) ** 2 for a in rx))
    den_y = math.sqrt(sum((b - my) ** 2 for b in ry))
    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    return num / (den_x * den_y)


def per_code_error_table(keys, residual, predictions):
    """Aggregate per-code (tp, fp, fn) across predictions and attach compile-time rho_b."""
    res_list = residual.detach().cpu().tolist()
    index = {k: i for i, k in enumerate(keys)}
    tp = [0.0] * len(keys)
    fp = [0.0] * len(keys)
    fn = [0.0] * len(keys)
    for row in predictions:
        pred_set = set(row["predicted"])
        groups = [set(g) for g in row.get("label_groups", []) if g]
        if groups:
            supported = set().union(*groups)
            for k in pred_set - supported:
                if k in index:
                    fp[index[k]] += 1.0
            for group in groups:
                hits = [k for k in (pred_set & group) if k in index]
                if hits:
                    w = 1.0 / len(hits)
                    for k in hits:
                        tp[index[k]] += w
                else:
                    members = [k for k in group if k in index]
                    if members:
                        w = 1.0 / len(members)
                        for k in members:
                            fn[index[k]] += w
        else:
            truth_set = set(row["truth"])
            for k in pred_set & truth_set:
                if k in index:
                    tp[index[k]] += 1.0
            for k in pred_set - truth_set:
                if k in index:
                    fp[index[k]] += 1.0
            for k in truth_set - pred_set:
                if k in index:
                    fn[index[k]] += 1.0
    rows = []
    for i, key in enumerate(keys):
        support = tp[i] + fn[i]
        errors = fp[i] + fn[i]
        rows.append({
            "key": key,
            "rho": float(res_list[i]),
            "tp": tp[i],
            "fp": fp[i],
            "fn": fn[i],
            "support": support,
            "errors": errors,
            "f1": _f1(tp[i], fp[i], fn[i]),
        })
    return rows


def quartile_stratification(code_rows, n_quantiles=4):
    """Stratify supported target codes into equal-count rho_b quantiles."""
    supported = [r for r in code_rows if r["support"] > 0]
    if not supported:
        return {"supported_codes": 0, "spearman_rho_vs_f1": 0.0, "quantiles": []}
    ordered = sorted(supported, key=lambda r: r["rho"])
    n = len(ordered)
    quantiles = []
    for q in range(n_quantiles):
        start = (q * n) // n_quantiles
        stop = ((q + 1) * n) // n_quantiles
        chunk = ordered[start:stop]
        if not chunk:
            continue
        tp = sum(r["tp"] for r in chunk)
        fp = sum(r["fp"] for r in chunk)
        fn = sum(r["fn"] for r in chunk)
        quantiles.append({
            "quantile": q + 1,
            "codes": len(chunk),
            "rho_min": chunk[0]["rho"],
            "rho_max": chunk[-1]["rho"],
            "rho_mean": sum(r["rho"] for r in chunk) / len(chunk),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "micro_f1": _f1(tp, fp, fn),
            "macro_f1": sum(r["f1"] for r in chunk) / len(chunk),
        })
    corr = _spearman_rank_corr([r["rho"] for r in supported], [r["f1"] for r in supported])
    return {
        "supported_codes": n,
        "spearman_rho_vs_f1": float(corr),
        "quantiles": quantiles,
    }


def selective_escalation_curve(code_rows, budgets=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5)):
    """Evaluate compile-time code escalation ordered by descending rho_b vs random expectation."""
    total_codes = len(code_rows)
    total_tp = sum(r["tp"] for r in code_rows)
    total_fp = sum(r["fp"] for r in code_rows)
    total_fn = sum(r["fn"] for r in code_rows)
    total_errors = total_fp + total_fn
    if total_codes == 0:
        return []
    ordered = sorted(code_rows, key=lambda r: r["rho"], reverse=True)
    curve = []
    for budget in budgets:
        k = int(round(budget * total_codes))
        escalated = ordered[:k]
        caught_fp = sum(r["fp"] for r in escalated)
        caught_fn = sum(r["fn"] for r in escalated)
        caught_errors = caught_fp + caught_fn
        # Under oracle System-2 verification on escalated codes: caught_fp -> 0, caught_fn -> tp
        s2_tp = total_tp + caught_fn
        s2_fp = total_fp - caught_fp
        s2_fn = total_fn - caught_fn
        # Random baseline expectation at budget b: resolves b * errors
        rand_tp = total_tp + budget * total_fn
        rand_fp = total_fp * (1.0 - budget)
        rand_fn = total_fn * (1.0 - budget)
        curve.append({
            "budget": float(budget),
            "escalated_codes": k,
            "rho_threshold": ordered[k - 1]["rho"] if k > 0 else 1.0,
            "error_capture_rate": (caught_errors / total_errors) if total_errors > 0 else 1.0,
            "random_capture_rate": float(budget),
            "system2_micro_f1": _f1(s2_tp, s2_fp, s2_fn),
            "random_micro_f1": _f1(rand_tp, rand_fp, rand_fn),
        })
    return curve


def analyze_extrapolation_routing(eval_dir, budgets=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5), n_quantiles=4):
    metrics, geometry, predictions = load_eval_artifacts(eval_dir)
    rows = per_code_error_table(geometry["keys"], geometry["residual"], predictions)
    return {
        "dataset": metrics.get("dataset"),
        "split": metrics.get("split"),
        "head_mode": metrics.get("head_mode"),
        "base_micro_f1": metrics.get("micro_f1"),
        "stratification": quartile_stratification(rows, n_quantiles=n_quantiles),
        "escalation_curve": selective_escalation_curve(rows, budgets=budgets),
    }
