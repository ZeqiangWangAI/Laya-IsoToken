import json
from pathlib import Path

import numpy as np


def patient_counts(path):
    patients, admissions = {}, set()
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            key = (row["subject_id"], row["id"])
            if key in admissions:
                raise ValueError("Duplicate prediction admission")
            admissions.add(key)
            unit = ("document", row["id"]) if row["subject_id"] is None else ("patient", row["subject_id"])
            patients.setdefault(unit, np.zeros(3, dtype=np.int64))
            patients[unit] += [row["tp"], row["fp"], row["fn"]]
    return patients, admissions


def micro(counts):
    tp, fp, fn = counts
    return 2 * tp / max(2 * tp + fp + fn, 1)


def unit_f1(counts_matrix):
    tp = counts_matrix[:, 0].astype(np.float64)
    fp = counts_matrix[:, 1].astype(np.float64)
    fn = counts_matrix[:, 2].astype(np.float64)
    denom = np.maximum(2.0 * tp + fp + fn, 1.0)
    return 2.0 * tp / denom


def paired_effect_sizes(a, b):
    """Compute standardized paired effect sizes across patient/document units."""
    diff = unit_f1(a) - unit_f1(b)
    mean_d = float(diff.mean())
    sd_d = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    cohens_dz = float(mean_d / sd_d) if sd_d > 1e-12 else 0.0
    wins = int((diff > 1e-12).sum())
    losses = int((diff < -1e-12).sum())
    ties = int(len(diff) - wins - losses)
    non_ties = max(wins + losses, 1)
    return {
        "unit_f1_mean_difference": mean_d,
        "unit_f1_sd_difference": sd_d,
        "cohens_dz": cohens_dz,
        "cliffs_delta_paired": float((wins - losses) / max(len(diff), 1)),
        "rank_biserial": float((wins - losses) / non_ties),
        "win_rate_non_ties": float(wins / non_ties),
        "wins": wins,
        "losses": losses,
        "ties": ties,
    }


def compare(args):
    left, left_ids = patient_counts(args.method)
    right, right_ids = patient_counts(args.baseline)
    if left_ids != right_ids:
        raise ValueError("Paired comparison requires exactly the same admissions")
    keys = sorted(left)
    a, b = np.stack([left[key] for key in keys]), np.stack([right[key] for key in keys])
    rng = np.random.default_rng(args.seed)
    deltas = []
    for _ in range(args.replicates):
        indices = rng.integers(len(keys), size=len(keys))
        deltas.append(micro(a[indices].sum(0)) - micro(b[indices].sum(0)))
    deltas_arr = np.asarray(deltas, dtype=np.float64)
    obs = micro(a.sum(0)) - micro(b.sum(0))
    if obs >= 0:
        tail = int((deltas_arr <= 0).sum())
    else:
        tail = int((deltas_arr >= 0).sum())
    p_two_sided = float(min(1.0, 2.0 * (tail + 1) / (len(deltas_arr) + 1)))
    result = {"patients": sum(key[0] == "patient" for key in keys), "resampling_units": len(keys),
              "resampling_policy": "patient when available; otherwise document",
              "micro_f1_difference": obs,
              "patient_bootstrap_95_ci": np.quantile(deltas_arr, [0.025, 0.975]).tolist(),
              "p_value_two_sided": p_two_sided,
              "effect_size": paired_effect_sizes(a, b),
              "replicates": args.replicates, "seed": args.seed}
    with Path(args.output).open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))

