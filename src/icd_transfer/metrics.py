import numpy as np


def f1_metrics(tp, fp, fn):
    denominator = 2 * tp + fp + fn
    per_label = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0)
    supported = tp + fn > 0
    return {
        "micro_f1": float(2 * tp.sum() / max(denominator.sum(), 1)),
        "micro_f1_supported": float(2 * tp[supported].sum() / max(denominator[supported].sum(), 1)),
        "macro_f1": float(per_label.mean()) if len(per_label) else None,
        "macro_f1_supported": float(per_label[supported].mean()) if supported.any() else None,
        "precision": float(tp.sum() / max((tp + fp).sum(), 1)),
        "recall": float(tp.sum() / max((tp + fn).sum(), 1)),
        "labels": len(tp), "supported_labels": int(supported.sum()),
    }


class Counts:
    def __init__(self, thresholds, num_codes):
        self.thresholds = np.asarray(thresholds)
        self.tp = np.zeros((len(thresholds), num_codes), dtype=np.int64)
        self.fp = np.zeros_like(self.tp)
        self.fn = np.zeros_like(self.tp)

    def update(self, probabilities, truth):
        predicted = probabilities[None, :, :] >= self.thresholds[:, None, None]
        truth = truth.astype(bool)[None, :, :]
        self.tp += (predicted & truth).sum(1)
        self.fp += (predicted & ~truth).sum(1)
        self.fn += (~predicted & truth).sum(1)

    def result(self, index, indices=None):
        selected = slice(None) if indices is None else indices
        return f1_metrics(self.tp[index, selected], self.fp[index, selected], self.fn[index, selected])

    def best(self):
        scores = [self.result(index)["micro_f1"] for index in range(len(self.thresholds))]
        return max(range(len(scores)), key=lambda i: (scores[i], -abs(self.thresholds[i] - 0.5)))


def binary_roc_auc_exact(scores, targets):
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.bool_)
    positives = int(targets.sum())
    negatives = int(len(targets) - positives)
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_targets = targets[order]
    # Tie-aware average 1-based ranks via run-length boundaries.
    changes = np.flatnonzero(np.diff(sorted_scores) != 0) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [len(sorted_scores)]))
    avg_ranks = 0.5 * (starts + 1 + ends)
    ranks = np.repeat(avg_ranks, ends - starts)
    pos_rank_sum = float(ranks[sorted_targets].sum())
    return float((pos_rank_sum - 0.5 * positives * (positives + 1)) / (positives * negatives))


def histogram_roc_auc(pos_hist, neg_hist):
    pos = np.asarray(pos_hist, dtype=np.float64)
    neg = np.asarray(neg_hist, dtype=np.float64)
    total_pos = float(pos.sum())
    total_neg = float(neg.sum())
    if total_pos <= 0.0 or total_neg <= 0.0:
        return None
    neg_below = np.cumsum(neg) - neg
    concordant = float(np.sum(pos * (neg_below + 0.5 * neg)))
    return float(concordant / (total_pos * total_neg))


def histogram_auprc(pos_hist, neg_hist):
    """Streaming average precision (Micro-AUPRC) from high-resolution score histograms."""
    pos = np.asarray(pos_hist, dtype=np.float64)[::-1]
    neg = np.asarray(neg_hist, dtype=np.float64)[::-1]
    total_pos = float(pos.sum())
    if total_pos <= 0.0:
        return None
    cum_tp = np.cumsum(pos)
    cum_fp = np.cumsum(neg)
    nz = pos > 0
    prec = cum_tp[nz] / np.maximum(cum_tp[nz] + cum_fp[nz], 1.0)
    return float(np.sum(prec * (pos[nz] / total_pos)))


def binary_auprc_exact(scores, targets):
    """Exact tie-aware Average Precision for one binary label."""
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.bool_)
    positives = int(targets.sum())
    if positives == 0 or positives == len(targets):
        return None
    order = np.argsort(-scores, kind="mergesort")
    s_sorted = scores[order]
    y_sorted = targets[order].astype(np.float64)
    changes = np.flatnonzero(np.diff(s_sorted) != 0) + 1
    ends = np.concatenate((changes, [len(s_sorted)]))
    starts = np.concatenate(([0], changes))
    tp_group = np.add.reduceat(y_sorted, starts)
    cnt_group = (ends - starts).astype(np.float64)
    cum_tp = np.cumsum(tp_group)
    cum_all = np.cumsum(cnt_group)
    nz = tp_group > 0
    return float(np.sum((cum_tp[nz] / cum_all[nz]) * (tp_group[nz] / positives)))


class CalibrationTracker:
    """Streaming 15-bin ECE, Brier score, and reliability diagram tracker for large binary label matrices."""

    def __init__(self, n_bins=15):
        self.n_bins = int(n_bins)
        self.bin_counts = np.zeros(self.n_bins, dtype=np.int64)
        self.bin_prob_sum = np.zeros(self.n_bins, dtype=np.float64)
        self.bin_pos_sum = np.zeros(self.n_bins, dtype=np.float64)
        self.brier_sum = 0.0
        self.total = 0
        self.pos_total = 0
        # Also track calibration restricted to predicted/non-trivial probabilities (p >= 0.05) for clinical audit.
        self.active_counts = np.zeros(self.n_bins, dtype=np.int64)
        self.active_prob_sum = np.zeros(self.n_bins, dtype=np.float64)
        self.active_pos_sum = np.zeros(self.n_bins, dtype=np.float64)
        self.active_brier_sum = 0.0
        self.active_total = 0

    def update(self, probs, targets):
        p = np.asarray(probs, dtype=np.float64).ravel()
        y = np.asarray(targets, dtype=np.float64).ravel()
        if p.size == 0:
            return
        idx = np.clip((p * self.n_bins).astype(np.int64), 0, self.n_bins - 1)
        self.bin_counts += np.bincount(idx, minlength=self.n_bins)
        self.bin_prob_sum += np.bincount(idx, weights=p, minlength=self.n_bins)
        self.bin_pos_sum += np.bincount(idx, weights=y, minlength=self.n_bins)
        diff = p - y
        self.brier_sum += float(np.dot(diff, diff))
        self.total += int(p.size)
        self.pos_total += int(y.sum())

        mask = p >= 0.05
        if mask.any():
            pa = p[mask]
            ya = y[mask]
            ia = idx[mask]
            self.active_counts += np.bincount(ia, minlength=self.n_bins)
            self.active_prob_sum += np.bincount(ia, weights=pa, minlength=self.n_bins)
            self.active_pos_sum += np.bincount(ia, weights=ya, minlength=self.n_bins)
            da = pa - ya
            self.active_brier_sum += float(np.dot(da, da))
            self.active_total += int(pa.size)

    def result(self):
        if self.total == 0:
            return {"ece": None, "brier": None, "active_ece_p05": None, "active_brier_p05": None, "bins": []}
        ece = 0.0
        bins = []
        for b in range(self.n_bins):
            cnt = int(self.bin_counts[b])
            lo = b / self.n_bins
            hi = (b + 1) / self.n_bins
            if cnt > 0:
                avg_p = float(self.bin_prob_sum[b] / cnt)
                emp_y = float(self.bin_pos_sum[b] / cnt)
                ece += (cnt / self.total) * abs(avg_p - emp_y)
            else:
                avg_p = None
                emp_y = None
            bins.append({"bin_lo": round(lo, 4), "bin_hi": round(hi, 4), "count": cnt, "avg_conf": avg_p, "emp_acc": emp_y})
        active_ece = None
        active_brier = None
        if self.active_total > 0:
            active_ece = 0.0
            for b in range(self.n_bins):
                cnt = int(self.active_counts[b])
                if cnt > 0:
                    active_ece += (cnt / self.active_total) * abs(
                        self.active_prob_sum[b] / cnt - self.active_pos_sum[b] / cnt
                    )
            active_brier = float(self.active_brier_sum / self.active_total)
        return {
            "ece": float(ece),
            "brier": float(self.brier_sum / self.total),
            "active_ece_p05": active_ece,
            "active_brier_p05": active_brier,
            "active_count_p05": int(self.active_total),
            "total_predictions": int(self.total),
            "positive_rate": float(self.pos_total / self.total),
            "bins": bins,
        }



