from collections import Counter

import numpy as np

from .metrics import f1_metrics


def group_counts(predicted, groups):
    groups = [set(group) for group in groups if group]
    supported = set().union(*groups)
    hits = sum(bool(predicted & group) for group in groups)
    return hits, len(predicted - supported), len(groups) - hits


class TransferCounts:
    def __init__(self, keys, mode):
        self.keys = set(keys)
        self.mode = mode
        self.tp, self.fp, self.fn = Counter(), Counter(), Counter()
        self.totals = np.zeros(3, dtype=np.int64)
        self.examples = 0
        self.example_f1 = 0.0
        self.correct = 0
        self.oov = Counter()
        self._ranking_examples = 0
        self._p_at = {5: 0.0, 8: 0.0, 15: 0.0}
        self._r_at = {15: 0.0, 50: 0.0}
        self._top_hit = {1: 0, 5: 0, 10: 0}

    def update_ranking(self, ranked_keys, row):
        self._ranking_examples += 1
        if self.mode in ("alternative_label_groups", "single_label_alternative_groups"):
            groups = [set(group) for group in row["label_groups"] if group]
            n_truth = len(groups)
            for k in (5, 8, 15):
                top = set(ranked_keys[:k])
                hits = sum(bool(top & group) for group in groups)
                self._p_at[k] += hits / k
            for k in (15, 50):
                top = set(ranked_keys[:k])
                hits = sum(bool(top & group) for group in groups)
                self._r_at[k] += hits / max(n_truth, 1)
            for k in (1, 5, 10):
                top = set(ranked_keys[:k])
                self._top_hit[k] += int(any(bool(top & group) for group in groups))
        else:
            truth = set(row["labels"])
            n_truth = len(truth)
            for k in (5, 8, 15):
                hits = sum(1 for key in ranked_keys[:k] if key in truth)
                self._p_at[k] += hits / k
            for k in (15, 50):
                hits = sum(1 for key in ranked_keys[:k] if key in truth)
                self._r_at[k] += hits / max(n_truth, 1)
            for k in (1, 5, 10):
                self._top_hit[k] += int(any(key in truth for key in ranked_keys[:k]))

    def update(self, predicted, row):
        predicted = set(predicted)
        if predicted - self.keys:
            raise ValueError("Predictions must belong to the compiled candidate catalog")
        if self.mode.startswith("single_label") and len(predicted) != 1:
            raise ValueError("DRG requires exactly one predicted label")
        if self.mode in ("alternative_label_groups", "single_label_alternative_groups"):
            groups = row["label_groups"]
            if self.mode.startswith("single_label") and len(groups) != 1:
                raise ValueError("A DRG admission needs one acceptable-label group")
            counts = group_counts(predicted, groups)
            self.oov.update(key for group in groups for key in set(group) - self.keys)
        else:
            truth = set(row["labels"])
            self.tp.update(predicted & truth)
            self.fp.update(predicted - truth)
            self.fn.update(truth - predicted)
            self.oov.update(truth - self.keys)
            counts = len(predicted & truth), len(predicted - truth), len(truth - predicted)
        self.totals += counts
        tp, fp, fn = counts
        self.examples += 1
        self.example_f1 += 2 * tp / max(2 * tp + fp + fn, 1)
        self.correct += int(tp == 1 and fp == 0 and fn == 0)
        return counts

    def label_result(self, keys):
        return f1_metrics(*(np.array([counts[key] for key in keys], dtype=np.int64)
                            for counts in (self.tp, self.fp, self.fn)))

    def result(self):
        tp, fp, fn = map(int, self.totals)
        result = {"examples": self.examples, "tp": tp, "fp": fp, "fn": fn,
                  "micro_f1": 2 * tp / max(2 * tp + fp + fn, 1),
                  "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                  "example_macro_f1": self.example_f1 / max(self.examples, 1),
                  "candidate_labels": len(self.keys), "oov_truth_labels": len(self.oov),
                  "oov_truth_assignments": sum(self.oov.values())}
        if self.mode.startswith("single_label"):
            result["accuracy"] = self.correct / max(self.examples, 1)
        if self._ranking_examples > 0:
            n = self._ranking_examples
            result["ranking"] = {
                "p_at_5": self._p_at[5] / n,
                "p_at_8": self._p_at[8] / n,
                "p_at_15": self._p_at[15] / n,
                "r_at_15": self._r_at[15] / n,
                "r_at_50": self._r_at[50] / n,
                "top1_accuracy": self._top_hit[1] / n,
                "top5_accuracy": self._top_hit[5] / n,
                "top10_accuracy": self._top_hit[10] / n,
            }
        if self.mode in ("alternative_label_groups", "single_label_alternative_groups"):
            result["macro_f1"] = None
            result["macro_f1_definition"] = "Per-code F1 is undefined for alternative gold labels; report example macro-F1"
            return result
        universe = sorted(self.keys | self.oov.keys())
        result["macro_f1"] = self.label_result(universe)["macro_f1"]
        result["macro_f1_definition"] = "Mean over public candidates plus observed OOV truth labels"
        result["candidate_only"] = self.label_result(sorted(self.keys))
        rare = [key for key in universe if 0 < self.tp[key] + self.fn[key] < 10]
        result["evaluation_support_1_to_9"] = self.label_result(rare)
        return result
