import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch

from icd_transfer.codebook import Code
from icd_transfer.compare import patient_counts
from icd_transfer.evaluate_transfer import prediction_keys, transfer_data
from icd_transfer.transfer_metrics import TransferCounts, group_counts


class TransferMetricTests(unittest.TestCase):
    def test_greek_groups_agree_with_official_scorer(self):
        path = Path(__file__).resolve().parents[1] / "data/transfer/elcardiocc-2026/reference/evaluate.py"
        spec = importlib.util.spec_from_file_location("greek_official", path)
        official = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(official)
        groups = [["a", "b"], ["b", "c"], ["d"]]
        for predicted in (set(), {"b"}, {"a", "b", "x"}, {"a", "c", "d"}):
            self.assertEqual(group_counts(predicted, groups), official.compute_group_metrics_single(list(predicted), groups))
        self.assertEqual(group_counts({"b"}, groups), (2, 0, 1))

    def test_oov_truth_is_false_negative_and_enters_macro_denominator(self):
        counts = TransferCounts(["a", "b"], "multilabel")
        self.assertEqual(counts.update(["a"], {"labels": ["a", "unknown"]}), (1, 0, 1))
        result = counts.result()
        self.assertAlmostEqual(result["micro_f1"], 2 / 3)
        self.assertAlmostEqual(result["macro_f1"], 1 / 3)
        self.assertEqual(result["oov_truth_assignments"], 1)
        self.assertEqual(result["candidate_only"]["macro_f1"], 0.5)

    def test_drg_accepts_one_alias_and_argmax_uses_logits(self):
        codes = [Code("a", "ms_drg", "a", "a"), Code("b", "ms_drg", "b", "b")]
        predicted = prediction_keys(torch.tensor([90., 100.]), codes, "single_label_alternative_groups", 0.5)
        self.assertEqual(predicted, ["b"])
        counts = TransferCounts(["a", "b"], "single_label_alternative_groups")
        self.assertEqual(counts.update(predicted, {"label_groups": [["a", "b"]]}), (1, 0, 0))
        self.assertEqual(counts.result()["accuracy"], 1)
        with self.assertRaises(ValueError):
            counts.update(["a", "b"], {"label_groups": [["a", "b"]]})

    def test_missing_patient_ids_are_separate_resampling_units(self):
        path = Path(tempfile.mkdtemp()) / "predictions.jsonl"
        rows = [{"id": str(i), "subject_id": None, "tp": 1, "fp": 0, "fn": 0} for i in range(3)]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        counts, admissions = patient_counts(path)
        self.assertEqual(len(counts), 3)
        self.assertEqual(len(admissions), 3)

    def test_unapproved_test_access_is_rejected_before_read(self):
        with self.assertRaises(PermissionError):
            transfer_data("/does/not/exist", "test", False)


if __name__ == "__main__":
    unittest.main()
