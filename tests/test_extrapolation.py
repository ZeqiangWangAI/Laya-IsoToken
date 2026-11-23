import json
from pathlib import Path
import tempfile
import unittest

import torch

from icd_transfer.extrapolation import (
    analyze_extrapolation_routing,
    per_code_error_table,
    quartile_stratification,
    selective_escalation_curve,
)
from icd_transfer.report_stage import format_markdown_summary, summarize_runs


class ExtrapolationRoutingTests(unittest.TestCase):
    def test_stratification_and_escalation_capture_high_rho_errors(self):
        # 8 codes: low rho (0.1..0.4) have perfect predictions; high rho (0.6..0.9) fail
        keys = [f"c{i}" for i in range(8)]
        residual = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        preds = [
            {"predicted": ["c0", "c1", "c2", "c3"], "truth": ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7"]},
            {"predicted": ["c0", "c1", "c2", "c3", "c7"], "truth": ["c0", "c1", "c2", "c3", "c4", "c5", "c6"]},
        ]
        rows = per_code_error_table(keys, residual, preds)
        strat = quartile_stratification(rows, n_quantiles=2)
        self.assertEqual(strat["supported_codes"], 8)
        self.assertLess(strat["spearman_rho_vs_f1"], -0.8)
        self.assertGreater(strat["quantiles"][0]["micro_f1"], strat["quantiles"][1]["micro_f1"])

        curve = selective_escalation_curve(rows, budgets=(0.0, 0.5, 1.0))
        # At 50% budget (top 4 rho codes), 100% of errors are captured!
        self.assertAlmostEqual(curve[1]["error_capture_rate"], 1.0, places=5)
        self.assertAlmostEqual(curve[1]["system2_micro_f1"], 1.0, places=5)
        self.assertGreater(curve[1]["system2_micro_f1"], curve[1]["random_micro_f1"])

    def test_analyze_eval_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            torch.save(
                {"keys": ["a", "b"], "crowding": torch.tensor([0.2, 0.3]), "residual": torch.tensor([0.1, 0.9])},
                root / "geometry.pt",
            )
            (root / "metrics.json").write_text(json.dumps({"dataset": "test", "split": "dev", "micro_f1": 0.5}))
            (root / "predictions.jsonl").write_text(json.dumps({"predicted": ["a"], "truth": ["a", "b"]}) + "\n")
            report = analyze_extrapolation_routing(root, budgets=(0.0, 0.5))
            self.assertEqual(report["dataset"], "test")
            self.assertAlmostEqual(report["escalation_curve"][1]["error_capture_rate"], 1.0, places=5)

    def test_label_groups_do_not_double_count_aliases(self):
        keys = ["d1:v25", "d1:v33", "d2:v33"]
        residual = torch.tensor([0.1, 0.2, 0.8])
        preds = [
            # Exact hit on alias v25: must yield total tp=1, fp=0, fn=0 (no fn on v33!)
            {"predicted": ["d1:v25"], "truth": ["d1:v25", "d1:v33"], "label_groups": [["d1:v25", "d1:v33"]]},
            # Wrong prediction d2:v33 for group [d1:v25, d1:v33]: total tp=0, fp=1, fn=1
            {"predicted": ["d2:v33"], "truth": ["d1:v25", "d1:v33"], "label_groups": [["d1:v25", "d1:v33"]]},
        ]
        rows = per_code_error_table(keys, residual, preds)
        self.assertAlmostEqual(sum(r["tp"] for r in rows), 1.0, places=6)
        self.assertAlmostEqual(sum(r["fp"] for r in rows), 1.0, places=6)
        self.assertAlmostEqual(sum(r["fn"] for r in rows), 1.0, places=6)

    def test_report_stage_summarizes_run_and_formats_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run-demo"
            (run / "train").mkdir(parents=True)
            (run / "eval" / "demo-dev").mkdir(parents=True)
            (run / "train" / "arguments.json").write_text(json.dumps({"head_mode": "isometric", "views": "all"}))
            (run / "train" / "summary.json").write_text(json.dumps({"best": {"step": 10, "score": 0.35}}))
            (run / "train" / "validation-00000010.json").write_text(
                json.dumps({"threshold": 0.2, "atomic": {"micro_f1": 0.35, "macro_f1": 0.03, "macro_f1_supported": 0.05}})
            )
            (run / "eval" / "demo-dev" / "metrics.json").write_text(
                json.dumps({"dataset": "demo", "split": "dev", "micro_f1": 0.25, "precision": 0.3, "recall": 0.21, "fp": 5, "milliseconds_per_note": 12.3})
            )
            torch.save(
                {"keys": ["a", "b"], "crowding": torch.tensor([0.1, 0.2]), "residual": torch.tensor([0.1, 0.9])},
                run / "eval" / "demo-dev" / "geometry.pt",
            )
            (run / "eval" / "demo-dev" / "predictions.jsonl").write_text(json.dumps({"predicted": ["a"], "truth": ["a", "b"]}) + "\n")
            reports = summarize_runs([run])
            md = format_markdown_summary(reports)
            self.assertIn("run-demo", md)
            self.assertIn("isometric", md)
            self.assertIn("0.2500", md)


if __name__ == "__main__":
    unittest.main()
