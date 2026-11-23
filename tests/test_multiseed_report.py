import json
import tempfile
import unittest
from pathlib import Path

from icd_transfer.multiseed_report import (
    mean_std,
    render_multiseed_latex,
    render_multiseed_markdown,
)


class MultiSeedReportTests(unittest.TestCase):
    def test_mean_std_and_tables(self) -> None:
        st = mean_std([0.60, 0.62, 0.64])
        self.assertEqual(st["n"], 3)
        self.assertAlmostEqual(float(st["mean"]), 0.62, places=6)
        self.assertAlmostEqual(float(st["std"]), 0.02, places=6)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            s_paths = []
            t_paths = []
            for i, val in enumerate([0.61, 0.63]):
                sp = root / f"src_{i}.json"
                sp.write_text(json.dumps({
                    "milliseconds_per_note": 7.0 + i * 0.2,
                    "pure": {"freq10": {"micro_f1": val - 0.04, "macro_f1": 0.33}, "full": {"micro_f1": val - 0.05, "macro_auc_supported": 0.97}},
                    "prior_cal": {
                        "freq10": {"micro_f1": val, "macro_f1": 0.29, "micro_auc": 0.993, "macro_auc_supported": 0.973, "p_at_5": 0.78, "p_at_8": 0.70, "p_at_15": 0.53},
                        "full": {"micro_f1": val - 0.002, "macro_auc_supported": 0.971},
                    },
                }))
                s_paths.append(sp)
                tp = root / f"tgt_{i}.json"
                tp.write_text(json.dumps({
                    "output_mode": "multilabel",
                    "micro_f1": 0.32 + i * 0.004,
                    "precision": 0.39,
                    "recall": 0.27,
                    "macro_f1_supported": 0.075,
                    "code_groups": {"icd10_diag": {"micro_f1": 0.348}, "icd10_proc": {"micro_f1": 0.04}},
                    "frequency_buckets": {"1_to_9": {"micro_f1": 0.11}, "10_to_99": {"micro_f1": 0.26}, "100_plus": {"micro_f1": 0.43}},
                    "ranking": {"p_at_5": 0.43, "r_at_50": 0.52, "top5_accuracy": 0.91},
                    "milliseconds_per_note": 9.5,
                    "compile_seconds": 0.3,
                }))
                t_paths.append(tp)

            md, summary = render_multiseed_markdown(
                [("IsoToken", s_paths)],
                [("IsoToken", {"ICD-10": t_paths})],
                ["ICD-10"],
            )
            tex = render_multiseed_latex(summary, ["ICD-10"])
            self.assertIn("62.00 ± 1.41", md)
            self.assertIn("32.20 ± 0.28", md)
            self.assertIn(r"\toprule", tex)
            self.assertIn(r"62.00 $\pm$ 1.41", tex)


if __name__ == "__main__":
    unittest.main()
