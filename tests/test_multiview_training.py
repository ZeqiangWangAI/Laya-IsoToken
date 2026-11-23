import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch
import pyarrow as pa
import pyarrow.feather as feather

from icd_transfer.cli import parser
from icd_transfer.codebook import load_codebook, write_codebook
from icd_transfer.data import Notes, load_tokenizer
from icd_transfer.models import build_model
from icd_transfer.smoke import fixture
from icd_transfer.source_features import SourceFeatures, cache_source_features
from icd_transfer.source_schemas import schema_codes
from icd_transfer.train_multiview import train_multiview
from icd_transfer.evaluate_transfer import evaluate_transfer


class CachedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.root = Path(tempfile.mkdtemp(prefix="multiview-training-"))
        with contextlib.redirect_stdout(io.StringIO()):
            cls.model_dir = fixture(cls.root / "fixture")
            codes = load_codebook(cls.root / "fixture/codes.json")
            groups = [{"key": f"source_schema_term:{operator}-{partition}", "operator": operator,
                       "members": [codes[0].key, codes[-1].key], "description": "asthma oxygen",
                       "partition": partition, "family": f"{operator}-{partition}"}
                      for operator in ("OR", "AND") for partition in ("train", "selection", "audit")]
            schema = cls.root / "schemas"
            schema.mkdir()
            write_codebook(codes + schema_codes(groups), schema / "codebook.json")
            (schema / "groups.json").write_text(json.dumps(groups))
            (schema / "metadata.json").write_text(json.dumps({"role": "source_structural_training"}))
            args = parser().parse_args([
                "cache-source-features", "--train-data", str(cls.root / "fixture/train"),
                "--validation-data", str(cls.root / "fixture/validation"), "--schemas", str(schema),
                "--model-dir", str(cls.model_dir), "--output", str(cls.root / "features"),
                "--device", "cpu", "--batch-size", "2", "--label-max-len", "128"])
            cache_source_features(args)

    def train(self, name, steps, resume=None, schema_mode="or_and"):
        command = ["train-multiview", "--features", str(self.root / "features"), "--device", "cpu",
                   "--output", str(self.root / name), "--epochs", "2", "--patience", "0",
                   "--batch-size", "2", "--candidates", "3", "--schema-candidates", "2",
                   "--schema-mode", schema_mode, "--projection-dim", "16", "--lr", "0.01",
                   "--warmup-steps", "0", "--max-steps", str(steps), "--checkpoint-every", "4"]
        if resume:
            command += ["--resume", str(resume)]
        with contextlib.redirect_stdout(io.StringIO()):
            return train_multiview(parser().parse_args(command))

    def test_selection_score_matches_trained_schema_mode(self):
        none_run = self.train("mode-none", 4, schema_mode="none")
        or_run = self.train("mode-or", 4, schema_mode="or")
        full_run = self.train("mode-full", 4, schema_mode="or_and")
        self.assertAlmostEqual(none_run["best"]["selection_score"], none_run["best"]["atomic"]["micro_f1"])
        self.assertAlmostEqual(
            or_run["best"]["selection_score"],
            (or_run["best"]["atomic"]["micro_f1"] + or_run["best"]["OR"]["micro_f1"]) / 2,
        )
        self.assertAlmostEqual(
            full_run["best"]["selection_score"],
            sum(full_run["best"][k]["micro_f1"] for k in ("atomic", "OR", "AND")) / 3,
        )

    def test_cached_notes_match_direct_encoding(self):
        cached = SourceFeatures(self.root / "features", "train")
        notes = Notes(self.root / "fixture/train", "train")
        codes = load_codebook(self.root / "fixture/codes.json")
        model = build_model("multiview", self.model_dir, codes).eval()
        tokenizer = load_tokenizer(self.model_dir)
        with torch.no_grad():
            direct = model.encode_notes([notes[0]], tokenizer.pad_token_id, "cpu")[0][0]
        torch.testing.assert_close(cached.hidden(0, "cpu"), direct)

    def test_resumed_training_matches_continuous_updates(self):
        full = self.train("full", 8)
        partial = self.train("partial", 3)
        resumed = self.train("resumed", 8, self.root / "partial/step-00000003.pt")
        self.assertIsNone(partial["best"])
        self.assertEqual(full["step"], 8)
        self.assertEqual(full["best"]["threshold"], resumed["best"]["threshold"])
        self.assertEqual(full["best"]["selection_score"], resumed["best"]["selection_score"])
        a = torch.load(self.root / "full/step-00000008.pt", weights_only=False)
        b = torch.load(self.root / "resumed/step-00000008.pt", weights_only=False)
        for key in a["model"]:
            torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
        self.assertEqual(a["sampling_rngs"], b["sampling_rngs"])
        self.assertFalse(any(key.startswith("encoder.") for key in a["model"]))

    def test_target_and_test_features_are_rejected_before_read(self):
        with self.assertRaises(PermissionError):
            SourceFeatures(self.root / "missing", "test")

    def test_final_update_checkpoint_resumes_pending_validation(self):
        full = self.train("boundary-full", 8)
        resumed = self.train("boundary-resumed", 8, self.root / "boundary-full/train-step-00000008.pt")
        self.assertEqual(resumed["step"], 8)
        self.assertEqual(resumed["epoch"], 2)
        self.assertEqual(full["best"]["selection_score"], resumed["best"]["selection_score"])
        self.assertEqual(full["best"]["threshold"], resumed["best"]["threshold"])

    def test_frozen_target_evaluation_runs_all_task_modes(self):
        trained = self.train("transfer-source", 8)
        checkpoint = trained["best"]["checkpoint"]
        original = torch.load(checkpoint, weights_only=False)["model"]
        source = self.root / "fixture/validation"
        rows = feather.read_table(source / "validation.arrow").to_pylist()
        for row in rows:
            row["label_groups"] = [row["labels"]]
            row["subject_id"] = None
        for mode in ("multilabel", "alternative_label_groups", "single_label_alternative_groups"):
            target = self.root / f"target-{mode}"
            target.mkdir()
            feather.write_feather(pa.Table.from_pylist(rows), target / "dev.arrow")
            (target / "codebook.json").write_text((source / "codebook.json").read_text())
            meta = json.loads((source / "metadata.json").read_text())
            meta.update(dataset="synthetic target", role="target_evaluation_only", model_fit_allowed=False,
                        threshold_selection_allowed=False, splits={"dev": {"rows": 4}}, output_mode=mode)
            (target / "metadata.json").write_text(json.dumps(meta))
            args = parser().parse_args(["evaluate-transfer", "--data", str(target), "--split", "dev",
                                       "--checkpoint", checkpoint, "--model-dir", str(self.model_dir),
                                       "--device", "cpu", "--output", str(self.root / f"evaluation-{mode}")])
            args.validation_limit = 1 if mode == "multilabel" else 0
            with contextlib.redirect_stdout(io.StringIO()):
                result = evaluate_transfer(args)
            expected = 1 if mode == "multilabel" else 4
            self.assertEqual(result["examples"], expected)
            self.assertEqual(result["rows_without_patient_id"], expected)
            self.assertEqual(result["threshold"], trained["best"]["threshold"])
            self.assertFalse(result["model_fit_on_target"])
        after = torch.load(checkpoint, weights_only=False)["model"]
        for key in original:
            torch.testing.assert_close(original[key], after[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
