import contextlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather
import torch

from icd_transfer.cli import parser
from icd_transfer.codebook import Code, load_codebook, write_codebook
from icd_transfer.evaluate_transfer import evaluate_transfer
from icd_transfer.label_views import LabelViews
from icd_transfer.models import AlignedMultiViewHead
from icd_transfer.sampling import sample_candidates
from icd_transfer.smoke import fixture
from icd_transfer.source_features import cache_source_features
from icd_transfer.source_schemas import schema_codes
from icd_transfer.train_multiview import train_multiview


class AlignedHeadAndSamplingTests(unittest.TestCase):
    def test_aligned_score_batch_matches_single_note_score_across_masks_and_views(self):
        torch.manual_seed(11)
        head = AlignedMultiViewHead(width=16, projection_dim=8).eval()
        notes = [torch.randn(3, 16), torch.randn(5, 16), torch.randn(1, 16)]
        vectors = torch.randn(7, 3, 16)
        available = torch.tensor([
            [True, True, False],
            [True, False, False],
            [False, False, True],
            [True, True, True],
            [True, False, True],
            [True, True, False],
            [False, False, True],
        ])
        operators = torch.tensor([0, 0, 1, 2, 1, 0, 2], dtype=torch.long)
        labels = LabelViews(vectors, available, operators)
        batched = head.score_batch(notes, labels)
        single = torch.stack([head.score(note, labels) for note in notes])
        torch.testing.assert_close(batched, single, atol=1e-5, rtol=1e-5)

    def test_aligned_head_has_bounded_note_aligned_logits_without_label_bias(self):
        torch.manual_seed(3)
        head = AlignedMultiViewHead(width=12, projection_dim=6).eval()
        labels = LabelViews(
            torch.randn(10, 3, 12),
            torch.tensor([[True, False, False]] * 10),
            torch.zeros(10, dtype=torch.long),
        )
        # Flipping the sign of a single-chunk note flips the cosine term around the global scalar bias.
        note = torch.randn(1, 12)
        pos = head.score(note, labels)
        neg = head.score(-note, labels)
        scale = head.log_scale.exp().item()
        bias = head.bias.item()
        self.assertTrue(((pos >= bias - scale - 1e-4) & (pos <= bias + scale + 1e-4)).all())
        self.assertTrue(((neg >= bias - scale - 1e-4) & (neg <= bias + scale + 1e-4)).all())

    def test_sample_candidates_full_mode_and_min_negatives(self):
        codes = [Code(f"d:{i}", "icd9_diag", str(i), f"condition {i}") for i in range(10)]
        rows = [{"labels": [codes[0].key, codes[1].key, codes[2].key, codes[3].key]}]
        full_idx, full_w = sample_candidates(rows, codes, 0, random.Random(0), "cpu")
        self.assertEqual(full_idx, list(range(10)))
        torch.testing.assert_close(full_w, torch.ones(10))

        # When positives (4) exceed limit (3), min_negatives prevents collapse to 1 negative.
        idx, w = sample_candidates(rows, codes, 3, random.Random(0), "cpu", min_negatives=4)
        self.assertEqual(len(idx), 8)
        neg_weights = [float(weight) for i, weight in zip(idx, w) if i >= 4]
        self.assertEqual(len(neg_weights), 4)
        for weight in neg_weights:
            self.assertAlmostEqual(weight, 6 / 4, places=5)


class AlignedEndToEndTests(unittest.TestCase):
    def test_aligned_full_codebook_train_resume_and_transfer(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(prefix="aligned-e2e-") as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                model_dir = fixture(root / "fixture")
                codes = load_codebook(root / "fixture/codes.json")
                groups = [{"key": f"source_schema_term:{op}-{part}", "operator": op,
                           "members": [codes[0].key, codes[-1].key], "description": "asthma oxygen",
                           "partition": part, "family": f"{op}-{part}"}
                          for op in ("OR", "AND") for part in ("train", "selection", "audit")]
                schema = root / "schemas"
                schema.mkdir()
                write_codebook(codes + schema_codes(groups), schema / "codebook.json")
                (schema / "groups.json").write_text(json.dumps(groups))
                (schema / "metadata.json").write_text(json.dumps({"role": "source_structural_training"}))
                cache_source_features(parser().parse_args([
                    "cache-source-features", "--train-data", str(root / "fixture/train"),
                    "--validation-data", str(root / "fixture/validation"), "--schemas", str(schema),
                    "--model-dir", str(model_dir), "--output", str(root / "features"),
                    "--device", "cpu", "--batch-size", "2", "--label-max-len", "128"]))
                first = train_multiview(parser().parse_args([
                    "train-multiview", "--features", str(root / "features"), "--device", "cpu",
                    "--output", str(root / "run1"), "--head-mode", "aligned", "--candidates", "0",
                    "--schema-mode", "none", "--epochs", "2", "--patience", "0", "--batch-size", "2",
                    "--projection-dim", "16", "--lr", "0.01", "--warmup-steps", "0",
                    "--checkpoint-every", "0", "--max-steps", "2"]))
                resumed = train_multiview(parser().parse_args([
                    "train-multiview", "--features", str(root / "features"), "--device", "cpu",
                    "--output", str(root / "run2"), "--resume", str(root / "run1/step-00000002.pt"),
                    "--head-mode", "aligned", "--candidates", "0", "--schema-mode", "none",
                    "--epochs", "2", "--patience", "0", "--batch-size", "2",
                    "--projection-dim", "16", "--lr", "0.01", "--warmup-steps", "0",
                    "--checkpoint-every", "0"]))
                source = root / "fixture/validation"
                rows = feather.read_table(source / "validation.arrow").to_pylist()
                for row in rows:
                    row["label_groups"] = [row["labels"]]
                    row["subject_id"] = None
                target = root / "target"
                target.mkdir()
                feather.write_feather(pa.Table.from_pylist(rows), target / "dev.arrow")
                (target / "codebook.json").write_text((source / "codebook.json").read_text())
                meta = json.loads((source / "metadata.json").read_text())
                meta.update(dataset="synthetic target", role="target_evaluation_only", model_fit_allowed=False,
                            threshold_selection_allowed=False, splits={"dev": {"rows": len(rows)}},
                            output_mode="multilabel")
                (target / "metadata.json").write_text(json.dumps(meta))
                metrics = evaluate_transfer(parser().parse_args([
                    "evaluate-transfer", "--data", str(target), "--split", "dev",
                    "--checkpoint", resumed["best"]["checkpoint"], "--model-dir", str(model_dir),
                    "--device", "cpu", "--prior-shift-scale", "1.0", "--crowding-alpha", "0.5",
                    "--crowding-k", "2", "--record-geometry", "--output", str(root / "eval")]))
            self.assertEqual(first["step"], 2)
            self.assertEqual(metrics["head_mode"], "aligned")
            self.assertTrue(metrics["calibration"]["geometry_computed"])


if __name__ == "__main__":
    unittest.main()
