import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather
import torch

from icd_transfer.calibration import (
    apply_codebook_alignment,
    apply_whitener,
    cardinality_prior_shift,
    codebook_crowding,
    compile_calibration,
    extrapolation_residual,
    fit_codebook_alignment,
    fit_whitener,
    project_out_note_shift,
)
from icd_transfer.label_views import LabelViews
from icd_transfer.cli import parser
from icd_transfer.codebook import load_codebook, write_codebook
from icd_transfer.evaluate_transfer import evaluate_transfer
from icd_transfer.smoke import fixture
from icd_transfer.source_features import cache_source_features
from icd_transfer.source_schemas import schema_codes
from icd_transfer.train_multiview import train_multiview


class CalibrationMathTests(unittest.TestCase):
    def test_zca_whitener_equalizes_covariance_under_autocast(self):
        torch.manual_seed(7)
        base = torch.randn(400, 6)
        anisotropic = base @ torch.diag(torch.tensor([10.0, 5.0, 2.0, 0.5, 0.2, 0.1])) + 3.0
        with torch.autocast("cpu", dtype=torch.bfloat16):
            mean, transform = fit_whitener(anisotropic, eps=1e-6)
            whitened = (anisotropic.double() - mean) @ transform
        cov = (whitened.T @ whitened) / (len(whitened) - 1)
        torch.testing.assert_close(cov, torch.eye(6, dtype=torch.float64), atol=1e-3, rtol=1e-3)
        unit = apply_whitener(anisotropic, mean, transform)
        torch.testing.assert_close(unit.norm(dim=-1), torch.ones(len(anisotropic)), atol=1e-5, rtol=1e-5)

    def test_codebook_crowding_excludes_self_and_matches_batch_sizes(self):
        vectors = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.99, 0.141067, 0.0, 0.0],
            [0.98, -0.198997, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        full = codebook_crowding(vectors, k=2, batch_size=16)
        batched = codebook_crowding(vectors, k=2, batch_size=2)
        torch.testing.assert_close(full, batched)
        self.assertGreater(full[0].item(), 0.95)
        self.assertAlmostEqual(full[3].item(), 0.0, places=5)
        self.assertAlmostEqual(full[4].item(), 0.0, places=5)

    def test_extrapolation_residual_matches_svd_formula_and_bounds(self):
        source = torch.zeros(20, 4)
        source[:10, 0] = 1.0
        source[10:, 1] = 1.0
        target = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0],
        ])
        ridge = 1.0
        # Normalized Gram G_A = (d / N_A) Z_A^T Z_A = (4 / 20) * diag(10, 10, 0, 0) = diag(2, 2, 0, 0)
        rho = extrapolation_residual(source, target, ridge=ridge)
        expected = torch.tensor([
            math.sqrt(ridge / (2.0 + ridge)),
            1.0,
            math.sqrt(0.5 * (ridge / (2.0 + ridge)) + 0.5 * 1.0),
        ])
        torch.testing.assert_close(rho, expected, atol=1e-5, rtol=1e-5)
        # Verify invariance to source codebook size N_A (10x replication produces identical rho)
        rho_large = extrapolation_residual(source.repeat(10, 1), target, ridge=ridge)
        torch.testing.assert_close(rho_large, rho, atol=1e-5, rtol=1e-5)

    def test_compile_calibration_preserves_anisotropy_in_extrapolation_residual(self):
        torch.manual_seed(11)
        # Anisotropic 2D clinical subspace embedded in 8D space (with tiny noise in dims 2..7)
        source = torch.zeros(200, 8)
        source[:, :2] = torch.randn(200, 2)
        source[:, 2:] = 1e-3 * torch.randn(200, 6)
        in_span = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        out_span = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        target = torch.cat((in_span, out_span), dim=0)
        cal = compile_calibration(
            source, target, "multilabel", source_count=200, target_count=2,
            crowding_alpha=0.5, crowding_k=5, record_geometry=True,
        )
        # In raw centered space, in-span has low rho (~0.4) while orthogonal out-of-span has rho ~ 1.0
        self.assertLess(cal.residual[0].item(), 0.6)
        self.assertGreater(cal.residual[1].item(), 0.95)

    def test_cardinality_prior_shift_rules(self):
        self.assertAlmostEqual(cardinality_prior_shift(11324, 11324, "multilabel", 1.0), 0.0)
        self.assertAlmostEqual(
            cardinality_prior_shift(11324, 26085, "multilabel", 1.0),
            -math.log(26085 / 11324),
        )
        self.assertEqual(cardinality_prior_shift(11324, 1581, "single_label_alternative_groups", 1.0), 0.0)
        self.assertEqual(cardinality_prior_shift(11324, 26085, "multilabel", 0.0), 0.0)

    def test_compile_calibration_rejects_missing_embeddings_for_geometry(self):
        with self.assertRaises(ValueError):
            compile_calibration(None, None, "multilabel", source_count=10, target_count=20, crowding_alpha=0.5)
        scalar = compile_calibration(None, None, "multilabel", source_count=10, target_count=20, prior_shift_scale=1.0)
        torch.testing.assert_close(scalar.logit_bias, torch.full((20,), -math.log(2.0)))

    def test_bures_wasserstein_alignment_matches_moments_is_spd_and_removes_domain_shift(self):
        gen = torch.Generator().manual_seed(17)
        d = 8
        # Source with anisotropic covariance and non-zero mean
        scale_a = torch.tensor([3.0, 2.0, 1.5, 1.0, 0.8, 0.5, 0.4, 0.2])
        src = torch.randn(400, d, generator=gen) * scale_a + torch.linspace(-1.0, 1.0, d)
        # Target with different anisotropy + large constant language/format offset delta
        scale_b = torch.tensor([0.5, 4.0, 0.3, 2.5, 1.2, 0.2, 3.0, 1.0])
        delta = torch.tensor([12.0, -8.0, 5.0, 0.0, -4.0, 7.0, -2.0, 3.0])
        tgt = torch.randn(300, d, generator=gen) * scale_b + delta

        # 1. Mean alignment eliminates shift exactly
        align_mean = fit_codebook_alignment(src, tgt, mode="mean")
        views = LabelViews(tgt[:, None, :].expand(-1, 3, -1).clone(),
                           torch.tensor([[True, False, False]]).expand(len(tgt), -1),
                           torch.zeros(len(tgt), dtype=torch.long))
        aligned_mean = apply_codebook_alignment(views, align_mean)
        torch.testing.assert_close(aligned_mean.vectors[:, 0].mean(0), src.mean(0), atol=1e-4, rtol=1e-4)

        # 2. Bures-Wasserstein map is SPD, matches mean & shrunk covariance, and equals I when cov_A == cov_B
        align_bw = fit_codebook_alignment(src, tgt, mode="bures_wasserstein", reg=0.05)
        t_bw = align_bw.linear_map
        torch.testing.assert_close(t_bw, t_bw.T, atol=1e-5, rtol=1e-5)
        self.assertGreater(torch.linalg.eigvalsh(t_bw.double()).min().item(), 0.0)
        aligned_bw = apply_codebook_alignment(views, align_bw)
        torch.testing.assert_close(aligned_bw.vectors[:, 0].mean(0), src.mean(0), atol=1e-4, rtol=1e-4)
        # Identity property when source == target
        align_id = fit_codebook_alignment(src, src, mode="bures_wasserstein", reg=0.05)
        torch.testing.assert_close(align_id.linear_map, torch.eye(d), atol=1e-4, rtol=1e-4)

        # 3. Single-note shift projection cancels positive shared shift while preserving orthogonal notes
        mu_a = src.mean(0)
        unit = align_mean.shift_unit
        same_lang_note = (mu_a + 15.0 * unit).unsqueeze(0)
        orth = torch.zeros(d)
        orth[3] = 1.0  # orthogonal to unit[3] == 0? Check: delta[3] == 0, src.mean(0)[3] != 0 so make exact orth
        orth = orth - (orth @ unit) * unit
        orth = orth / orth.norm()
        english_note = (mu_a + 2.0 * orth).unsqueeze(0)
        proj_same = project_out_note_shift(same_lang_note, align_mean)
        proj_en = project_out_note_shift(english_note, align_mean)
        self.assertLess(abs(((proj_same[0] - mu_a) @ unit).item()), 1e-4)
        torch.testing.assert_close(proj_en, english_note, atol=1e-5, rtol=1e-5)


class CalibrationTransferIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.root = Path(tempfile.mkdtemp(prefix="calibration-eval-"))
        with contextlib.redirect_stdout(io.StringIO()):
            cls.model_dir = fixture(cls.root / "fixture")
            codes = load_codebook(cls.root / "fixture/codes.json")
            groups = [{"key": f"source_schema_term:{op}-{part}", "operator": op,
                       "members": [codes[0].key, codes[-1].key], "description": "asthma oxygen",
                       "partition": part, "family": f"{op}-{part}"}
                      for op in ("OR", "AND") for part in ("train", "selection", "audit")]
            schema = cls.root / "schemas"
            schema.mkdir()
            write_codebook(codes + schema_codes(groups), schema / "codebook.json")
            (schema / "groups.json").write_text(json.dumps(groups))
            (schema / "metadata.json").write_text(json.dumps({"role": "source_structural_training"}))
            cache_source_features(parser().parse_args([
                "cache-source-features", "--train-data", str(cls.root / "fixture/train"),
                "--validation-data", str(cls.root / "fixture/validation"), "--schemas", str(schema),
                "--model-dir", str(cls.model_dir), "--output", str(cls.root / "features"),
                "--device", "cpu", "--batch-size", "2", "--label-max-len", "128"]))
            trained = train_multiview(parser().parse_args([
                "train-multiview", "--features", str(cls.root / "features"), "--device", "cpu",
                "--output", str(cls.root / "train"), "--epochs", "1", "--patience", "0",
                "--batch-size", "2", "--candidates", "3", "--schema-candidates", "2",
                "--projection-dim", "16", "--lr", "0.01", "--warmup-steps", "0", "--checkpoint-every", "0"]))
            cls.checkpoint = trained["best"]["checkpoint"]
            source = cls.root / "fixture/validation"
            rows = feather.read_table(source / "validation.arrow").to_pylist()
            for row in rows:
                row["label_groups"] = [row["labels"]]
                row["subject_id"] = None
            cls.target = cls.root / "target"
            cls.target.mkdir()
            feather.write_feather(pa.Table.from_pylist(rows), cls.target / "dev.arrow")
            (cls.target / "codebook.json").write_text((source / "codebook.json").read_text())
            meta = json.loads((source / "metadata.json").read_text())
            meta.update(dataset="synthetic target", role="target_evaluation_only", model_fit_allowed=False,
                        threshold_selection_allowed=False, splits={"dev": {"rows": len(rows)}}, output_mode="multilabel")
            (cls.target / "metadata.json").write_text(json.dumps(meta))

    def test_evaluate_transfer_records_calibration_and_geometry_without_mutating_model(self):
        before = torch.load(self.checkpoint, weights_only=False)["model"]
        with contextlib.redirect_stdout(io.StringIO()):
            base = evaluate_transfer(parser().parse_args([
                "evaluate-transfer", "--data", str(self.target), "--split", "dev",
                "--checkpoint", self.checkpoint, "--model-dir", str(self.model_dir),
                "--device", "cpu", "--output", str(self.root / "eval-base")]))
            cal = evaluate_transfer(parser().parse_args([
                "evaluate-transfer", "--data", str(self.target), "--split", "dev",
                "--checkpoint", self.checkpoint, "--model-dir", str(self.model_dir),
                "--device", "cpu", "--prior-shift-scale", "1.0", "--crowding-alpha", "0.5",
                "--crowding-k", "2", "--record-geometry", "--output", str(self.root / "eval-cal")]))
            aligned = evaluate_transfer(parser().parse_args([
                "evaluate-transfer", "--data", str(self.target), "--split", "dev",
                "--checkpoint", self.checkpoint, "--model-dir", str(self.model_dir),
                "--device", "cpu", "--codebook-align", "bures_wasserstein", "--note-shift-proj",
                "--crowding-alpha", "0.5", "--crowding-k", "2", "--record-geometry",
                "--output", str(self.root / "eval-bw")]))
        self.assertNotIn("calibration", base)
        self.assertFalse((self.root / "eval-base/geometry.pt").exists())
        self.assertTrue(cal["calibration"]["geometry_computed"])
        self.assertIn("residual_buckets", cal)
        self.assertEqual(aligned["codebook_alignment"]["mode"], "bures_wasserstein")
        self.assertTrue(aligned["codebook_alignment"]["note_shift_proj"])
        geom = torch.load(self.root / "eval-cal/geometry.pt", weights_only=False)
        self.assertEqual(len(geom["keys"]), cal["candidate_labels"])
        self.assertTrue(((geom["residual"] >= 0.0) & (geom["residual"] <= 1.0)).all())
        after = torch.load(self.checkpoint, weights_only=False)["model"]
        for key in before:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
