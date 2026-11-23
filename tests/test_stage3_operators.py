import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.feather as feather
import torch

from icd_transfer.attention_cast import compile_attention_cast, fit_attention_cast
from icd_transfer.cli import parser
from icd_transfer.codebook import Code, load_codebook, write_codebook
from icd_transfer.evaluate_transfer import evaluate_transfer
from icd_transfer.schema_algebra import (
    apply_group_algebra,
    enrich_icd10_directory_with_who,
    noisy_and_logit,
    noisy_or_logit,
)
from icd_transfer.smoke import fixture
from icd_transfer.source_features import cache_source_features
from icd_transfer.source_schemas import schema_codes
from icd_transfer.train_multiview import train_multiview


class Stage3OperatorsTests(unittest.TestCase):
    def test_noisy_or_and_identities_and_extreme_stability(self):
        # Single-member exact identity across extreme logit magnitudes
        singles = torch.tensor([[-100.0], [-20.0], [-2.5], [0.0], [3.7], [50.0]], dtype=torch.float32)
        self.assertTrue(torch.allclose(noisy_or_logit(singles), singles.squeeze(-1), atol=1e-4))
        self.assertTrue(torch.allclose(noisy_and_logit(singles), singles.squeeze(-1), atol=1e-4))

        # Exact independent probability law: two p=0.5 events -> OR=0.75 (logit=log(3)), AND=0.25 (logit=-log(3))
        zeros = torch.tensor([0.0, 0.0], dtype=torch.float32)
        self.assertAlmostEqual(noisy_or_logit(zeros).item(), math.log(3.0), places=5)
        self.assertAlmostEqual(noisy_and_logit(zeros).item(), -math.log(3.0), places=5)

        # De Morgan duality across random logits
        torch.manual_seed(0)
        x = torch.randn(16, 7) * 5.0
        self.assertTrue(torch.allclose(noisy_and_logit(x), -noisy_or_logit(-x), atol=1e-5))
        self.assertTrue(torch.isfinite(noisy_or_logit(x)).all())
        self.assertTrue(torch.isfinite(noisy_and_logit(x)).all())

    def test_frechet_clamp_kills_fp_hallucinations_and_enforces_fkg_bounds(self):
        codes = [
            Code("a", "s", "1", "c1"),
            Code("b", "s", "2", "c2"),
            Code("g_or", "s", "g1", ""),
            Code("g_and", "s", "g2", ""),
        ]
        groups = [
            {"key": "g_or", "operator": "OR", "members": ["a", "b"]},
            {"key": "g_and", "operator": "AND", "members": ["a", "b"]},
        ]
        # Case 1: Both children strongly negative (-8, -8), neural OR hallucinates +6.0
        logits = torch.tensor([-8.0, -8.0, 6.0, 5.0])
        clamped = apply_group_algebra(logits, codes, groups, mode="frechet_clamp")
        expected_upper_or = noisy_or_logit(logits[:2]).item()
        self.assertAlmostEqual(clamped[2].item(), expected_upper_or, places=4)
        self.assertLess(clamped[2].item(), -7.0)  # Hallucinated FP completely eliminated!
        self.assertAlmostEqual(clamped[3].item(), -8.0, places=4)  # AND clamped to <= min

        # Case 2: One child strongly positive (+4.0, -6.0), neural OR under-predicts -5.0
        logits2 = torch.tensor([4.0, -6.0, -5.0, -20.0])
        clamped2 = apply_group_algebra(logits2, codes, groups, mode="frechet_clamp")
        self.assertAlmostEqual(clamped2[2].item(), 4.0, places=4)  # Raised to >= max
        self.assertAlmostEqual(clamped2[3].item(), noisy_and_logit(logits2[:2]).item(), places=4)

        # Exact frechet_max mode
        fmax = apply_group_algebra(logits2, codes, groups, mode="frechet_max")
        self.assertAlmostEqual(fmax[2].item(), 4.0, places=5)
        self.assertAlmostEqual(fmax[3].item(), -6.0, places=5)

    def test_attention_cast_recovers_linear_map_and_shrinks_orthogonal_targets(self):
        torch.manual_seed(7)
        # Source subspace spans first 8 dims; dims 8..15 are orthogonal out-of-subspace directions
        n_src, h_dim, d_proj = 64, 16, 12
        src_embed = torch.zeros(n_src, h_dim)
        src_embed[:, :8] = torch.randn(n_src, 8)
        true_mq = torch.randn(h_dim, d_proj)
        true_mv = torch.randn(h_dim, d_proj)
        # True residuals are linear function of whitened source vectors (plus 4 unsupported zero rows)
        op_probe = fit_attention_cast(src_embed, torch.zeros(n_src, d_proj), torch.zeros(n_src, d_proj), ridge=1.0)
        from icd_transfer.calibration import apply_whitener
        z_src_unit = apply_whitener(src_embed, op_probe.mean, op_probe.transform).double()
        dq = (z_src_unit @ true_mq.double()).float()
        dv = (z_src_unit @ true_mv.double()).float()
        counts = [10] * (n_src - 4) + [0] * 4
        # Corrupt the 4 unsupported rows with huge garbage to verify they are masked out!
        dq[-4:] = 999.0
        dv[-4:] = -999.0

        op = fit_attention_cast(src_embed, dq, dv, support_counts=counts, ridge=None, weighting="none")
        self.assertEqual(op.supported_count, n_src - 4)
        self.assertGreater(op.loo_r2, 0.95)

        # In-subspace target codes should reconstruct true residuals accurately
        in_tgt = torch.zeros(8, h_dim)
        in_tgt[:, :8] = torch.randn(8, 8)
        z_in = apply_whitener(in_tgt, op.mean, op.transform).double()
        expected_dq = (z_in @ true_mq.double()).float()
        pred_dq, _ = compile_attention_cast(op, in_tgt, scale=1.0)
        rel_err = (pred_dq - expected_dq).norm() / expected_dq.norm()
        self.assertLess(rel_err.item(), 0.05)

        # Orthogonal target codes (only nonzero in dims 8..15) should automatically shrink near 0
        orth_tgt = torch.zeros(4, h_dim)
        orth_tgt[:, 8:] = torch.randn(4, 8) * 5.0
        orth_dq, orth_dv = compile_attention_cast(op, orth_tgt, scale=1.0)
        self.assertLess(orth_dq.norm().item(), 1e-2)
        self.assertLess(orth_dv.norm().item(), 1e-2)

        # Low-rank SVD truncation respects target rank
        op_lr = fit_attention_cast(src_embed, dq, dv, support_counts=counts, ridge=1.0, rank=3)
        sv = torch.linalg.svdvals(op_lr.map_query)
        self.assertLess(sv[3].item(), 1e-6)

    def test_attention_cast_end_to_end_train_resume_and_transfer(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(prefix="cast-e2e-") as tmp:
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
                full_summary = train_multiview(parser().parse_args([
                    "train-multiview", "--features", str(root / "features"), "--device", "cpu",
                    "--output", str(root / "train-full"), "--head-mode", "attention_cast",
                    "--residual-dropout", "0.25", "--candidates", "0", "--schema-mode", "or_and",
                    "--schema-weight", "0.1", "--epochs", "2", "--patience", "0", "--batch-size", "2",
                    "--projection-dim", "16", "--lr", "0.01", "--warmup-steps", "0",
                    "--checkpoint-every", "0"]))
                best_ckpt = Path(full_summary["best"]["checkpoint"])
                state = torch.load(best_ckpt, map_location="cpu", weights_only=False)
                self.assertIn("delta_query", state["model"])
                self.assertIn("source_label_counts", state)
                self.assertEqual(len(state["source_label_counts"]), len(state["source_codes"]))

                # Resume equivalence check
                train_multiview(parser().parse_args([
                    "train-multiview", "--features", str(root / "features"), "--device", "cpu",
                    "--output", str(root / "train-part1"), "--head-mode", "attention_cast",
                    "--residual-dropout", "0.25", "--candidates", "0", "--schema-mode", "or_and",
                    "--schema-weight", "0.1", "--epochs", "2", "--patience", "0", "--batch-size", "2",
                    "--projection-dim", "16", "--lr", "0.01", "--warmup-steps", "0",
                    "--checkpoint-every", "0", "--max-steps", "2"]))
                resumed = train_multiview(parser().parse_args([
                    "train-multiview", "--features", str(root / "features"), "--device", "cpu",
                    "--output", str(root / "train-part2"), "--resume", str(root / "train-part1/step-00000002.pt"),
                    "--head-mode", "attention_cast", "--residual-dropout", "0.25", "--candidates", "0",
                    "--schema-mode", "or_and", "--schema-weight", "0.1", "--epochs", "2", "--patience", "0",
                    "--batch-size", "2", "--projection-dim", "16", "--lr", "0.01", "--warmup-steps", "0",
                    "--checkpoint-every", "0"]))
                self.assertAlmostEqual(
                    full_summary["best"]["selection_score"], resumed["best"]["selection_score"], places=5,
                )

                # Build synthetic target with an OR group to test Attention-CAST + Fréchet clamp + Stage 2 calibration
                source = root / "fixture/validation"
                rows = feather.read_table(source / "validation.arrow").to_pylist()
                raw_codes = json.loads((source / "codebook.json").read_text())
                group_key = "icd10_diag:GRP_OR"
                raw_codes.append({
                    "key": group_key, "system": "icd10_diag", "code": "GRP_OR", "description": "composite group", "ancestors": [],
                })
                for row in rows:
                    row["labels"] = list(row["labels"]) + [group_key]
                    row["label_groups"] = [[lbl] for lbl in row["labels"]]
                    row["subject_id"] = None
                target = root / "target"
                target.mkdir()
                feather.write_feather(pa.Table.from_pylist(rows), target / "dev.arrow")
                (target / "codebook.json").write_text(json.dumps(raw_codes))
                (target / "groups.json").write_text(json.dumps([
                    {"key": group_key, "operator": "OR", "description": "", "members": [raw_codes[0]["key"], raw_codes[1]["key"]]},
                ]))
                meta = json.loads((source / "metadata.json").read_text())
                meta.update(dataset="synthetic-cast-target", role="target_evaluation_only", model_fit_allowed=False,
                            threshold_selection_allowed=False, splits={"dev": {"rows": len(rows)}},
                            output_mode="multilabel")
                (target / "metadata.json").write_text(json.dumps(meta))

                res_cast = evaluate_transfer(parser().parse_args([
                    "evaluate-transfer", "--checkpoint", str(best_ckpt), "--model-dir", str(model_dir),
                    "--data", str(target), "--split", "dev", "--device", "cpu", "--batch-size", "2",
                    "--label-batch-size", "2", "--prior-shift-scale", "1.0", "--crowding-alpha", "0.5",
                    "--crowding-k", "2", "--record-geometry", "--group-algebra", "frechet_clamp",
                    "--cast-scale", "1.0", "--output", str(root / "eval-cast")]))
                res_base = evaluate_transfer(parser().parse_args([
                    "evaluate-transfer", "--checkpoint", str(best_ckpt), "--model-dir", str(model_dir),
                    "--data", str(target), "--split", "dev", "--device", "cpu", "--batch-size", "2",
                    "--label-batch-size", "2", "--prior-shift-scale", "1.0", "--crowding-alpha", "0.5",
                    "--crowding-k", "2", "--record-geometry", "--group-algebra", "frechet_clamp",
                    "--cast-scale", "0.0", "--output", str(root / "eval-nocast")]))
            self.assertTrue(res_cast["attention_cast"]["enabled"])
            self.assertFalse(res_base["attention_cast"]["enabled"])
            self.assertEqual(res_cast["group_algebra"], "frechet_clamp")
            self.assertEqual(res_cast["head_mode"], "attention_cast")
            self.assertTrue((root / "eval-cast/geometry.pt").exists())

    def test_who_enrichment_on_real_xml_if_present(self):
        who_zip = Path("/Volumes/OutsourceData/ICDCodingJev/data/transfer/who-icd10-2019/icd102019en.xml.zip")
        if not who_zip.exists():
            self.skipTest("WHO ICD-10 XML zip not present")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "src_target"
            src.mkdir()
            (src / "metadata.json").write_text(json.dumps({"dataset": "t"}))
            (src / "codebook.json").write_text(json.dumps([
                {"key": "icd10_diag:A0100", "system": "icd10_diag", "code": "A0100", "description": "Typhoid fever unspecified", "ancestors": []},
                {"key": "icd10_diag:I509", "system": "icd10_diag", "code": "I509", "description": "Heart failure, unspecified", "ancestors": []},
                {"key": "icd10_proc:00160J2", "system": "icd10_proc", "code": "00160J2", "description": "Bypass", "ancestors": []},
            ]))
            summary = enrich_icd10_directory_with_who(src, who_zip, root / "enriched")
            self.assertEqual(summary["diag_codes"], 2)
            self.assertEqual(summary["diag_enriched"], 2)
            cb = json.loads((root / "enriched/codebook.json").read_text())
            self.assertGreaterEqual(len(cb[0]["ancestors"]), 3)  # Chapter -> Block -> Category
            self.assertIn("Heart failure", " ".join(cb[1]["ancestors"]))
            self.assertEqual(cb[2]["ancestors"], [])
            with self.assertRaises(FileExistsError):
                enrich_icd10_directory_with_who(src, who_zip, root / "enriched")

    def test_isometric_exact_cosine_identity_and_view_dropout(self):
        from icd_transfer.label_views import LabelViews
        from icd_transfer.models import IsometricAttentionCASTHead

        torch.manual_seed(17)
        width = 24
        head = IsometricAttentionCASTHead(width=width, source_size=6, residual_dropout=0.5, view_dropout=0.5)
        head.set_label_mean(torch.randn(width))
        head.eval()
        notes = [torch.randn(3, width), torch.randn(5, width), torch.randn(2, width)]
        vectors = torch.randn(6, 3, width)
        available = torch.tensor([
            [True, True, False],
            [True, False, False],
            [True, True, True],
            [False, False, True],
            [True, True, False],
            [True, False, True],
        ])
        operators = torch.tensor([0, 0, 1, 2, 0, 1], dtype=torch.long)
        labels = LabelViews(
            vectors, available, operators,
            delta_query=0.1 * torch.randn(6, width),
            delta_value=0.1 * torch.randn(6, width),
        )
        batched = head.score_batch(notes, labels)
        singles = torch.stack([head.score(n, labels) for n in notes])
        torch.testing.assert_close(batched, singles, atol=1e-5, rtol=1e-5)

        # Verify Gram-trick cosine matches explicit [L, D] cosine_similarity on single-view labels
        one_view = LabelViews(
            vectors, torch.tensor([[True, False, False]] * 6), torch.zeros(6, dtype=torch.long),
            delta_query=labels.delta_query, delta_value=labels.delta_value,
        )
        h = notes[1]
        got = head.score(h, one_view)
        _, q, p = head._active_label_views(one_view)
        key = torch.nn.functional.normalize(head._center(head.note_key(h)), dim=-1, eps=1e-6)
        val = head._center(head.note_value(h))
        attn = (head.attn_log_scale.exp().clamp(max=100.0) * (q @ key.T)).softmax(-1)
        explicit_ev = torch.nn.functional.normalize(attn @ val, dim=-1, eps=1e-6)
        expected = head.log_scale.exp().clamp(max=100.0) * (explicit_ev * p).sum(-1) + head.bias
        torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)

        # Verify locked-label isometry: changing note reader leaves base label cosine geometry untouched
        base_before = torch.nn.functional.normalize(head._center(vectors[:, 0]), dim=-1)
        with torch.no_grad():
            for param in head.note_value.parameters():
                param.add_(0.5)
        base_after = torch.nn.functional.normalize(head._center(vectors[:, 0]), dim=-1)
        torch.testing.assert_close(base_before @ base_before.T, base_after @ base_after.T)

        # Verify view_dropout=1.0 during training matches view-0-only scores (view 1 always dropped, view 0 never dropped)
        head.view_dropout = 1.0
        head.train()
        both_avail = LabelViews(vectors[:4], torch.tensor([[True, True, False]] * 4), torch.zeros(4, dtype=torch.long))
        only_v0 = LabelViews(vectors[:4], torch.tensor([[True, False, False]] * 4), torch.zeros(4, dtype=torch.long))
        head.note_key.dropout.p = 0.0
        for blk in head.note_value:
            blk.dropout.p = 0.0
        s_drop = head.score_batch([h], both_avail)
        s_v0 = head.score_batch([h], only_v0)
        torch.testing.assert_close(s_drop, s_v0, atol=1e-5, rtol=1e-5)

    def test_isometric_cast_end_to_end_train_resume_and_transfer(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(prefix="iso-cast-e2e-") as tmp:
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
                full_summary = train_multiview(parser().parse_args([
                    "train-multiview", "--features", str(root / "features"), "--device", "cpu",
                    "--output", str(root / "train-full"), "--head-mode", "isometric_cast",
                    "--residual-dropout", "0.5", "--view-dropout", "0.5", "--candidates", "0",
                    "--schema-mode", "none", "--epochs", "2", "--patience", "0", "--batch-size", "2",
                    "--lr", "0.01", "--warmup-steps", "0", "--checkpoint-every", "0"]))
                best_ckpt = Path(full_summary["best"]["checkpoint"])
                state = torch.load(best_ckpt, map_location="cpu", weights_only=False)
                self.assertTrue(bool(state["model"]["has_label_mean"].item()))
                self.assertIn("delta_query", state["model"])

                # Exact resume equivalence
                train_multiview(parser().parse_args([
                    "train-multiview", "--features", str(root / "features"), "--device", "cpu",
                    "--output", str(root / "train-p1"), "--head-mode", "isometric_cast",
                    "--residual-dropout", "0.5", "--view-dropout", "0.5", "--candidates", "0",
                    "--schema-mode", "none", "--epochs", "2", "--patience", "0", "--batch-size", "2",
                    "--lr", "0.01", "--warmup-steps", "0", "--checkpoint-every", "0", "--max-steps", "2"]))
                resumed = train_multiview(parser().parse_args([
                    "train-multiview", "--features", str(root / "features"), "--device", "cpu",
                    "--output", str(root / "train-p2"), "--resume", str(root / "train-p1/step-00000002.pt"),
                    "--head-mode", "isometric_cast", "--residual-dropout", "0.5", "--view-dropout", "0.5",
                    "--candidates", "0", "--schema-mode", "none", "--epochs", "2", "--patience", "0",
                    "--batch-size", "2", "--lr", "0.01", "--warmup-steps", "0", "--checkpoint-every", "0"]))
                self.assertAlmostEqual(full_summary["best"]["selection_score"], resumed["best"]["selection_score"], places=5)

                # Transfer evaluation with batched scoring + CAST + calibration + Fréchet clamp
                source = root / "fixture/validation"
                rows = feather.read_table(source / "validation.arrow").to_pylist()
                raw_codes = json.loads((source / "codebook.json").read_text())
                for row in rows:
                    row["label_groups"] = [[lbl] for lbl in row["labels"]]
                    row["subject_id"] = None
                target = root / "target"
                target.mkdir()
                feather.write_feather(pa.Table.from_pylist(rows), target / "dev.arrow")
                (target / "codebook.json").write_text(json.dumps(raw_codes))
                meta = json.loads((source / "metadata.json").read_text())
                meta.update(dataset="synthetic-iso-target", role="target_evaluation_only", model_fit_allowed=False,
                            threshold_selection_allowed=False, splits={"dev": {"rows": len(rows)}},
                            output_mode="multilabel")
                (target / "metadata.json").write_text(json.dumps(meta))
                res = evaluate_transfer(parser().parse_args([
                    "evaluate-transfer", "--checkpoint", str(best_ckpt), "--model-dir", str(model_dir),
                    "--data", str(target), "--split", "dev", "--device", "cpu", "--batch-size", "2",
                    "--label-batch-size", "2", "--prior-shift-scale", "1.0", "--crowding-alpha", "0.5",
                    "--crowding-k", "2", "--record-geometry", "--cast-scale", "1.0",
                    "--output", str(root / "eval-iso")]))
            self.assertEqual(res["head_mode"], "isometric_cast")
            self.assertTrue(res["attention_cast"]["enabled"])
            self.assertTrue((root / "eval-iso/geometry.pt").exists())


if __name__ == "__main__":
    unittest.main()
