import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.feather as feather
import torch
from torch.nn import functional as F

from icd_transfer.cli import parser
from icd_transfer.eval_isotoken import evaluate_isotoken_transfer
from icd_transfer.isotoken import IsoTokenHead, PlmIcdHead, build_isotoken_model, pool_note_spans
from icd_transfer.semantic_encoder import encode_semantic_labels_cli
from icd_transfer.smoke import fixture
from icd_transfer.train_isotoken import train_isotoken
from icd_transfer.zero_shot_retrieval import evaluate_zeroshot_retrieval


def _write_synthetic_target(source_dir, target_dir):
    rows = feather.read_table(source_dir / "validation.arrow").to_pylist()
    raw_codes = json.loads((source_dir / "codebook.json").read_text())
    group_key = "icd10_diag:GRP_OR"
    raw_codes.append({
        "key": group_key,
        "system": "icd10_diag",
        "code": "GRP_OR",
        "description": "composite group",
        "ancestors": [],
    })
    # Remap keys to icd10_diag to simulate a distinct target codebook
    remapped = []
    key_map = {}
    for row in raw_codes[:-1]:
        new_key = "icd10_diag:" + row["code"]
        key_map[row["key"]] = new_key
        remapped.append({**row, "key": new_key, "system": "icd10_diag"})
    remapped.append(raw_codes[-1])
    groups = [{"key": group_key, "operator": "OR", "members": [remapped[0]["key"], remapped[-2]["key"]]}]
    out_rows = []
    for row in rows:
        out_rows.append({
            **row,
            "subject_id": "target_" + str(row["subject_id"]),
            "labels": [key_map[k] for k in row["labels"]] + [group_key],
            "label_groups": [],
        })
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "codebook.json").write_text(json.dumps(remapped, indent=2))
    (target_dir / "groups.json").write_text(json.dumps(groups, indent=2))
    src_meta = json.loads((source_dir / "metadata.json").read_text())
    meta = {
        "dataset": "synthetic-icd10",
        "role": "target_evaluation_only",
        "model_fit_allowed": False,
        "threshold_selection_allowed": False,
        "splits": ["dev"],
        "output_mode": "multilabel",
        "tokenizer": src_meta["tokenizer"],
    }
    (target_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    schema = pa.schema([
        ("id", pa.string()),
        ("subject_id", pa.string()),
        ("chunks", pa.list_(pa.list_(pa.int32()))),
        ("labels", pa.list_(pa.string())),
        ("label_groups", pa.list_(pa.list_(pa.string()))),
    ])
    feather.write_feather(pa.Table.from_pylist(out_rows, schema=schema), target_dir / "dev.arrow")


class IsoTokenTests(unittest.TestCase):
    def test_pool_note_spans_masks_specials_and_pools_exact_means(self):
        # 2 chunks for note 0, 1 chunk for note 1; L=6, span_size=2
        hidden = torch.arange(3 * 6 * 4, dtype=torch.float32).reshape(3, 6, 4)
        mask = torch.tensor([
            [True, True, True, True, True, False],
            [True, True, True, False, False, False],
            [True, True, False, False, False, False],
        ])
        ids = torch.tensor([
            [101, 5, 6, 7, 102, 0],   # content tokens at positions 1, 2, 3 -> spans: [1], [2, 3]
            [101, 8, 102, 0, 0, 0],   # content token at pos 1 -> 1 span
            [101, 102, 0, 0, 0, 0],   # specials only -> fallback to valid [0, 1] -> 1 span
        ])
        spans, span_mask = pool_note_spans(hidden, mask, [2, 1], span_size=2, input_ids=ids, special_ids=(101, 102))
        self.assertEqual(tuple(spans.shape), (2, 3, 4))
        self.assertEqual(span_mask[0].tolist(), [True, True, True])
        self.assertEqual(span_mask[1].tolist(), [True, False, False])
        self.assertTrue(torch.allclose(spans[0, 0], hidden[0, 1]))
        self.assertTrue(torch.allclose(spans[0, 1], 0.5 * (hidden[0, 2] + hidden[0, 3])))
        self.assertTrue(torch.allclose(spans[0, 2], hidden[1, 1]))
        self.assertTrue(torch.allclose(spans[1, 0], 0.5 * (hidden[2, 0] + hidden[2, 1])))

    def test_isotoken_head_chunk_equivalence_bounded_and_detached_bias(self):
        torch.manual_seed(7)
        head = IsoTokenHead(note_dim=16, label_dim=24, source_size=11, hidden_dim=32, dropout=0.0, label_chunk_size=3)
        raw_labels = torch.randn(11, 24)
        head.set_label_mean(raw_labels.mean(0))
        protos = head.prepare_prototypes(raw_labels)
        spans = torch.randn(3, 7, 16)
        mask = torch.tensor([
            [True, True, True, True, False, False, False],
            [True, True, True, True, True, True, True],
            [True, True, False, False, False, False, False],
        ])
        logits_chunked, top_spans = head.score_spans(spans, mask, protos, label_chunk_size=3, return_evidence=True)
        logits_full = head.score_spans(spans, mask, protos, label_chunk_size=100)
        self.assertTrue(torch.allclose(logits_chunked, logits_full, atol=1e-5))
        self.assertEqual(tuple(top_spans.shape), (3, 11))
        # Check strict boundedness: |logits - bias| <= scale
        scale = head.log_scale.exp().item()
        max_dev = (logits_full - head.bias).abs().max().item()
        self.assertLessEqual(max_dev, scale + 1e-4)

        # Verify source_bias gradient NEVER touches note_key / note_value / scale / bias
        target = torch.zeros_like(logits_full)
        target[0, 2] = 1.0
        cal_loss = F.binary_cross_entropy_with_logits(head.detached_source_logits(logits_full), target)
        cal_loss.backward()
        self.assertIsNotNone(head.source_bias.grad)
        self.assertGreater(head.source_bias.grad.abs().sum().item(), 0.0)
        self.assertIsNone(head.note_key.fc1.weight.grad)
        self.assertIsNone(head.note_value.fc1.weight.grad)
        self.assertIsNone(head.log_scale.grad)

    def test_plmicd_head_chunk_equivalence_and_custom_cast_weights(self):
        torch.manual_seed(11)
        head = PlmIcdHead(note_dim=16, num_codes=9, attn_dim=12, dropout=0.0, label_chunk_size=4)
        spans = torch.randn(2, 6, 16)
        mask = torch.tensor([[True, True, True, True, False, False], [True, True, True, True, True, True]])
        out_chunk = head.score_spans(spans, mask, label_chunk_size=4)
        out_full = head.score_spans(spans, mask, label_chunk_size=100)
        self.assertTrue(torch.allclose(out_chunk, out_full, atol=1e-5))
        # Custom compiled target weights of different cardinality (|B| = 15 != 9)
        custom_q = torch.randn(15, 12)
        custom_w = torch.randn(15, 16)
        custom_b = torch.full((15,), -5.0)
        out_tgt = head.score_spans(spans, mask, custom_query=custom_q, custom_weight=custom_w, custom_bias=custom_b, label_chunk_size=5)
        self.assertEqual(tuple(out_tgt.shape), (2, 15))
        self.assertTrue(torch.isfinite(out_tgt).all())

    def test_trainable_encoder_layers_freezing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                model_dir = fixture(Path(tmp) / "fixture")
            model = build_isotoken_model(model_dir, arch="isotoken", label_dim=16, source_size=4, trainable_encoder_layers=1)
            layers = list(model.encoder.layers)
            self.assertGreaterEqual(len(layers), 1)
            # Embeddings must stay frozen
            for p in model.encoder.embeddings.parameters():
                self.assertFalse(p.requires_grad)
            # Top layer trainable, earlier layers (if any) frozen
            for p in layers[-1].parameters():
                self.assertTrue(p.requires_grad)
            for layer in layers[:-1]:
                for p in layer.parameters():
                    self.assertFalse(p.requires_grad)

    def test_end_to_end_isotoken_and_plmicd_cast_transfer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                model_dir = fixture(root / "fixture")
                target_dir = root / "target_icd10"
                _write_synthetic_target(root / "fixture/validation", target_dir)

                # Pre-encode source semantic labels via CLI
                src_sem = root / "src_sem.pt"
                encode_semantic_labels_cli(parser().parse_args([
                    "encode-semantic-labels",
                    "--codebook", str(root / "fixture/train/codebook.json"),
                    "--semantic-model", str(model_dir),
                    "--device", "cpu",
                    "--output", str(src_sem),
                ]))
                self.assertTrue(src_sem.exists())

                for arch in ("isotoken", "plmicd"):
                    train_out = root / f"train_{arch}"
                    train_isotoken(parser().parse_args([
                        "train-isotoken",
                        "--train-data", str(root / "fixture/train"),
                        "--validation-data", str(root / "fixture/validation"),
                        "--model-dir", str(model_dir),
                        "--semantic-model", str(model_dir),
                        "--semantic-cache", str(src_sem),
                        "--arch", arch,
                        "--span-size", "2",
                        "--trainable-encoder-layers", "1",
                        "--head-hidden-dim", "32",
                        "--attn-dim", "16",
                        "--epochs", "2",
                        "--batch-size", "2",
                        "--eval-batch-size", "2",
                        "--encoder-lr", "1e-3",
                        "--head-lr", "1e-2",
                        "--warmup-steps", "0",
                        "--patience", "0",
                        "--device", "cpu",
                        "--output", str(train_out),
                    ]))
                    summary = json.loads((train_out / "summary.json").read_text())
                    best_ckpt = summary["best"]["checkpoint"]
                    self.assertTrue(Path(best_ckpt).exists())

                    eval_out = root / f"eval_{arch}"
                    evaluate_isotoken_transfer(parser().parse_args([
                        "eval-isotoken",
                        "--checkpoint", best_ckpt,
                        "--data", str(target_dir),
                        "--split", "dev",
                        "--model-dir", str(model_dir),
                        "--source-semantic-cache", str(src_sem),
                        "--target-semantic-cache", str(root / f"tgt_sem_{arch}.pt"),
                        "--batch-size", "2",
                        "--codebook-align", "bures_wasserstein",
                        "--prior-shift-scale", "1.0",
                        "--crowding-alpha", "1.0",
                        "--crowding-k", "2",
                        "--group-algebra", "frechet_clamp",
                        "--record-geometry",
                        "--device", "cpu",
                        "--output", str(eval_out),
                    ]))
                    metrics = json.loads((eval_out / "metrics.json").read_text())
                    self.assertEqual(metrics["arch"], arch)
                    self.assertTrue((eval_out / "geometry.pt").exists())
                    self.assertTrue((eval_out / "extrapolation.json").exists())
                    self.assertIn("codebook_alignment", metrics)
                    self.assertIn("extrapolation", metrics)
                    self.assertIn("ranking", metrics)
                    self.assertIn("r_at_50", metrics["ranking"])
                    if arch == "plmicd":
                        self.assertIn("cast", metrics)
                        xwalk_out = root / "eval_plmicd_xwalk"
                        evaluate_isotoken_transfer(parser().parse_args([
                            "eval-isotoken",
                            "--checkpoint", best_ckpt,
                            "--data", str(target_dir),
                            "--split", "dev",
                            "--model-dir", str(model_dir),
                            "--source-semantic-cache", str(src_sem),
                            "--target-semantic-cache", str(root / f"tgt_sem_{arch}.pt"),
                            "--batch-size", "2",
                            "--plmicd-transfer", "crosswalk",
                            "--crosswalk-k", "2",
                            "--codebook-align", "mean",
                            "--crowding-alpha", "0.5",
                            "--crowding-k", "2",
                            "--record-geometry",
                            "--device", "cpu",
                            "--output", str(xwalk_out),
                        ]))
                        xwalk_m = json.loads((xwalk_out / "metrics.json").read_text())
                        self.assertIn("crosswalk", xwalk_m)
                        self.assertEqual(xwalk_m["crosswalk"]["mode"], "semantic_crosswalk")
                        self.assertIn("ranking", xwalk_m)
                    else:
                        nms_out = root / "eval_isotoken_nms_prior"
                        evaluate_isotoken_transfer(parser().parse_args([
                            "eval-isotoken",
                            "--checkpoint", best_ckpt,
                            "--data", str(target_dir),
                            "--split", "dev",
                            "--model-dir", str(model_dir),
                            "--source-semantic-cache", str(src_sem),
                            "--target-semantic-cache", str(root / f"tgt_sem_{arch}.pt"),
                            "--batch-size", "2",
                            "--prior-proj-tau", "0.5",
                            "--prior-proj-k", "2",
                            "--wiener-lambda", "0.2",
                            "--span-nms-cos", "0.80",
                            "--span-nms-window", "1",
                            "--device", "cpu",
                            "--output", str(nms_out),
                        ]))
                        nms_m = json.loads((nms_out / "metrics.json").read_text())
                        self.assertIn("span_nms", nms_m)
                        self.assertIn("wiener_gain", nms_m)
                        self.assertEqual(nms_m["prior_projection"]["prior_proj_k"], 2)
                        self.assertIn("prior_projection", nms_m)

                # Test Bayesian prior-deconfounded IsoToken (prior_tau=0.5) + evidence span extraction.
                from icd_transfer.evidence import inspect_evidence

                bayes_out = root / "train_isotoken_bayes"
                train_isotoken(parser().parse_args([
                    "train-isotoken",
                    "--train-data", str(root / "fixture/train"),
                    "--validation-data", str(root / "fixture/validation"),
                    "--model-dir", str(model_dir),
                    "--semantic-model", str(model_dir),
                    "--semantic-cache", str(src_sem),
                    "--arch", "isotoken",
                    "--span-size", "2",
                    "--trainable-encoder-layers", "1",
                    "--head-hidden-dim", "32",
                    "--prior-tau", "0.5",
                    "--epochs", "2",
                    "--batch-size", "2",
                    "--eval-batch-size", "2",
                    "--warmup-steps", "0",
                    "--patience", "0",
                    "--device", "cpu",
                    "--output", str(bayes_out),
                ]))
                bayes_summary = json.loads((bayes_out / "summary.json").read_text())
                bayes_ckpt = bayes_summary["best"]["checkpoint"]
                self.assertEqual(bayes_summary["best"]["selection_metric"], "prior_cal_micro_f1")
                bayes_eval = root / "eval_isotoken_bayes"
                evaluate_isotoken_transfer(parser().parse_args([
                    "eval-isotoken",
                    "--checkpoint", bayes_ckpt,
                    "--data", str(target_dir),
                    "--split", "dev",
                    "--model-dir", str(model_dir),
                    "--source-semantic-cache", str(src_sem),
                    "--target-semantic-cache", str(root / "tgt_sem_isotoken.pt"),
                    "--batch-size", "2",
                    "--codebook-align", "bures_wasserstein",
                    "--crowding-alpha", "1.0",
                    "--crowding-k", "2",
                    "--device", "cpu",
                    "--output", str(bayes_eval),
                ]))
                ev_path = root / "evidence.json"
                ev_rows = inspect_evidence(parser().parse_args([
                    "inspect-evidence",
                    "--checkpoint", bayes_ckpt,
                    "--model-dir", str(model_dir),
                    "--data", str(target_dir),
                    "--split", "dev",
                    "--source-codebook", str(root / "fixture/train/codebook.json"),
                    "--source-semantic-cache", str(src_sem),
                    "--target-semantic-cache", str(root / "tgt_sem_isotoken.pt"),
                    "--crowding-k", "2",
                    "--limit", "2",
                    "--top-k", "2",
                    "--device", "cpu",
                    "--output", str(ev_path),
                ]))
                self.assertEqual(len(ev_rows), 2)
                self.assertTrue(ev_path.exists())
                self.assertGreater(len(ev_rows[0]["predictions"]), 0)

                zs_out = root / "eval_zeroshot"
                zs_res = evaluate_zeroshot_retrieval(parser().parse_args([
                    "eval-zeroshot-retrieval",
                    "--source-val-data", str(root / "fixture/validation"),
                    "--source-train-subjects", str(root / "fixture/train/subjects.json"),
                    "--source-semantic-cache", str(src_sem),
                    "--data", str(target_dir),
                    "--split", "dev",
                    "--target-semantic-cache", str(root / "tgt_sem_isotoken.pt"),
                    "--model-dir", str(model_dir),
                    "--semantic-model", str(model_dir),
                    "--semantic-pooling", "mean",
                    "--note-pool", "max",
                    "--source-val-limit", "10",
                    "--batch-size", "2",
                    "--chunk-batch-size", "4",
                    "--crowding-k", "2",
                    "--modes", "raw,bw-crowd-fkg",
                    "--device", "cpu",
                    "--output", str(zs_out),
                ]))
                self.assertIn("raw", zs_res)
                self.assertIn("bw-crowd-fkg", zs_res)
                self.assertTrue((zs_out / "source_calibration.json").exists())
                self.assertTrue((zs_out / f"{target_dir.name}-dev-bw-crowd-fkg/metrics.json").exists())

                # Full source benchmark metrics (Micro/Macro ROC-AUC, P@5/8/15).
                from icd_transfer.eval_isotoken import evaluate_isotoken_source

                src_eval_out = root / "eval_source_bayes"
                src_rep = evaluate_isotoken_source(parser().parse_args([
                    "eval-source-isotoken",
                    "--checkpoint", bayes_ckpt,
                    "--model-dir", str(model_dir),
                    "--train-data", str(root / "fixture/train"),
                    "--validation-data", str(root / "fixture/validation"),
                    "--source-semantic-cache", str(src_sem),
                    "--batch-size", "2",
                    "--device", "cpu",
                    "--output", str(src_eval_out),
                ]))
                self.assertIn("pure", src_rep)
                self.assertIn("prior_cal", src_rep)
                self.assertIn("micro_auc", src_rep["pure"]["full"])
                self.assertIn("macro_auc_supported", src_rep["pure"]["full"])
                self.assertIn("p_at_5", src_rep["pure"]["full"])

    def test_exact_and_histogram_roc_auc_match_pairwise_definition(self):
        import numpy as np
        from icd_transfer.metrics import binary_roc_auc_exact, histogram_roc_auc

        rng = np.random.default_rng(7)
        scores = np.round(rng.normal(0.0, 2.0, size=200), 1)  # deliberate ties
        targets = rng.random(200) < 0.25
        pos = scores[targets]
        neg = scores[~targets]
        gt = float((np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :])) / (len(pos) * len(neg)))
        exact = binary_roc_auc_exact(scores, targets)
        self.assertAlmostEqual(exact, gt, places=12)

        bins = np.clip(((scores + 30.0) * (65536 / 60.0)).astype(np.int64), 0, 65535)
        hist_auc = histogram_roc_auc(
            np.bincount(bins[targets], minlength=65536),
            np.bincount(bins[~targets], minlength=65536),
        )
        self.assertAlmostEqual(hist_auc, gt, places=4)

    def test_wiener_gain_and_mixture_prior_projection(self):
        import math
        import torch.nn.functional as F
        from icd_transfer.calibration import (
            _recover_source_probs,
            semantic_prior_projection,
            wiener_spherical_gain,
        )

        src = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
        ])
        tgt = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.8, 0.0, 0.6, 0.0],
            [0.25, 0.0, math.sqrt(1.0 - 0.0625), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        gain, summary = wiener_spherical_gain(src, tgt, wiener_lambda=0.2)
        self.assertAlmostEqual(float(gain[0]), 1.0, places=5)
        self.assertGreater(float(gain[1]), 1.10)
        self.assertAlmostEqual(float(gain[2]), 1.0, places=5)
        self.assertAlmostEqual(float(gain[3]), 0.0, places=5)
        self.assertAlmostEqual(summary["crossover_cos"], 0.25, places=6)

        counts = [5, 5000, 10, 10]
        n_true = 20000
        pi_true = torch.tensor([(c + 0.5) / (n_true + 1.0) for c in counts], dtype=torch.float64)
        ell = torch.log(pi_true / (1.0 - pi_true))
        bar_pi = pi_true.mean()
        delta = (ell - torch.log(bar_pi / (1.0 - bar_pi))).float()
        pi_rec, _ = _recover_source_probs(delta, counts)
        self.assertLess((pi_rec - pi_true).abs().max().item(), 1e-6)

        src2 = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.6, 0.8, 0.0, 0.0],
            [-0.8, -0.4, 0.4472136, 0.0],
            [-0.8, -0.4, -0.4472136, 0.0],
        ])
        mu = src2.mean(dim=0, keepdim=True)
        z0 = F.normalize(src2[0:1] - mu, dim=-1)
        z1 = F.normalize(src2[1:2] - mu, dim=-1)
        t_mix = (0.52 * z0 + 0.48 * z1) + mu
        t_rare = F.normalize(src2[2:3] - mu, dim=-1) + mu
        tgt2 = torch.cat([t_mix, t_rare], dim=0)
        p1, _ = semantic_prior_projection(src2, tgt2, delta, tau=0.5, top_k=1, source_counts=counts)
        p3, _ = semantic_prior_projection(src2, tgt2, delta, tau=0.5, top_k=3, temp=0.05, source_counts=counts)
        self.assertGreater(float(p3[0] - p3[1]), float(p1[0] - p1[1]) + 1.0)
        self.assertAlmostEqual(float(p3.mean()), 0.0, places=5)

    def test_span_semantic_nms_suppresses_same_span_sibling_only(self):
        from icd_transfer.calibration import apply_span_semantic_nms, semantic_prior_projection
        from icd_transfer.codebook import Code

        codes = [
            Code(key="E785", system="icd10_diag", code="E785", description="Hyperlipidemia unspecified"),
            Code(key="E784", system="icd10_diag", code="E784", description="Other hyperlipidemia"),
            Code(key="E7849", system="icd10_diag", code="E7849", description="Other hyperlipidemia alt"),
            Code(key="I482", system="icd10_diag", code="I482", description="Chronic AFib"),
        ]
        protos = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.98, 0.2, 0.0],
            [0.98, 0.2, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=torch.float32)
        probs = torch.tensor([0.75, 0.62, 0.60, 0.70], dtype=torch.float32)
        spans = torch.tensor([16, 16, 50, 16], dtype=torch.long)
        kept = apply_span_semantic_nms(probs, spans, protos, codes, threshold=0.5, cos_threshold=0.85, span_window=1)
        self.assertEqual(kept, ["E785", "I482", "E7849"])

        src = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
        tgt = torch.tensor([[1.0, 0.0, 0.0], [-1.0, -1.0, -1.0]], dtype=torch.float32)
        delta = torch.tensor([2.0, -1.0, 0.0], dtype=torch.float32)
        proj, summary = semantic_prior_projection(src, tgt, delta, tau=0.5)
        self.assertAlmostEqual(float(proj.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(proj[0] - proj[1]), 1.0, places=4)
        self.assertEqual(summary["prior_proj_tau"], 0.5)

        plm = PlmIcdHead(note_dim=12, num_codes=6, attn_dim=8, dropout=0.0, label_chunk_size=3).eval()
        spans_t = torch.randn(2, 5, 12)
        mask_t = torch.tensor([[True, True, True, False, False], [True, True, True, True, True]])
        plm_logits, plm_spans = plm.score_spans(spans_t, mask_t, return_evidence=True)
        self.assertEqual(tuple(plm_spans.shape), (2, 6))
        self.assertTrue((plm_spans[0] < 3).all())
        self.assertTrue(torch.allclose(plm_logits, plm.score_spans(spans_t, mask_t), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
