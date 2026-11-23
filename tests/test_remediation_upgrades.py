import unittest
from pathlib import Path

import numpy as np
import torch

from icd_transfer.calibration import (
    apply_span_semantic_nms,
    compile_cms_gem_crosswalk,
    resolve_zero_target_policy,
)
from icd_transfer.codebook import Code
from icd_transfer.isotoken import IsoTokenHead
from icd_transfer.metrics import CalibrationTracker, binary_auprc_exact, histogram_auprc


class RemediationUpgradesTests(unittest.TestCase):
    def test_unlocked_label_mlp_gradient_flow(self):
        head = IsoTokenHead(note_dim=16, label_dim=12, source_size=5, hidden_dim=16, prior_tau=0.5, unlock_label_mlp=True)
        head.set_label_mean(torch.zeros(12))
        raw = torch.randn(5, 12)
        protos = head.prepare_prototypes(raw)
        self.assertEqual(protos.shape, (5, 12))
        self.assertTrue(protos.requires_grad)
        spans = torch.randn(2, 4, 16)
        mask = torch.ones(2, 4, dtype=torch.bool)
        logits = head.score_spans(spans, mask, protos)
        logits.sum().backward()
        self.assertIsNotNone(head.label_proj.fc2.weight.grad)
        self.assertGreater(float(head.label_proj.fc2.weight.grad.abs().sum().item()), 0.0)

    def test_resolve_zero_target_policy_deterministic(self):
        en_codes = [Code(f"icd10_diag:A{i:03d}", "icd10_diag", f"A{i:03d}", "Essential hypertension") for i in range(2000)]
        p_icd10 = resolve_zero_target_policy("multilabel", en_codes, source_count=11324, has_groups=False, source_prior_tau=0.5)
        self.assertEqual(p_icd10["codebook_align"], "none")
        self.assertAlmostEqual(p_icd10["crowding_alpha"], 0.5)
        self.assertAlmostEqual(p_icd10["wiener_lambda"], 1.0 / 3.0)
        self.assertAlmostEqual(p_icd10["prior_proj_tau"], 0.5)
        self.assertAlmostEqual(p_icd10["prior_shift_scale"], 0.0)
        self.assertAlmostEqual(p_icd10["span_nms_cos"], 0.80)
        self.assertEqual(p_icd10["group_algebra"], "none")

        drg_codes = [Code(f"drg:{i}", "drg", str(i), "HEART FAILURE AND SHOCK W MCC") for i in range(1581)]
        p_drg = resolve_zero_target_policy("single_label_alternative_groups", drg_codes, source_count=11324, has_groups=False, source_prior_tau=0.5)
        self.assertEqual(p_drg["codebook_align"], "mean")
        self.assertEqual(p_drg["span_nms_cos"], 0.0)

        es_codes = [Code(f"codiesp:{i}", "codiesp", str(i), "Hipertensión esencial primaria") for i in range(2000)]
        p_es = resolve_zero_target_policy("multilabel", es_codes, source_count=11324, has_groups=True, source_prior_tau=0.5)
        self.assertEqual(p_es["codebook_align"], "none")
        self.assertEqual(p_es["group_algebra"], "frechet_clamp")

        el_codes = [Code(f"el:{i}", "el", str(i), "Essential hypertension") for i in range(115)]
        p_el = resolve_zero_target_policy("multilabel", el_codes, source_count=11324, has_groups=False, source_prior_tau=0.5)
        self.assertEqual(p_el["codebook_align"], "bures_wasserstein")
        self.assertAlmostEqual(p_el["prior_shift_scale"], 1.0)
        self.assertAlmostEqual(p_el["prior_proj_tau"], 0.0)

        zh_codes = [Code(f"zh:{i}", "zh", str(i), "原发性高血压伴心力衰竭") for i in range(3000)]
        p_zh = resolve_zero_target_policy("multilabel", zh_codes, source_count=11324, has_groups=False, source_prior_tau=0.5)
        self.assertEqual(p_zh["codebook_align"], "bures_wasserstein")
        self.assertAlmostEqual(p_zh["prior_proj_tau"], 0.5)

    def test_calibration_and_auprc_metrics(self):
        rng = np.random.default_rng(7)
        y = (rng.random(500) < 0.2).astype(np.bool_)
        logits = rng.normal(0.0, 1.0, size=500) + 2.0 * y
        probs = 1.0 / (1.0 + np.exp(-logits))
        tr = CalibrationTracker(n_bins=15)
        tr.update(probs, y)
        res = tr.result()
        self.assertGreaterEqual(res["ece"], 0.0)
        self.assertLessEqual(res["ece"], 1.0)
        self.assertAlmostEqual(res["brier"], float(np.mean((probs - y) ** 2)), places=10)
        ap_exact = binary_auprc_exact(logits, y)
        bins = np.clip(((logits + 30.0) * (65536 / 60.0)).astype(np.int64), 0, 65535)
        ap_hist = histogram_auprc(np.bincount(bins[y], minlength=65536), np.bincount(bins[~y], minlength=65536))
        self.assertAlmostEqual(ap_exact, ap_hist, places=3)

    def test_vectorized_span_nms_exact(self):
        codes = [Code(f"k:{i}", "icd10_diag", str(i), f"d{i}") for i in range(4)]
        probs = torch.tensor([0.9, 0.8, 0.7, 0.2])
        spans = torch.tensor([5, 6, 20, 5])
        protos = torch.tensor([
            [1.0, 0.0],
            [0.95, 0.31225],
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        kept = apply_span_semantic_nms(probs, spans, protos, codes, threshold=0.5, cos_threshold=0.80, span_window=1)
        self.assertEqual(kept, ["k:0", "k:2"])

    def test_cms_gem_compile_real_if_present(self):
        gdir = Path("data/transfer/cms-gems-2018")
        if not (gdir / "2018_I9gem.txt").exists():
            self.skipTest("CMS GEMs not present locally")
        src = [Code("icd9_diag:4019", "icd9_diag", "4019", "HTN"), Code("icd9_diag:53081", "icd9_diag", "53081", "GERD")]
        tgt = [Code("icd10_diag:I10", "icd10_diag", "I10", "HTN"), Code("icd10_diag:K219", "icd10_diag", "K219", "GERD")]
        idx, w, summary = compile_cms_gem_crosswalk(src, tgt, gdir, direction="both")
        self.assertEqual(summary["mapped_targets"], 2)
        self.assertAlmostEqual(float(w[0].sum().item()), 1.0, places=5)
        self.assertEqual(int(idx[0, 0].item()), 0)

    def test_paired_effect_sizes_and_cls_pooling(self):
        from icd_transfer.compare import paired_effect_sizes
        from icd_transfer.semantic_encoder import _infer_pooling, pool_semantic_hidden

        a = np.array([[4, 1, 1], [5, 1, 0], [3, 1, 2], [6, 0, 1]], dtype=np.int64)
        b = np.array([[2, 2, 2], [3, 2, 2], [1, 2, 3], [4, 2, 2]], dtype=np.int64)
        es = paired_effect_sizes(a, b)
        self.assertGreater(es["cohens_dz"], 2.0)
        self.assertAlmostEqual(es["cliffs_delta_paired"], 1.0)
        self.assertAlmostEqual(es["win_rate_non_ties"], 1.0)
        self.assertEqual(es["wins"], 4)
        self.assertEqual(es["losses"], 0)

        self.assertEqual(_infer_pooling("ncbi/MedCPT-Query-Encoder", "auto"), "cls")
        hidden = torch.tensor([[[10.0, 20.0], [1.0, 2.0], [3.0, 4.0]]])
        mask = torch.tensor([[1, 1, 0]])
        cls_out = pool_semantic_hidden(hidden, mask, "cls")
        self.assertTrue(torch.allclose(cls_out, torch.tensor([[10.0, 20.0]])))


if __name__ == "__main__":
    unittest.main()
