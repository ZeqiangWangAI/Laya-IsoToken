import json
import tempfile
import unittest
from pathlib import Path

from icd_transfer.system2_cascade import (
    _extract_code_tokens,
    build_hybrid_prediction,
    run_cascade_evaluation,
)


class System2CascadeTests(unittest.TestCase):
    def test_extract_code_tokens_json_thinking_and_fallback(self):
        codes, trunc = _extract_code_tokens('<thought>Let me check Z99.9</thought>```json\n{"codes": ["I10", "E78.5", "I10"]}\n```')
        self.assertEqual(codes, ["I10", "E785"])
        self.assertFalse(trunc)
        codes2, trunc2 = _extract_code_tokens("Supported: N17.9 and K21.9. PRESENT SURGERY THERAPY CURRENT")
        self.assertEqual(codes2, ["N179", "K219"])
        self.assertFalse(trunc2)
        codes3, trunc3 = _extract_code_tokens("<|thought|>PRESENT SURGERY GENERAL CURRENT I10")
        self.assertEqual(codes3, [])
        self.assertTrue(trunc3)
        # Valid 7-char PCS code with digits is still matched cleanly
        codes4, _ = _extract_code_tokens("Procedure: 02HV33Z and PRESENT.")
        self.assertEqual(codes4, ["02HV33Z"])

    def test_hybrid_preserves_high_conf_anchors(self):
        cands = [
            {"key": "icd10_diag:I10", "prob": 0.85},
            {"key": "icd10_diag:K219", "prob": 0.40},
            {"key": "icd10_diag:E785", "prob": 0.55},
        ]
        out = build_hybrid_prediction(
            s1_pred={"icd10_diag:I10", "icd10_diag:E785"},
            llm_pred={"icd10_diag:K219"},
            candidates=cands,
            hi_conf_prob=0.65,
        )
        self.assertEqual(out, {"icd10_diag:I10", "icd10_diag:K219"})

    def test_cascade_arms_hybrid_and_checkpoint_resume(self):
        notes = [
            {"id": "1", "subject_id": "p1", "text": "History of hypertension and GERD."},
            {"id": "2", "subject_id": "p2", "text": "Acute kidney injury and atrial fibrillation."},
        ]
        cand_map = {
            "1": {
                "id": "1",
                "predicted": ["icd10_diag:I10"],
                "truth": ["icd10_diag:I10", "icd10_diag:K219"],
                "threshold": 0.5,
                "candidates": [
                    {"key": "icd10_diag:I10", "code": "I10", "system": "icd10_diag", "description": "HTN", "prob": 0.8, "rho_b": 0.3, "span_text": "hypertension"},
                    {"key": "icd10_diag:K219", "code": "K219", "system": "icd10_diag", "description": "GERD", "prob": 0.45, "rho_b": 0.7, "span_text": "GERD"},
                ],
            },
            "2": {
                "id": "2",
                "predicted": ["icd10_diag:N179"],
                "truth": ["icd10_diag:N179"],
                "threshold": 0.5,
                "candidates": [
                    {"key": "icd10_diag:N179", "code": "N179", "system": "icd10_diag", "description": "AKI", "prob": 0.9, "rho_b": 0.2, "span_text": "kidney injury"},
                ],
            },
        }
        code_lookup = {"I10": "icd10_diag:I10", "K219": "icd10_diag:K219", "N179": "icd10_diag:N179"}
        calls = []

        def fake_gen(messages):
            content = messages[-1]["content"]
            calls.append(content[:20])
            if "Section A" in content or "Candidate ICD-10" in content:
                if "GERD" in content:
                    return json.dumps({"codes": ["K219"]}), 1.0  # drops I10 in arm3a, hybrid preserves I10!
                return json.dumps({"codes": ["N179"]}), 1.0
            if "GERD" in content:
                return json.dumps({"codes": ["I10", "Z9999"]}), 1.5
            return json.dumps({"codes": ["N179"]}), 1.5

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "per_note.jsonl"
            res = run_cascade_evaluation(notes, cand_map, code_lookup, fake_gen, top_k=5, sys1_ms_per_note=10.0, checkpoint_path=ckpt)
            self.assertGreater(res["arm1_standalone_llm"]["invalid_code_rate"], 0.0)
            self.assertEqual(res["arm3_full_cascade"]["invalid_code_rate"], 0.0)
            self.assertAlmostEqual(res["arm3b_hybrid_cascade"]["micro_f1"], 1.0, places=6)
            self.assertGreater(res["arm3b_hybrid_cascade"]["micro_f1"], res["arm2_system1_isotoken"]["micro_f1"])
            mid_rho = [pt for pt in res["arm4b_selective_escalation_hybrid"]["rho_b_guided"] if pt["budget"] == 0.5][0]
            self.assertAlmostEqual(mid_rho["micro_f1"], 1.0, places=6)
            # Resume skips already-completed notes cleanly
            n_before = len(calls)
            res2 = run_cascade_evaluation(notes, cand_map, code_lookup, fake_gen, top_k=5, sys1_ms_per_note=10.0, checkpoint_path=ckpt)
            self.assertEqual(len(calls), n_before)
            self.assertEqual(res2["examples"], 2)


if __name__ == "__main__":
    unittest.main()
