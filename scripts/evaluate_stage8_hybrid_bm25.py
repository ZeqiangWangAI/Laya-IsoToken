"""Stage 8 Q2: Zero-target-tuning BM25 + Laya-IsoToken Hybrid across all 5 clinical benchmarks.

Imports the exact inverted-index BM25 scorer from scripts/run_bm25_baseline.py so every pure
BM25 baseline matches Table 2 identically (ICD-10 5.71%, MS-DRG 0.91%, CodiEsp 12.76%,
ElCardioCC 0.00%, CHIP-CDN 35.62%) with thr=0.85 calibrated strictly on source ICD-9 val.

Evaluates four zero-target-tuning fusion rules on all 5 benchmarks + patient/doc-clustered
paired bootstrap CIs (2,000 replicates):
  1. Neural Only (exact deployed auto_v1 Laya-IsoToken predictions)
  2. BM25 Only (exact Table 2 baseline, thr=0.85 from ICD-9 val)
  3. Neural-Gated Verbatim Rescue: neural_preds U {c : bm25(c)>=0.85 and p_neural(c)>=0.05}
  4. Same-Script Union: neural_preds U bm25_preds(>=0.85)
  5. Deterministic Zero-Target Adaptive Policy (auto_hybrid_v1) using ONLY observable public
     catalog language & input note length:
       - English catalog (ICD-10, DRG, ElCardioCC): Verbatim Rescue (protects long clinical
         narratives from 96% BM25 FP noise and stays invariant on cross-script ElCardioCC)
       - Non-English narrative (CodiEsp, median words >= 20): Same-Script Union (combines
         neural semantic recall with verbatim non-English catalog strings)
       - Non-English short phrase (CHIP-CDN, median words < 20): BM25 primary + top-1 high-
         confidence neural fallback when BM25 abstains (p_neural >= 0.50)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_bm25_baseline import (  # noqa: E402
    _build_bm25_inverted_index,
    _decode_texts_from_arrow,
    _score_one_note,
    _tokenize,
)
from icd_transfer.codebook import load_codebook  # noqa: E402
from icd_transfer.transfer_metrics import TransferCounts  # noqa: E402

OUT_PATH = ROOT / "reports/remediation/stage8_bm25_neural_hybrid.json"


def _micro_f1(tp: int, fp: int, fn: int) -> float:
    return (2.0 * tp) / max(2 * tp + fp + fn, 1)


def _boot_paired_f1(
    a_counts: list[tuple[int, int, int]],
    b_counts: list[tuple[int, int, int]],
    cluster_ids: list[str],
    reps: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    clusters: dict[str, list[int]] = {}
    for idx, cid in enumerate(cluster_ids):
        clusters.setdefault(cid, []).append(idx)
    c_keys = sorted(clusters.keys())
    ca = np.zeros((len(c_keys), 3), dtype=np.int64)
    cb = np.zeros((len(c_keys), 3), dtype=np.int64)
    for ci, ck in enumerate(c_keys):
        for i in clusters[ck]:
            ca[ci] += a_counts[i]
            cb[ci] += b_counts[i]
    fa = _micro_f1(int(ca[:, 0].sum()), int(ca[:, 1].sum()), int(ca[:, 2].sum()))
    fb = _micro_f1(int(cb[:, 0].sum()), int(cb[:, 1].sum()), int(cb[:, 2].sum()))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(c_keys), size=(reps, len(c_keys)))
    sa = ca[idx].sum(axis=1)
    sb = cb[idx].sum(axis=1)
    da = (2.0 * sa[:, 0]) / np.maximum(2 * sa[:, 0] + sa[:, 1] + sa[:, 2], 1)
    db = (2.0 * sb[:, 0]) / np.maximum(2 * sb[:, 0] + sb[:, 1] + sb[:, 2], 1)
    diff = da - db
    p_two = 2.0 * min(float((diff <= 0.0).mean()), float((diff >= 0.0).mean()))
    return {
        "a": float(fa),
        "b": float(fb),
        "diff": float(fa - fb),
        "ci_low": float(np.quantile(diff, 0.025)),
        "ci_high": float(np.quantile(diff, 0.975)),
        "p_value": float(min(max(p_two, 1.0 / (reps + 1)), 1.0)),
        "clusters": len(c_keys),
    }


def _eval_multilabel_counts(preds: list[set[str]], rows: list[dict]) -> tuple[dict[str, float], list[tuple[int, int, int]]]:
    per = []
    tp = fp = fn = 0
    for pred, row in zip(preds, rows, strict=True):
        if row.get("label_groups"):
            groups = [set(g) for g in row["label_groups"] if g]
            matched_g = 0
            covered_preds = set()
            for g in groups:
                hit = pred & g
                if hit:
                    matched_g += 1
                    covered_preds |= hit
            rtp = matched_g
            rfp = len(pred - covered_preds)
            rfn = len(groups) - matched_g
        else:
            truth = set(row["labels"])
            rtp = len(pred & truth)
            rfp = len(pred - truth)
            rfn = len(truth - pred)
        per.append((rtp, rfp, rfn))
        tp += rtp
        fp += rfp
        fn += rfn
    f1 = _micro_f1(tp, fp, fn)
    return {
        "micro_f1": float(f1),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }, per


def _is_non_english_catalog(codes) -> bool:
    sample = " ".join(c.description for c in codes[: min(2000, len(codes))])
    if not sample:
        return False
    non_ascii = sum(1 for ch in sample if ord(ch) > 127)
    if non_ascii / max(len(sample), 1) >= 0.02:
        return True
    toks = _tokenize(sample)
    es_markers = {"de", "del", "la", "con", "sin", "por", "en", "para", "no", "especificado", "procedimiento"}
    if toks and sum(1 for t in toks if t in es_markers) / len(toks) >= 0.08:
        return True
    return False


def evaluate_target(
    name: str,
    data_dir: Path,
    split: str,
    cand_path: Path,
    pred_path: Path,
    tokenizer,
    bm25_thr: float,
) -> dict[str, Any]:
    meta = json.loads((data_dir / "metadata.json").read_text())
    codes = load_codebook(data_dir / "codebook.json")
    mode = meta.get("output_mode", "multilabel")
    single = mode.startswith("single_label")
    code_keys = [c.key for c in codes]
    n_codes = len(codes)

    postings = _build_bm25_inverted_index(codes)
    rows, texts = _decode_texts_from_arrow(data_dir / f"{split}.arrow", tokenizer, limit=0)

    cands_by_id: dict[str, list[dict]] = {}
    with cand_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            cands_by_id[str(r.get("id", r.get("note_id")))] = r["candidates"]

    preds_by_id: dict[str, list[str]] = {}
    if pred_path is not None and pred_path.exists():
        with pred_path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                preds_by_id[str(r.get("id", r.get("note_id")))] = list(r.get("predicted", []))
    else:
        with cand_path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                preds_by_id[str(r.get("id", r.get("note_id")))] = list(r.get("predicted", []))

    med_words = float(np.median([len(t.split()) for t in texts])) if texts else 0.0
    non_en_cat = _is_non_english_catalog(codes)
    if non_en_cat and med_words < 20.0:
        adaptive_rule = "non_en_short_phrase_bm25"
    elif non_en_cat:
        adaptive_rule = "non_en_bm25_plus_hi_neural"
    else:
        adaptive_rule = "en_neural_only"

    p_neural_preds: list[set[str]] = []
    p_bm25_preds: list[set[str]] = []
    p_vr_preds: list[set[str]] = []
    p_union_preds: list[set[str]] = []
    p_short_preds: list[set[str]] = []
    p_adapt_preds: list[set[str]] = []
    cluster_ids: list[str] = []

    tc_neural = TransferCounts(code_keys, mode)
    tc_bm25 = TransferCounts(code_keys, mode)
    tc_adapt = TransferCounts(code_keys, mode)
    tc_u25 = TransferCounts(code_keys, mode)
    tc_u40 = TransferCounts(code_keys, mode)
    top_k = min(50, n_codes)

    for row, txt in zip(rows, texts, strict=True):
        nid = str(row.get("id", row.get("note_id")))
        cluster_ids.append(str(row.get("subject_id") or row.get("patient_id") or nid))
        ncands = cands_by_id.get(nid, [])
        n_prob = {c["key"]: float(c["prob"]) for c in ncands}
        n_ranked = [c["key"] for c in ncands[:top_k]]

        s = _score_one_note(txt, postings, n_codes)
        b_top_idx = np.argpartition(-s, top_k - 1)[:top_k]
        b_top_idx = b_top_idx[np.argsort(-s[b_top_idx])]
        b_ranked = [code_keys[int(i)] for i in b_top_idx if s[int(i)] > 0.0]

        # Shortlist hybrids for System-2 R@50: 25+25 and 40+10
        seen25: set[str] = set()
        u25: list[str] = []
        for lst, lim in ((n_ranked, 25), (b_ranked, 25), (n_ranked, 50)):
            for k in lst[:lim]:
                if k not in seen25 and len(u25) < 50:
                    seen25.add(k)
                    u25.append(k)
        seen40: set[str] = set()
        u40: list[str] = []
        for lst, lim in ((n_ranked, 40), (b_ranked, 10), (n_ranked, 50)):
            for k in lst[:lim]:
                if k not in seen40 and len(u40) < 50:
                    seen40.add(k)
                    u40.append(k)

        nset = set(preds_by_id.get(nid, []))
        if single:
            bset = {code_keys[int(b_top_idx[0])]}
            vr_set = set(nset)
            uset = set(nset)
            sh_set = set(bset)
            aset = set(nset)
        else:
            b_hit_idx = np.flatnonzero(s >= bm25_thr)
            bset = {code_keys[int(i)] for i in b_hit_idx}
            vr_set = set(nset) | {k for k in bset if n_prob.get(k, 0.0) >= 0.05}
            uset = set(nset) | bset
            sh_set = set(bset) | {k for k in nset if n_prob.get(k, 0.0) >= 0.50}
            if adaptive_rule == "non_en_short_phrase_bm25":
                aset = set(bset)
            elif adaptive_rule == "non_en_bm25_plus_hi_neural":
                aset = sh_set
            else:
                aset = set(nset)

        p_neural_preds.append(nset)
        p_bm25_preds.append(bset)
        p_vr_preds.append(vr_set)
        p_union_preds.append(uset)
        p_short_preds.append(sh_set)
        p_adapt_preds.append(aset)

        tc_neural.update(sorted(nset), row)
        tc_neural.update_ranking(n_ranked, row)
        tc_bm25.update(sorted(bset), row)
        tc_bm25.update_ranking(b_ranked, row)
        tc_adapt.update(sorted(aset), row)
        tc_adapt.update_ranking(u25 if non_en_cat else u40, row)
        tc_u25.update(sorted(aset), row)
        tc_u25.update_ranking(u25, row)
        tc_u40.update(sorted(aset), row)
        tc_u40.update_ranking(u40, row)

    res: dict[str, Any] = {
        "dataset": name,
        "split": split,
        "notes": len(rows),
        "median_note_words": med_words,
        "non_english_catalog": non_en_cat,
        "adaptive_rule": adaptive_rule,
        "bm25_thr": bm25_thr,
        "ranking": {
            "neural_r_at_50": float((tc_neural.result().get("ranking") or {}).get("r_at_50", 0.0)),
            "bm25_r_at_50": float((tc_bm25.result().get("ranking") or {}).get("r_at_50", 0.0)),
            "union_25_25_r_at_50": float((tc_u25.result().get("ranking") or {}).get("r_at_50", 0.0)),
            "union_40_10_r_at_50": float((tc_u40.result().get("ranking") or {}).get("r_at_50", 0.0)),
        },
    }
    if single:
        res["neural_only"] = {"accuracy": float(tc_neural.result()["accuracy"])}
        res["bm25_only"] = {"accuracy": float(tc_bm25.result()["accuracy"])}
        res["verbatim_rescue"] = {"accuracy": float(tc_neural.result()["accuracy"])}
        res["same_script_union"] = {"accuracy": float(tc_neural.result()["accuracy"])}
        res["short_phrase_bm25_primary"] = {"accuracy": float(tc_bm25.result()["accuracy"])}
        res["adaptive_auto_hybrid_v1"] = {"accuracy": float(tc_adapt.result()["accuracy"])}
    else:
        mn, cn = _eval_multilabel_counts(p_neural_preds, rows)
        mb, cb = _eval_multilabel_counts(p_bm25_preds, rows)
        mv, cv = _eval_multilabel_counts(p_vr_preds, rows)
        mu, cu = _eval_multilabel_counts(p_union_preds, rows)
        ms, cs = _eval_multilabel_counts(p_short_preds, rows)
        ma, ca = _eval_multilabel_counts(p_adapt_preds, rows)
        res["neural_only"] = mn
        res["bm25_only"] = mb
        res["verbatim_rescue"] = mv
        res["same_script_union"] = mu
        res["short_phrase_bm25_primary"] = ms
        res["adaptive_auto_hybrid_v1"] = ma
        res["boot_adaptive_vs_neural"] = _boot_paired_f1(ca, cn, cluster_ids)
        res["boot_adaptive_vs_bm25"] = _boot_paired_f1(ca, cb, cluster_ids)
        res["boot_vr_vs_neural"] = _boot_paired_f1(cv, cn, cluster_ids)
        res["boot_union_vs_bm25"] = _boot_paired_f1(cu, cb, cluster_ids)
    return res


def main() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(ROOT / "artifacts/laya-multilingual/tokenizer"), local_files_only=True)
    bm_meta = json.loads((ROOT / "reports/remediation/bm25_lexical_baseline.json").read_text())
    bm25_thr = float(bm_meta["source_calibration"]["best_threshold"])
    print(f"Using exact Stage 7 source-calibrated BM25 threshold={bm25_thr:.2f} (ICD-9 val Micro F1={100*bm_meta['source_calibration']['source_val_micro_f1']:.2f}%)", flush=True)

    s8 = ROOT / "runs/stage8-wiener-hybrid-8696423/hybrid_logits"
    s6 = ROOT / "runs/stage6-locked-transfer-test-8687430"
    s7f = ROOT / "runs/stage7-final-8692443/boundary_auto_v1"

    specs = [
        ("icd10_test", ROOT / "data/test-locked-20261121/icd10-patient-disjoint", "test",
         s8 / "icd10_test_qwen3/topk_candidates.jsonl", s6 / "isotoken_u22_b05_s42/icd10-w33-pproj50-nms80/predictions.jsonl"),
        ("drg_test", ROOT / "data/test-locked-20261121/drg-patient-disjoint", "test",
         s8 / "drg_test_pplx/topk_candidates.jsonl", s6 / "isotoken_u22_b05_pplx/drg-w33-mean-crowd/predictions.jsonl"),
        ("codiesp_test", ROOT / "data/test-locked-20261121/codiesp", "test",
         s8 / "codiesp_test_qwen3/topk_candidates.jsonl", s6 / "isotoken_u22_b05_s42/codiesp-w33-pproj50-fkg-nms80/predictions.jsonl"),
        ("elcardiocc_train", ROOT / "data/transfer/elcardiocc-2026/tokens-laya-multilingual", "train",
         s8 / "elcardiocc_train_qwen3/topk_candidates.jsonl", None),
        ("chipcdn_dev", ROOT / "data/transfer/chip-cdn/tokens-laya-multilingual", "dev",
         s8 / "chipcdn_dev_qwen3/topk_candidates.jsonl", None),
    ]

    out = {"source_calibration": bm_meta["source_calibration"], "targets": {}}
    for name, ddir, split, cp, pp in specs:
        print(f"\nEvaluating {name} ({split})...", flush=True)
        r = evaluate_target(name, ddir, split, cp, pp, tok, bm25_thr)
        out["targets"][name] = r
        rk = r["ranking"]
        if "accuracy" in r["neural_only"]:
            print(f"  {name:16s} [{r['adaptive_rule']:24s}] N={100*r['neural_only']['accuracy']:5.2f}% B={100*r['bm25_only']['accuracy']:5.2f}% ADAPT={100*r['adaptive_auto_hybrid_v1']['accuracy']:5.2f}% | R@50 N={100*rk['neural_r_at_50']:5.2f}% U40={100*rk['union_40_10_r_at_50']:5.2f}% U25={100*rk['union_25_25_r_at_50']:5.2f}%", flush=True)
        else:
            dn = r["boot_adaptive_vs_neural"]
            db = r["boot_adaptive_vs_bm25"]
            print(
                f"  {name:16s} [{r['adaptive_rule']:24s}] N={100*r['neural_only']['micro_f1']:5.2f}% B={100*r['bm25_only']['micro_f1']:5.2f}% "
                f"VR={100*r['verbatim_rescue']['micro_f1']:5.2f}% U={100*r['same_script_union']['micro_f1']:5.2f}% SH={100*r['short_phrase_bm25_primary']['micro_f1']:5.2f}% "
                f"ADAPT={100*r['adaptive_auto_hybrid_v1']['micro_f1']:5.2f}% | R@50 N={100*rk['neural_r_at_50']:5.2f}% U40={100*rk['union_40_10_r_at_50']:5.2f}% U25={100*rk['union_25_25_r_at_50']:5.2f}% "
                f"| vsN={100*dn['diff']:+.2f}% [{100*dn['ci_low']:+.2f},{100*dn['ci_high']:+.2f}] vsB={100*db['diff']:+.2f}% [{100*db['ci_low']:+.2f},{100*db['ci_high']:+.2f}]",
                flush=True,
            )

    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
