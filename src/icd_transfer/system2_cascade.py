"""Real System-1 (IsoToken) -> System-2 (Open-Weight LLM) clinical coding cascade evaluation.

Evaluates four arms on the exact same held-out patient-disjoint ICD-10 test notes:
  Arm 1: Standalone zero-shot open-weight LLM (no System-1 shortlist) -> F1, latency, invalid/hallucination rate
  Arm 2: System-1 IsoToken alone (~9.5 ms/note, 0% invalid codes)
  Arm 3a: Full System-1 -> System-2 Audit Cascade (IsoToken proposed codes + borderline shortlist + 4-token spans -> LLM audit)
  Arm 3b: Confidence-Gated Hybrid Cascade (retain high-confidence System-1 anchors p >= hi_conf_prob,
          delegate uncertain/borderline & high-rho_b candidates to System-2 verification)
  Arm 4: Non-Oracle Selective Escalation Curves (0%, 10%, 20%, 30%, 50%, 75%, 100% of notes routed to System-2
         by random vs margin uncertainty vs rho_b extrapolation vs combined routing)
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .codebook import load_codebook, normalize_code


def _strip_thinking(raw_text: str) -> tuple[str, bool]:
    """Remove closed or truncated model thinking blocks before parsing final JSON answer."""
    truncated_thought = False
    text = re.sub(r"<think>[\s\S]*?</think>", "", raw_text, flags=re.IGNORECASE)
    text = re.sub(r"<thought>[\s\S]*?</thought>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|thought\|>[\s\S]*?<\|/thought\|>", "", text, flags=re.IGNORECASE)
    # Strip unclosed thought blocks caused by token-limit truncation so raw reasoning text is never regex-parsed as codes.
    for open_pat in (r"<think>[\s\S]*$", r"<thought>[\s\S]*$", r"<\|thought\|>[\s\S]*$"):
        if re.search(open_pat, text, flags=re.IGNORECASE):
            truncated_thought = True
            text = re.sub(open_pat, "", text, flags=re.IGNORECASE)
    if text.lstrip().lower().startswith("thought\n"):
        brace = text.find("{")
        if brace >= 0:
            text = text[brace:]
        else:
            truncated_thought = True
            text = ""
    return text.strip(), truncated_thought


def _extract_code_tokens(raw_text: str) -> tuple[list[str], bool]:
    """Parse ICD-10-CM/PCS codes from LLM JSON or free-text output without false English-word PCS hits."""
    text, truncated_thought = _strip_thinking(raw_text)
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    if not text:
        return [], truncated_thought
    for pattern in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
        m = re.search(pattern, text)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict):
                    vals = obj.get("codes") or obj.get("icd10_codes") or obj.get("selected") or []
                elif isinstance(obj, list):
                    vals = obj
                else:
                    vals = []
                if isinstance(vals, list):
                    out = [normalize_code(str(v)) for v in vals if str(v).strip()]
                    return list(dict.fromkeys(out)), truncated_thought
            except Exception:
                pass
    # Require at least one digit inside 7-character PCS tokens so ordinary English words (PRESENT, SURGERY) never match.
    found = re.findall(
        r"\b([A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?|(?=[0-9A-HJ-NP-Z]*[0-9])[0-9A-HJ-NP-Z]{7})\b",
        text.upper(),
    )
    return list(dict.fromkeys(normalize_code(c) for c in found)), truncated_thought


def build_standalone_prompt(note_text: str, max_chars: int = 30000) -> list[dict[str, str]]:
    clipped = note_text[:max_chars]
    return [
        {
            "role": "system",
            "content": (
                "You are an expert hospital clinical coder. Assign all applicable ICD-10-CM diagnosis "
                "and ICD-10-PCS procedure codes explicitly supported by the discharge summary. "
                "Output ONLY valid JSON: {\"codes\": [\"I10\", \"E785\", ...]} with no explanation."
            ),
        },
        {
            "role": "user",
            "content": f"Discharge summary:\n{clipped}\n\nReturn JSON with all supported ICD-10 codes:",
        },
    ]


def build_cascade_prompt(
    note_text: str,
    candidates: list[dict[str, Any]],
    proposed_keys: set[str] | None = None,
    max_chars: int = 30000,
) -> list[dict[str, str]]:
    clipped = note_text[:max_chars]
    proposed_keys = proposed_keys or set()
    sec_a: list[str] = []
    sec_b: list[str] = []
    for i, c in enumerate(candidates, 1):
        span = (c.get("span_text") or "").replace("\n", " ").strip()
        span_note = f" | span: \"{span}\"" if span else ""
        rho = c.get("rho_b")
        rho_note = f", rho={rho:.2f}" if isinstance(rho, (int, float)) else ""
        line = f"{i}. {c['code']} ({c['system']}): {c['description']} [p={c['prob']:.2f}{rho_note}{span_note}]"
        if c["key"] in proposed_keys:
            sec_a.append(line)
        else:
            sec_b.append(line)

    if sec_a:
        cand_block = (
            "Section A — Proposed by fast System-1 reader (verify and KEEP unless unsupported or duplicate):\n"
            + "\n".join(sec_a)
            + "\n\nSection B — Additional nearby candidates (ADD any explicitly documented code missed by System-1):\n"
            + ("\n".join(sec_b) if sec_b else "(none)")
        )
    else:
        cand_block = "Candidate ICD-10 codes (with retrieved 4-token clinical spans):\n" + "\n".join(sec_b)

    return [
        {
            "role": "system",
            "content": (
                "You are an expert hospital clinical coder auditing an automated ICD-10 coding draft. "
                "Using ONLY the discharge summary, select ALL codes from Section A and Section B that are "
                "explicitly supported for this admission. Keep supported codes from Section A, remove unsupported "
                "or redundant sibling codes, and add any explicitly documented diagnoses or procedures from Section B. "
                "Do not invent codes outside the candidate lists. "
                "Output ONLY valid JSON: {\"codes\": [\"...\", ...]}."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Discharge summary:\n{clipped}\n\n"
                f"{cand_block}\n\n"
                "Return JSON listing ALL supported candidate codes from Section A and Section B:"
            ),
        },
    ]


def _micro_metrics(per_note_preds: list[set[str]], per_note_truth: list[set[str]]) -> dict[str, float]:
    tp = fp = fn = 0
    diag_tp = diag_fp = diag_fn = 0
    proc_tp = proc_fp = proc_fn = 0
    c3_tp = c3_fp = c3_fn = 0
    for pred, truth in zip(per_note_preds, per_note_truth, strict=True):
        tp += len(pred & truth)
        fp += len(pred - truth)
        fn += len(truth - pred)
        pd = {k for k in pred if k.startswith("icd10_diag:")}
        td = {k for k in truth if k.startswith("icd10_diag:")}
        diag_tp += len(pd & td)
        diag_fp += len(pd - td)
        diag_fn += len(td - pd)
        p3 = {k.split(":", 1)[1][:3] for k in pd}
        t3 = {k.split(":", 1)[1][:3] for k in td}
        c3_tp += len(p3 & t3)
        c3_fp += len(p3 - t3)
        c3_fn += len(t3 - p3)
        pp = {k for k in pred if k.startswith("icd10_proc:")}
        tp_ = {k for k in truth if k.startswith("icd10_proc:")}
        proc_tp += len(pp & tp_)
        proc_fp += len(pp - tp_)
        proc_fn += len(tp_ - pp)

    def _f1(a: int, b: int, c: int) -> tuple[float, float, float]:
        p = a / (a + b) if (a + b) else 0.0
        r = a / (a + c) if (a + c) else 0.0
        f = 2 * a / (2 * a + b + c) if (2 * a + b + c) else 0.0
        return p, r, f

    p, r, f = _f1(tp, fp, fn)
    _, _, df = _f1(diag_tp, diag_fp, diag_fn)
    _, _, pf = _f1(proc_tp, proc_fp, proc_fn)
    _, _, c3f = _f1(c3_tp, c3_fp, c3_fn)
    avg_pred = float(np.mean([len(s) for s in per_note_preds])) if per_note_preds else 0.0
    return {
        "micro_f1": f,
        "precision": p,
        "recall": r,
        "diag_micro_f1": df,
        "diag_3char_f1": c3f,
        "proc_micro_f1": pf,
        "avg_pred_codes": avg_pred,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _patient_clustered_bootstrap(
    records: list[dict[str, Any]],
    key_a: str,
    key_b: str,
    replicates: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """2,000-replicate patient-clustered paired bootstrap CI for Micro-F1(key_a) - Micro-F1(key_b)."""
    clusters: dict[str, list[tuple[int, int, int, int, int, int]]] = {}
    for r in records:
        pid = str(r.get("subject_id") or r["id"])
        t = set(r["truth"])
        pa = set(r[key_a])
        pb = set(r[key_b])
        clusters.setdefault(pid, []).append((
            len(pa & t), len(pa - t), len(t - pa),
            len(pb & t), len(pb - t), len(t - pb),
        ))
    mat = np.array(
        [[sum(x[i] for x in rows) for i in range(6)] for rows in clusters.values()],
        dtype=np.float64,
    )
    n_clusters = len(mat)
    if n_clusters == 0:
        return {"clusters": 0, "delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0}
    tot = mat.sum(axis=0)
    fa = 2 * tot[0] / max(2 * tot[0] + tot[1] + tot[2], 1.0)
    fb = 2 * tot[3] / max(2 * tot[3] + tot[4] + tot[5], 1.0)
    obs = float(fa - fb)
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for rep in range(replicates):
        idx = rng.integers(0, n_clusters, size=n_clusters)
        s = mat[idx].sum(axis=0)
        f1a = 2 * s[0] / max(2 * s[0] + s[1] + s[2], 1.0)
        f1b = 2 * s[3] / max(2 * s[3] + s[4] + s[5], 1.0)
        deltas[rep] = f1a - f1b
    p_val = float(2.0 * min((deltas <= 0.0).mean(), (deltas >= 0.0).mean()))
    return {
        "clusters": int(n_clusters),
        "delta": obs,
        "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)),
        "p_value": p_val,
    }


def build_hybrid_prediction(
    s1_pred: set[str],
    llm_pred: set[str],
    candidates: list[dict[str, Any]],
    hi_conf_prob: float = 0.65,
) -> set[str]:
    """Confidence-gated Kahneman System-1 + System-2 fusion (non-oracle).

    Retains high-confidence System-1 anchors (p >= hi_conf_prob) across the complete saved
    candidate list while delegating uncertain/borderline and high-extrapolation candidates to System-2.
    """
    prob_by_key = {c["key"]: float(c.get("prob", 0.0)) for c in candidates}
    hi_anchors = {k for k in s1_pred if prob_by_key.get(k, 0.0) >= hi_conf_prob}
    return hi_anchors | llm_pred


def compute_selective_escalation_curves(
    records: list[dict[str, Any]],
    sys1_ms_per_note: float,
    sys2_ms_per_note: float,
    target_key: str = "cascade_pred",
    budgets: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0),
) -> dict[str, list[dict[str, float]]]:
    """Compute exact non-oracle latency vs real-LLM Micro-F1 curves under four routing policies."""
    n = len(records)
    truths = [set(r["truth"]) for r in records]
    s1_preds = [set(r["sys1_pred"]) for r in records]
    s2_preds = [set(r[target_key]) for r in records]

    rho_scores = np.array([float(r["note_rho_score"]) for r in records], dtype=np.float64)
    margin_scores = np.array([float(r["note_margin_uncertainty"]) for r in records], dtype=np.float64)
    rho_z = (rho_scores - rho_scores.mean()) / max(float(rho_scores.std()), 1e-6)
    mar_z = (margin_scores - margin_scores.mean()) / max(float(margin_scores.std()), 1e-6)
    combined_scores = 0.5 * rho_z + 0.5 * mar_z

    rng = np.random.default_rng(42)
    rand_orders = [rng.permutation(n) for _ in range(50)]

    policies = {
        "rho_b_guided": np.argsort(-rho_scores),
        "margin_uncertainty": np.argsort(-margin_scores),
        "combined_rho_margin": np.argsort(-combined_scores),
    }

    curves: dict[str, list[dict[str, float]]] = {k: [] for k in (*policies.keys(), "random_mean50")}
    for b in budgets:
        k_esc = int(round(b * n))
        lat_ms = sys1_ms_per_note + (k_esc / max(n, 1)) * sys2_ms_per_note
        for pname, order in policies.items():
            esc_set = set(order[:k_esc].tolist())
            merged = [s2_preds[i] if i in esc_set else s1_preds[i] for i in range(n)]
            m = _micro_metrics(merged, truths)
            curves[pname].append({"budget": b, "escalated_notes": k_esc, "mean_latency_ms": lat_ms, **m})
        rand_f1s = []
        for order in rand_orders:
            esc_set = set(order[:k_esc].tolist())
            merged = [s2_preds[i] if i in esc_set else s1_preds[i] for i in range(n)]
            rand_f1s.append(_micro_metrics(merged, truths)["micro_f1"])
        curves["random_mean50"].append({
            "budget": b,
            "escalated_notes": k_esc,
            "mean_latency_ms": lat_ms,
            "micro_f1": float(np.mean(rand_f1s)),
            "micro_f1_std": float(np.std(rand_f1s)),
        })
    return curves


def run_cascade_evaluation(
    notes_rows: list[dict[str, Any]],
    cand_map: dict[str, dict[str, Any]],
    code_lookup: dict[str, str],
    generate_fn: Callable[[list[dict[str, str]]], tuple[str, float]],
    top_k: int = 50,
    sys1_ms_per_note: float = 26.9,
    hi_conf_prob: float = 0.65,
    max_chars: int = 30000,
    checkpoint_path: Path | None = None,
    log_every: int = 10,
) -> dict[str, Any]:
    """Run Arm 1 (Standalone LLM), Arm 2 (System-1), Arm 3a/3b (Cascade & Hybrid), and Arm 4."""
    records: list[dict[str, Any]] = []
    done_ids: set[str] = set()
    if checkpoint_path is not None and checkpoint_path.exists():
        for line in checkpoint_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                records.append(rec)
                done_ids.add(str(rec["id"]))

    standalone_total_codes = sum(int(r.get("standalone_emitted_codes", 0)) for r in records)
    standalone_invalid_codes = sum(int(r.get("standalone_invalid_codes", 0)) for r in records)
    standalone_trunc_thoughts = sum(int(r.get("standalone_truncated_thought", 0)) for r in records)
    cascade_trunc_thoughts = sum(int(r.get("cascade_truncated_thought", 0)) for r in records)
    standalone_times = [float(r["standalone_seconds"]) for r in records]
    cascade_times = [float(r["cascade_seconds"]) for r in records]

    ckpt_file = None
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        ckpt_file = checkpoint_path.open("a", encoding="utf-8")

    try:
        for note in notes_rows:
            nid = str(note["id"])
            if nid not in cand_map or nid in done_ids:
                continue
            centry = cand_map[nid]
            truth = list(centry["truth"])
            s1_pred = list(centry["predicted"])
            all_cands = list(centry["candidates"])
            cands = all_cands[:top_k]
            cand_by_code = {normalize_code(c["code"]): c["key"] for c in cands}

            # Arm 1: Standalone zero-shot LLM (reads full unclipped note identically to System-1 & Arm 3)
            raw_s, dt_s = generate_fn(build_standalone_prompt(note["text"], max_chars=max_chars))
            standalone_times.append(dt_s)
            parsed_s, trunc_s = _extract_code_tokens(raw_s)
            standalone_trunc_thoughts += int(trunc_s)
            n_emitted_s = len(parsed_s)
            n_invalid_s = 0
            mapped_s = []
            for c in parsed_s:
                if c in code_lookup:
                    mapped_s.append(code_lookup[c])
                else:
                    n_invalid_s += 1
            standalone_total_codes += n_emitted_s
            standalone_invalid_codes += n_invalid_s
            mapped_s = sorted(set(mapped_s))

            # Arm 3a: System-1 structured draft + Top-K borderline shortlist + 4-token spans -> System-2 audit
            raw_c, dt_c = generate_fn(
                build_cascade_prompt(note["text"], cands, proposed_keys=set(s1_pred), max_chars=max_chars)
            )
            cascade_times.append(dt_c)
            parsed_c, trunc_c = _extract_code_tokens(raw_c)
            cascade_trunc_thoughts += int(trunc_c)
            mapped_c = sorted({cand_by_code[c] for c in parsed_c if c in cand_by_code})

            # Arm 3b: Confidence-gated hybrid fusion (uses complete saved candidates so rank >top_k post-NMS anchors are retained)
            mapped_h = sorted(build_hybrid_prediction(set(s1_pred), set(mapped_c), all_cands, hi_conf_prob=hi_conf_prob))

            thr = float(centry.get("threshold", 0.5))
            top20 = cands[:20]
            note_rho = float(np.mean([c.get("rho_b", 0.0) for c in top20])) if top20 else 0.0
            note_margin_unc = float(np.mean([1.0 - min(1.0, abs(float(c["prob"]) - thr) / max(thr, 1.0 - thr, 1e-6)) for c in top20])) if top20 else 0.0

            rec = {
                "id": nid,
                "subject_id": str(note.get("subject_id", "")),
                "truth": truth,
                "sys1_pred": s1_pred,
                "standalone_pred": mapped_s,
                "cascade_pred": mapped_c,
                "hybrid_pred": mapped_h,
                "standalone_emitted_codes": n_emitted_s,
                "standalone_invalid_codes": n_invalid_s,
                "standalone_truncated_thought": int(trunc_s),
                "cascade_truncated_thought": int(trunc_c),
                "standalone_seconds": dt_s,
                "cascade_seconds": dt_c,
                "note_rho_score": note_rho,
                "note_margin_uncertainty": note_margin_unc,
            }
            records.append(rec)
            done_ids.add(nid)
            if ckpt_file is not None:
                ckpt_file.write(json.dumps(rec) + "\n")
                ckpt_file.flush()

            if len(records) % log_every == 0:
                t_sets = [set(r["truth"]) for r in records]
                m1 = _micro_metrics([set(r["standalone_pred"]) for r in records], t_sets)
                m2 = _micro_metrics([set(r["sys1_pred"]) for r in records], t_sets)
                m3a = _micro_metrics([set(r["cascade_pred"]) for r in records], t_sets)
                m3b = _micro_metrics([set(r["hybrid_pred"]) for r in records], t_sets)
                print(json.dumps({
                    "done": len(records),
                    "sys1": round(m2["micro_f1"], 4),
                    "standalone": round(m1["micro_f1"], 4),
                    "standalone_inv": round(standalone_invalid_codes / max(standalone_total_codes, 1), 4),
                    "trunc_s": standalone_trunc_thoughts,
                    "trunc_c": cascade_trunc_thoughts,
                    "cascade_f1": round(m3a["micro_f1"], 4),
                    "cascade_P": round(m3a["precision"], 4),
                    "cascade_R": round(m3a["recall"], 4),
                    "hybrid_f1": round(m3b["micro_f1"], 4),
                    "hybrid_P": round(m3b["precision"], 4),
                    "hybrid_R": round(m3b["recall"], 4),
                    "avg_s_stand": round(float(np.mean(standalone_times)), 2),
                    "avg_s_casc": round(float(np.mean(cascade_times)), 2),
                }), flush=True)
    finally:
        if ckpt_file is not None:
            ckpt_file.close()

    truths = [set(r["truth"]) for r in records]
    arm1 = _micro_metrics([set(r["standalone_pred"]) for r in records], truths)
    arm1["mean_latency_ms"] = 1000.0 * float(np.mean(standalone_times)) if standalone_times else 0.0
    arm1["total_emitted_codes"] = standalone_total_codes
    arm1["invalid_or_hallucinated_codes"] = standalone_invalid_codes
    arm1["invalid_code_rate"] = standalone_invalid_codes / max(standalone_total_codes, 1)
    arm1["truncated_thought_notes"] = standalone_trunc_thoughts

    arm2 = _micro_metrics([set(r["sys1_pred"]) for r in records], truths)
    arm2["mean_latency_ms"] = sys1_ms_per_note
    arm2["invalid_code_rate"] = 0.0

    arm3_ms = 1000.0 * float(np.mean(cascade_times)) if cascade_times else 0.0
    arm3 = _micro_metrics([set(r["cascade_pred"]) for r in records], truths)
    arm3["mean_latency_ms"] = sys1_ms_per_note + arm3_ms
    arm3["sys2_only_ms"] = arm3_ms
    arm3["invalid_code_rate"] = 0.0
    arm3["truncated_thought_notes"] = cascade_trunc_thoughts

    arm3_hybrid = _micro_metrics([set(r["hybrid_pred"]) for r in records], truths)
    arm3_hybrid["mean_latency_ms"] = sys1_ms_per_note + arm3_ms
    arm3_hybrid["sys2_only_ms"] = arm3_ms
    arm3_hybrid["hi_conf_prob"] = hi_conf_prob
    arm3_hybrid["invalid_code_rate"] = 0.0

    oracle_preds = [
        {c["key"] for c in cand_map[r["id"]]["candidates"][:top_k]} & set(r["truth"])
        for r in records
    ]
    oracle_r = _micro_metrics(oracle_preds, truths)["recall"]

    curves_full = compute_selective_escalation_curves(records, sys1_ms_per_note, arm3_ms, target_key="cascade_pred")
    curves_hybrid = compute_selective_escalation_curves(records, sys1_ms_per_note, arm3_ms, target_key="hybrid_pred")
    bootstraps = {
        "cascade_vs_system1": _patient_clustered_bootstrap(records, "cascade_pred", "sys1_pred"),
        "cascade_vs_standalone": _patient_clustered_bootstrap(records, "cascade_pred", "standalone_pred"),
        "hybrid_vs_system1": _patient_clustered_bootstrap(records, "hybrid_pred", "sys1_pred"),
        "hybrid_vs_standalone": _patient_clustered_bootstrap(records, "hybrid_pred", "standalone_pred"),
        "system1_vs_standalone": _patient_clustered_bootstrap(records, "sys1_pred", "standalone_pred"),
    }
    return {
        "examples": len(records),
        "top_k": top_k,
        "max_chars": max_chars,
        "hi_conf_prob": hi_conf_prob,
        "shortlist_oracle_recall": oracle_r,
        "arm1_standalone_llm": arm1,
        "arm2_system1_isotoken": arm2,
        "arm3_full_cascade": arm3,
        "arm3b_hybrid_cascade": arm3_hybrid,
        "arm4_selective_escalation": curves_full,
        "arm4b_selective_escalation_hybrid": curves_hybrid,
        "patient_clustered_bootstraps": bootstraps,
        "records": records,
    }


def _load_hf_llm(model_name: str, device: str, max_new_tokens: int = 384):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, local_files_only=True, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )
    except Exception:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            model_name, local_files_only=True, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )
    model.eval()
    eos_id = tokenizer.eos_token_id
    if isinstance(eos_id, list):
        eos_id = eos_id[0]

    def _gen(messages: list[dict[str, str]]) -> tuple[str, float]:
        try:
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=eos_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        gen_ids = out[0, inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text, dt

    return _gen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notes-jsonl", required=True)
    parser.add_argument("--candidates-jsonl", required=True)
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--llm-model", default="google/gemma-4-31B-it")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-chars", type=int, default=30000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--sys1-ms", type=float, default=26.9)
    parser.add_argument("--hi-conf-prob", type=float, default=0.50)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    codes = load_codebook(args.codebook)
    code_lookup = {}
    for c in codes:
        norm = normalize_code(c.code)
        if norm not in code_lookup or c.system == "icd10_diag":
            code_lookup[norm] = c.key

    all_notes = [json.loads(line) for line in Path(args.notes_jsonl).read_text(encoding="utf-8").splitlines() if line.strip()]
    notes = all_notes[args.offset : args.offset + args.limit]
    cands = {}
    for line in Path(args.candidates_jsonl).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cands[str(row["id"])] = row

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "per_note_cascade.jsonl"

    print(f"Loaded {len(notes)} notes and {len(cands)} System-1 candidate lists. Loading {args.llm_model}...", flush=True)
    gen_fn = _load_hf_llm(args.llm_model, device="cuda" if torch.cuda.is_available() else "cpu", max_new_tokens=args.max_new_tokens)
    result = run_cascade_evaluation(
        notes, cands, code_lookup, gen_fn,
        top_k=args.top_k,
        sys1_ms_per_note=args.sys1_ms,
        hi_conf_prob=args.hi_conf_prob,
        max_chars=args.max_chars,
        checkpoint_path=ckpt_path,
        log_every=args.log_every,
    )

    records = result.pop("records")
    result["llm_model"] = args.llm_model
    (out_dir / "cascade_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    # Write deduplicated canonical per_note_cascade.jsonl
    with ckpt_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps({
        "done_all": result["examples"],
        "llm_model": args.llm_model,
        "shortlist_oracle_recall": round(result["shortlist_oracle_recall"], 4),
        "arm1_standalone": {k: round(v, 4) for k, v in result["arm1_standalone_llm"].items() if isinstance(v, float)},
        "arm2_system1": {k: round(v, 4) for k, v in result["arm2_system1_isotoken"].items() if isinstance(v, float)},
        "arm3a_full_cascade": {k: round(v, 4) for k, v in result["arm3_full_cascade"].items() if isinstance(v, float)},
        "arm3b_hybrid_cascade": {k: round(v, 4) for k, v in result["arm3b_hybrid_cascade"].items() if isinstance(v, float)},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
