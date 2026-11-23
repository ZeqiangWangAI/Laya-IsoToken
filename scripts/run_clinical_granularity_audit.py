"""Local zero-GPU clinical granularity, official DualLAAT code-scope, TRIPOD+AI cohort/fairness, and extrapolation audit."""

from __future__ import annotations

import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.ipc as ipc
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "remediation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _micro_from_sets(pred_list: list[set[str]], truth_list: list[set[str]]) -> dict[str, float]:
    tp = fp = fn = 0
    for p, t in zip(pred_list, truth_list, strict=True):
        tp += len(p & t)
        fp += len(p - t)
        fn += len(t - p)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    return {"micro_f1": f1, "precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}


def _scope_metrics_from_jsonl(pred_path: Path, freq_set: set[str], rare_set: set[str]) -> dict[str, dict]:
    rows = [json.loads(l) for l in pred_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = {}
    for scope_name, allowed in [("all", None), ("benchmark_frequent", freq_set), ("benchmark_rare", rare_set)]:
        tp = fp = fn = 0
        per_tp, per_fp, per_fn = Counter(), Counter(), Counter()
        for r in rows:
            p = set(r["predicted"]) if allowed is None else (set(r["predicted"]) & allowed)
            t = set(r["truth"]) if allowed is None else (set(r["truth"]) & allowed)
            tp += len(p & t)
            fp += len(p - t)
            fn += len(t - p)
            for k in (p & t):
                per_tp[k] += 1
            for k in (p - t):
                per_fp[k] += 1
            for k in (t - p):
                per_fn[k] += 1
        supp = set(per_tp.keys()) | set(per_fn.keys())
        f1s = [2 * per_tp[k] / (2 * per_tp[k] + per_fp[k] + per_fn[k]) for k in supp]
        denom_all = len(allowed) if allowed is not None else 26085
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        out[scope_name] = {
            "micro_f1": f1,
            "precision": prec,
            "recall": rec,
            "macro_f1_supported": float(np.mean(f1s)) if f1s else 0.0,
            "macro_f1_all": float(sum(f1s) / max(denom_all, 1)),
            "supported_labels": len(supp),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return out


def audit_icd10_official_scopes_and_chapters() -> dict:
    groups = json.loads((ROOT / "data/benchmarks/mimiciv-2.2/codebooks/icd10-groups.json").read_text())
    freq_set = set(groups["benchmark_frequent"])
    rare_set = set(groups["benchmark_rare"])

    s6 = ROOT / "runs/stage6-locked-transfer-test-8687430"
    runs = {
        "zeroshot_qwen3": s6 / "zeroshot_qwen3/icd10/crowd/predictions.jsonl",
        "cast_norm_s42": s6 / "plmicd_castnorm_s42/icd10-cast-norm/predictions.jsonl",
        "xwalk5_nms_s42": s6 / "plmicd_u22_s42/icd10-xwalk5-crowd-nms80/predictions.jsonl",
        "xwalk5_nms_s43": s6 / "plmicd_u22_s43/icd10-xwalk5-crowd-nms80/predictions.jsonl",
        "xwalk5_nms_s44": s6 / "plmicd_u22_s44/icd10-xwalk5-crowd-nms80/predictions.jsonl",
        "iso_qwen3_pure_s42": s6 / "isotoken_u22_b05_s42/icd10-pure-nms80/predictions.jsonl",
        "iso_qwen3_pure_s43": s6 / "isotoken_u22_b05_s43/icd10-pure-nms80/predictions.jsonl",
        "iso_qwen3_pure_s44": s6 / "isotoken_u22_b05_s44/icd10-pure-nms80/predictions.jsonl",
        "iso_qwen3_full_s42": s6 / "isotoken_u22_b05_s42/icd10-w33-pproj50-nms80/predictions.jsonl",
        "iso_qwen3_full_s43": s6 / "isotoken_u22_b05_s43/icd10-w33-pproj50-nms80/predictions.jsonl",
        "iso_qwen3_full_s44": s6 / "isotoken_u22_b05_s44/icd10-w33-pproj50-nms80/predictions.jsonl",
        "iso_pplx_pure_s42": s6 / "isotoken_u22_b05_pplx/icd10-pure-nms80/predictions.jsonl",
        "iso_pplx_full_s42": s6 / "isotoken_u22_b05_pplx/icd10-w33-pproj50-nms80/predictions.jsonl",
    }
    scope_res = {}
    for name, path in runs.items():
        if path.exists():
            scope_res[name] = _scope_metrics_from_jsonl(path, freq_set, rare_set)

    # 26-chapter + PCS breakdown on locked test (IsoToken Qwen3 s42 Full vs Crosswalk s42)
    chap_names = {
        "A": "Infectious (A-B)", "B": "Infectious (A-B)", "C": "Neoplasms (C-D49)",
        "D": "Blood/Immune (D50-D89)", "E": "Endocrine/Metabolic (E)", "F": "Mental/Behavioral (F)",
        "G": "Nervous System (G)", "H": "Eye/Ear (H)", "I": "Circulatory (I)",
        "J": "Respiratory (J)", "K": "Digestive (K)", "L": "Skin/Subcutaneous (L)",
        "M": "Musculoskeletal (M)", "N": "Genitourinary (N)", "O": "Pregnancy/Obstetrics (O)",
        "P": "Perinatal (P)", "Q": "Congenital (Q)", "R": "Symptoms/Signs (R)",
        "S": "Injury/Trauma (S-T)", "T": "Injury/Trauma (S-T)",
        "V": "External Causes (V-Y)", "W": "External Causes (V-Y)", "X": "External Causes (V-Y)", "Y": "External Causes (V-Y)",
        "Z": "Health Status/Factors (Z)",
    }

    def _chapter_key(code_key: str) -> str:
        sys, code = code_key.split(":", 1)
        if sys == "icd10_proc":
            return "ICD-10-PCS Procedures"
        ch = code[0].upper()
        if ch == "D" and len(code) >= 2 and code[1].isdigit() and int(code[1]) < 5:
            return "Neoplasms (C-D49)"
        return chap_names.get(ch, f"Other ({ch})")

    def _by_chapter(pred_path: Path) -> dict[str, dict]:
        rows = [json.loads(l) for l in pred_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        tp, fp, fn = Counter(), Counter(), Counter()
        truth_codes = defaultdict(set)
        for r in rows:
            p_set, t_set = set(r["predicted"]), set(r["truth"])
            for k in (p_set & t_set):
                c = _chapter_key(k)
                tp[c] += 1
                truth_codes[c].add(k)
            for k in (p_set - t_set):
                fp[_chapter_key(k)] += 1
            for k in (t_set - p_set):
                c = _chapter_key(k)
                fn[c] += 1
                truth_codes[c].add(k)
        res = {}
        for ch in sorted(set(tp) | set(fp) | set(fn)):
            a, b, c = tp[ch], fp[ch], fn[ch]
            p = a / (a + b) if (a + b) else 0.0
            rec = a / (a + c) if (a + c) else 0.0
            f1 = 2 * a / (2 * a + b + c) if (2 * a + b + c) else 0.0
            res[ch] = {
                "micro_f1": f1, "precision": p, "recall": rec,
                "truth_instances": a + c, "unique_truth_codes": len(truth_codes[ch]),
                "tp": a, "fp": b, "fn": c,
            }
        return res

    chapters_iso = _by_chapter(runs["iso_qwen3_full_s42"])
    chapters_xwalk = _by_chapter(runs["xwalk5_nms_s42"])
    chapter_table = []
    for ch in sorted(chapters_iso.keys(), key=lambda x: -chapters_iso[x]["truth_instances"]):
        iso = chapters_iso[ch]
        xw = chapters_xwalk.get(ch, {"micro_f1": 0.0, "precision": 0.0, "recall": 0.0})
        chapter_table.append({
            "chapter": ch,
            "truth_instances": iso["truth_instances"],
            "unique_truth_codes": iso["unique_truth_codes"],
            "isotoken_f1": iso["micro_f1"],
            "isotoken_p": iso["precision"],
            "isotoken_r": iso["recall"],
            "xwalk_f1": xw["micro_f1"],
            "delta_f1": iso["micro_f1"] - xw["micro_f1"],
        })
    return {"official_scopes": scope_res, "chapters": chapter_table}


def audit_drg_base_families() -> dict:
    cb = json.loads((ROOT / "data/test-locked-20261121/drg-patient-disjoint/codebook.json").read_text())
    code_rows = cb["codes"] if isinstance(cb, dict) and "codes" in cb else cb
    desc_map = {r["key"]: r["description"] for r in code_rows}

    suffix_re = re.compile(
        r"\s+(?:W(?:ITH)?\s+(?:MCC|CC(?:/MCC)?|MAJOR\s+CC)|W/O\s+(?:MCC|CC(?:/MCC)?)|WITHOUT\s+(?:MCC|CC))\b.*$",
        re.IGNORECASE,
    )

    def _base_family(key: str) -> str:
        d = desc_map.get(key, key).upper().strip()
        base = suffix_re.sub("", d).strip(" ,;-")
        return base or d

    s6 = ROOT / "runs/stage6-locked-transfer-test-8687430"
    drg_runs = {
        "xwalk5_s42": s6 / "plmicd_u22_s42/drg-xwalk5-bw-crowd/predictions.jsonl",
        "iso_qwen3_s42": s6 / "isotoken_u22_b05_s42/drg-w33-mean-crowd/predictions.jsonl",
        "iso_qwen3_s43": s6 / "isotoken_u22_b05_s43/drg-w33-mean-crowd/predictions.jsonl",
        "iso_qwen3_s44": s6 / "isotoken_u22_b05_s44/drg-w33-mean-crowd/predictions.jsonl",
        "iso_pplx_s42": s6 / "isotoken_u22_b05_pplx/drg-w33-mean-crowd/predictions.jsonl",
    }
    res = {}
    for name, path in drg_runs.items():
        if not path.exists():
            continue
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        exact_hit = 0
        base_hit = 0
        same_base_wrong_sev = 0
        for r in rows:
            pred = r["predicted"][0] if r["predicted"] else None
            # Acceptable alias group for MS-DRG single-label evaluation
            gold_set = set(r.get("truth", []))
            for g in r.get("label_groups", []):
                gold_set.update(g)
            if pred is None:
                continue
            is_exact = pred in gold_set
            if is_exact:
                exact_hit += 1
                base_hit += 1
            else:
                pb = _base_family(pred)
                if any(_base_family(g) == pb for g in gold_set):
                    base_hit += 1
                    same_base_wrong_sev += 1
        n = len(rows)
        res[name] = {
            "examples": n,
            "exact_top1_acc": exact_hit / max(n, 1),
            "base_family_top1_acc": base_hit / max(n, 1),
            "severity_split_only_errors": same_base_wrong_sev,
            "severity_split_share_of_base_hits": same_base_wrong_sev / max(base_hit, 1),
        }
    return res


def audit_tripod_cohort_and_subgroups() -> dict:
    """Extract Sex, Clinical Service, Note Length, and Comorbidity Burden from MIMIC-IV discharge headers + locked test."""
    s6 = ROOT / "runs/stage6-locked-transfer-test-8687430"
    iso_rows = [json.loads(l) for l in (s6 / "isotoken_u22_b05_s42/icd10-w33-pproj50-nms80/predictions.jsonl").read_text().splitlines() if l.strip()]
    xw_rows = [json.loads(l) for l in (s6 / "plmicd_u22_s42/icd10-xwalk5-crowd-nms80/predictions.jsonl").read_text().splitlines() if l.strip()]
    needed_hadm = {str(r["id"]) for r in iso_rows}

    sex_re = re.compile(r"Sex:\s*([MF])\b", re.IGNORECASE)
    svc_re = re.compile(r"Service:\s*([A-Z/\- ]{2,30})", re.IGNORECASE)
    meta_by_hadm = {}
    discharge_gz = ROOT / "data/raw/mimic-iv-note/2.2/note/discharge.csv.gz"
    if discharge_gz.exists():
        import csv

        with gzip.open(discharge_gz, "rt", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                hid = str(row["hadm_id"])
                if hid not in needed_hadm:
                    continue
                txt = row["text"]
                head = txt[:600]
                sm = sex_re.search(head)
                sv = svc_re.search(head)
                sex = sm.group(1).upper() if sm else "Unknown"
                raw_svc = sv.group(1).strip().split("\n")[0].strip().upper() if sv else "UNKNOWN"
                # Group clinical services cleanly
                if "MEDICINE" in raw_svc:
                    svc = "Medicine"
                elif "SURGERY" in raw_svc or "CARDIOTHORACIC" in raw_svc or "NEUROSURGERY" in raw_svc or "UROLOGY" in raw_svc or "PLASTIC" in raw_svc or "TRAUMA" in raw_svc:
                    svc = "Surgery"
                elif "OBSTETRICS" in raw_svc or "GYNECOLOGY" in raw_svc:
                    svc = "OB/GYN"
                elif "NEUROLOGY" in raw_svc:
                    svc = "Neurology"
                elif "ORTHOPAEDICS" in raw_svc or "ORTHOPEDICS" in raw_svc:
                    svc = "Orthopaedics"
                elif "PSYCHIATRY" in raw_svc:
                    svc = "Psychiatry"
                else:
                    svc = "Other"
                meta_by_hadm[hid] = {
                    "sex": sex,
                    "service": svc,
                    "words": len(txt.split()),
                    "chars": len(txt),
                }

    xw_by_id = {str(r["id"]): r for r in xw_rows}
    words_arr = np.array([meta_by_hadm.get(str(r["id"]), {}).get("words", 0) for r in iso_rows])
    truth_lens = np.array([len(r["truth"]) for r in iso_rows])
    w_q = np.quantile(words_arr[words_arr > 0], [0.25, 0.5, 0.75]) if (words_arr > 0).any() else [0, 0, 0]
    t_q = np.quantile(truth_lens, [0.25, 0.5, 0.75])

    def _bucket_eval(selector) -> dict:
        buckets = defaultdict(lambda: ([], [], [], []))
        for r in iso_rows:
            hid = str(r["id"])
            m = meta_by_hadm.get(hid, {"sex": "Unknown", "service": "Other", "words": 0, "chars": 0})
            b = selector(r, m)
            xr = xw_by_id[hid]
            buckets[b][0].append(set(r["predicted"]))
            buckets[b][1].append(set(r["truth"]))
            buckets[b][2].append(set(xr["predicted"]))
            buckets[b][3].append(str(r.get("subject_id", hid)))
        out = {}
        for b, (ip, tr, xp, pids) in sorted(buckets.items()):
            im = _micro_from_sets(ip, tr)
            xm = _micro_from_sets(xp, tr)
            out[b] = {
                "admissions": len(ip),
                "patients": len(set(pids)),
                "isotoken_f1": im["micro_f1"],
                "isotoken_p": im["precision"],
                "isotoken_r": im["recall"],
                "xwalk_f1": xm["micro_f1"],
                "delta_f1": im["micro_f1"] - xm["micro_f1"],
            }
        return out

    sex_res = _bucket_eval(lambda r, m: m["sex"])
    svc_res = _bucket_eval(lambda r, m: m["service"])
    len_res = _bucket_eval(
        lambda r, m: (
            "Q1 (shortest)" if m["words"] <= w_q[0]
            else ("Q2" if m["words"] <= w_q[1] else ("Q3" if m["words"] <= w_q[2] else "Q4 (longest)"))
        )
    )
    burden_res = _bucket_eval(
        lambda r, m: (
            f"Q1 (<= {int(t_q[0])} codes)" if len(r["truth"]) <= t_q[0]
            else (f"Q2 ({int(t_q[0])+1}-{int(t_q[1])})" if len(r["truth"]) <= t_q[1]
                  else (f"Q3 ({int(t_q[1])+1}-{int(t_q[2])})" if len(r["truth"]) <= t_q[2]
                        else f"Q4 (> {int(t_q[2])} codes)"))
        )
    )

    # Cohort summary across splits
    manifest = json.loads((ROOT / "data/test-locked-20261121/manifest.json").read_text()) if (ROOT / "data/test-locked-20261121/manifest.json").exists() else {}
    return {
        "matched_discharge_headers": len(meta_by_hadm),
        "icd10_test_note_words": {
            "median": float(np.median(words_arr[words_arr > 0])) if (words_arr > 0).any() else 0.0,
            "iqr_low": float(w_q[0]),
            "iqr_high": float(w_q[2]),
            "mean": float(np.mean(words_arr[words_arr > 0])) if (words_arr > 0).any() else 0.0,
        },
        "icd10_test_codes_per_admission": {
            "median": float(np.median(truth_lens)),
            "iqr_low": float(t_q[0]),
            "iqr_high": float(t_q[2]),
            "mean": float(np.mean(truth_lens)),
        },
        "subgroups": {
            "sex": sex_res,
            "service": svc_res,
            "note_length": len_res,
            "comorbidity_burden": burden_res,
        },
        "locked_test_manifest": manifest,
    }


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    def _rankdata(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        s = a[order]
        changes = np.flatnonzero(np.diff(s) != 0) + 1
        starts = np.concatenate(([0], changes))
        ends = np.concatenate((changes, [len(s)]))
        avg = 0.5 * (starts + ends - 1)
        out = np.empty(len(s), dtype=np.float64)
        out[order] = np.repeat(avg, ends - starts)
        return out

    rx, ry = _rankdata(np.asarray(x, dtype=np.float64)), _rankdata(np.asarray(y, dtype=np.float64))
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt(np.dot(rx, rx) * np.dot(ry, ry)))
    return float(np.dot(rx, ry) / denom) if denom > 0 else 0.0


def audit_extrapolation_predictors() -> dict:
    """Compare compile-time Gram residual rho_b quartiles on locked ICD-10 & MS-DRG test."""
    s6 = ROOT / "runs/stage6-locked-transfer-test-8687430/isotoken_u22_b05_s42/icd10-w33-pproj50-nms80"
    geom = torch.load(s6 / "geometry.pt", map_location="cpu", weights_only=False)
    keys = geom["keys"]
    rho = geom["residual"].float().numpy()

    # Reconstruct exact per-code F1 & support on locked ICD-10 test
    rows = [json.loads(l) for l in (s6 / "predictions.jsonl").read_text().splitlines() if l.strip()]
    tp, fp, fn = Counter(), Counter(), Counter()
    for r in rows:
        p, t = set(r["predicted"]), set(r["truth"])
        for k in (p & t): tp[k] += 1
        for k in (p - t): fp[k] += 1
        for k in (t - p): fn[k] += 1

    # Also compute exact quartile Micro-F1 AND Macro-F1 across ALL 4 quartiles Q1..Q4!
    supp_idx = [i for i, k in enumerate(keys) if tp[k] + fn[k] > 0]
    supp_rho = np.array([rho[i] for i in supp_idx])
    supp_f1 = np.array([2 * tp[keys[i]] / max(2 * tp[keys[i]] + fp[keys[i]] + fn[keys[i]], 1) for i in supp_idx])
    edges = np.quantile(supp_rho, [0.0, 0.25, 0.5, 0.75, 1.0])
    quartiles = []
    for q in range(4):
        lo, hi = float(edges[q]), float(edges[q + 1])
        sel = [i for i in supp_idx if (rho[i] >= lo and (rho[i] <= hi if q == 3 else rho[i] < hi))]
        a = sum(tp[keys[i]] for i in sel)
        b = sum(fp[keys[i]] for i in sel)
        c = sum(fn[keys[i]] for i in sel)
        mf1 = 2 * a / max(2 * a + b + c, 1)
        mac = float(np.mean([2 * tp[keys[i]] / max(2 * tp[keys[i]] + fp[keys[i]] + fn[keys[i]], 1) for i in sel]))
        # Top-3 truth-volume codes in each quartile to diagnose Q4 rebound transparently
        top3 = sorted(sel, key=lambda i: -(tp[keys[i]] + fn[keys[i]]))[:5]
        quartiles.append({
            "quartile": f"Q{q+1}",
            "rho_low": lo,
            "rho_high": hi,
            "supported_codes": len(sel),
            "truth_instances": a + c,
            "micro_f1": mf1,
            "macro_f1_supported": mac,
            "top_codes": [
                {"key": keys[i], "rho": round(float(rho[i]), 4), "truth": tp[keys[i]] + fn[keys[i]],
                 "f1": round(2 * tp[keys[i]] / max(2 * tp[keys[i]] + fp[keys[i]] + fn[keys[i]], 1), 4)}
                for i in top3
            ],
        })
    sp_val = _spearman_corr(supp_rho, supp_f1)

    # Also check MS-DRG locked test quartiles!
    drg_dir = ROOT / "runs/stage6-locked-transfer-test-8687430/isotoken_u22_b05_s42/drg-w33-mean-crowd"
    drg_ext = json.loads((drg_dir / "extrapolation.json").read_text()) if (drg_dir / "extrapolation.json").exists() else {}
    return {
        "icd10_s42_locked_test": {
            "supported_codes": len(supp_idx),
            "spearman_rho_vs_f1": sp_val,
            "quartiles": quartiles,
        },
        "drg_s42_locked_test_extrapolation": drg_ext,
    }


def main() -> None:
    print("1. Auditing official ICD-10 DualLAAT scopes (7,942 freq / 18,143 rare) & 26 chapters...", flush=True)
    icd10_audit = audit_icd10_official_scopes_and_chapters()
    (OUT_DIR / "icd10_official_scopes_and_chapters.json").write_text(json.dumps(icd10_audit, indent=2) + "\n")

    print("2. Auditing MS-DRG base clinical families (CC/MCC severity decomposition)...", flush=True)
    drg_audit = audit_drg_base_families()
    (OUT_DIR / "drg_base_family_audit.json").write_text(json.dumps(drg_audit, indent=2) + "\n")

    print("3. Auditing TRIPOD+AI cohort characteristics & clinical subgroup fairness...", flush=True)
    tripod = audit_tripod_cohort_and_subgroups()
    (OUT_DIR / "tripod_cohort_and_subgroups.json").write_text(json.dumps(tripod, indent=2) + "\n")

    print("4. Auditing locked-test Q1-Q4 extrapolation profile & Q4 rebound...", flush=True)
    ext = audit_extrapolation_predictors()
    (OUT_DIR / "locked_test_extrapolation_audit.json").write_text(json.dumps(ext, indent=2) + "\n")

    print("\n=== SUMMARY ===")
    for k in ("xwalk5_nms_s42", "iso_qwen3_pure_s42", "iso_qwen3_full_s42", "iso_pplx_pure_s42", "iso_pplx_full_s42"):
        r = icd10_audit["official_scopes"][k]
        print(
            f"  {k:20s} | ALL={100*r['all']['micro_f1']:.2f}% "
            f"| FREQ(7942)={100*r['benchmark_frequent']['micro_f1']:.2f}% "
            f"| RARE(18143)={100*r['benchmark_rare']['micro_f1']:.2f}% (TP={r['benchmark_rare']['tp']})"
        )
    print("\n--- MS-DRG Base Family vs Exact Top-1 ---")
    for k, v in drg_audit.items():
        print(f"  {k:16s} | Exact Top-1={100*v['exact_top1_acc']:.2f}% | Base-Family={100*v['base_family_top1_acc']:.2f}% (+{100*(v['base_family_top1_acc']-v['exact_top1_acc']):.2f}%)")
    print("\n--- Clinical Subgroups (Sex & Top Services) ---")
    for gname in ("sex", "service"):
        print(f"  [{gname}]")
        for k, v in tripod["subgroups"][gname].items():
            print(f"    {k:16s} n={v['admissions']:5d} | IsoToken={100*v['isotoken_f1']:.2f}% | Crosswalk={100*v['xwalk_f1']:.2f}% | delta={100*v['delta_f1']:+.2f}%")
    print("\n--- Locked ICD-10 Test Quartiles Q1..Q4 ---")
    for q in ext["icd10_s42_locked_test"]["quartiles"]:
        top_str = ", ".join(f"{c['key'].split(':')[1]}(f1={100*c['f1']:.0f}%,n={c['truth']})" for c in q["top_codes"][:3])
        print(f"  {q['quartile']} [{q['rho_low']:.3f},{q['rho_high']:.3f}] n_codes={q['supported_codes']:4d} | MiF1={100*q['micro_f1']:.2f}% | MaF1={100*q['macro_f1_supported']:.2f}% | top: {top_str}")


if __name__ == "__main__":
    main()
