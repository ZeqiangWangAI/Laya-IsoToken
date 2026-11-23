#!/usr/bin/env python3
"""Compute patient-clustered bootstrap CIs, two-sided p-values, and standardized paired effect sizes.

Metrics computed for every paired comparison across patient/document resampling units:
- micro_f1_difference (or accuracy_difference for single-label MS-DRG)
- patient_bootstrap_95_ci (B=2,000 replicates, seed=42)
- p_value_two_sided (finite-sample Phipson-Smyth correction (k+1)/(B+1))
- cohens_dz: paired standardized mean difference across patient/document unit F1 (mean(d_i) / sd(d_i))
- cliffs_delta_paired: (wins - losses) / N_units
- rank_biserial: (wins - losses) / (wins + losses)
- win_rate_non_ties: wins / (wins + losses)

Saves reports/remediation/effect_sizes_summary.json and builds paper/tables/tabS6_effect_sizes.tex.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from icd_transfer.compare import paired_effect_sizes, patient_counts


def _eval_pair(
    a: np.ndarray,
    b: np.ndarray,
    n_patients: int,
    mode: str = "f1",
    replicates: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    def agg(mat: np.ndarray) -> float:
        s = mat.sum(0)
        if mode == "acc":
            # For single-label Top-1 accuracy, each admission has tp+fn=1 (or tp in {0,1} and n_adm = tp+fn)
            denom = s[0] + s[2]
            return float(s[0] / denom) if denom > 0 else 0.0
        denom = 2 * s[0] + s[1] + s[2]
        return float(2 * s[0] / denom) if denom > 0 else 0.0

    rng = np.random.default_rng(seed)
    n = len(a)
    deltas = np.empty(replicates, dtype=np.float64)
    for r in range(replicates):
        idx = rng.integers(n, size=n)
        deltas[r] = agg(a[idx]) - agg(b[idx])
    obs = agg(a) - agg(b)
    tail = int((deltas <= 0).sum()) if obs >= 0 else int((deltas >= 0).sum())
    p_two = float(min(1.0, 2.0 * (tail + 1) / (replicates + 1)))
    ci = np.quantile(deltas, [0.025, 0.975]).tolist()

    if mode == "acc":
        # Unit accuracy per patient: tp / (tp + fn)
        ua = a[:, 0].astype(np.float64) / np.maximum((a[:, 0] + a[:, 2]).astype(np.float64), 1.0)
        ub = b[:, 0].astype(np.float64) / np.maximum((b[:, 0] + b[:, 2]).astype(np.float64), 1.0)
        diff = ua - ub
        mean_d = float(diff.mean())
        sd_d = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
        dz = float(mean_d / sd_d) if sd_d > 1e-12 else 0.0
        wins = int((diff > 1e-12).sum())
        losses = int((diff < -1e-12).sum())
        ties = int(len(diff) - wins - losses)
        non_ties = max(wins + losses, 1)
        es = {
            "unit_mean_difference": mean_d,
            "unit_sd_difference": sd_d,
            "cohens_dz": dz,
            "cliffs_delta_paired": float((wins - losses) / max(len(diff), 1)),
            "rank_biserial": float((wins - losses) / non_ties),
            "win_rate_non_ties": float(wins / non_ties),
            "wins": wins,
            "losses": losses,
            "ties": ties,
        }
    else:
        es_raw = paired_effect_sizes(a, b)
        es = {
            "unit_mean_difference": es_raw["unit_f1_mean_difference"],
            "unit_sd_difference": es_raw["unit_f1_sd_difference"],
            **{k: v for k, v in es_raw.items() if not k.startswith("unit_f1_")},
        }

    return {
        "units": n,
        "patients": n_patients,
        "metric": "top1_accuracy" if mode == "acc" else "micro_f1",
        "difference": obs,
        "ci_95": ci,
        "p_value_two_sided": p_two,
        "replicates": replicates,
        "effect_size": es,
    }


def compare_jsonl(path_a: Path, path_b: Path, mode: str = "f1") -> dict[str, Any]:
    left, left_ids = patient_counts(path_a)
    right, right_ids = patient_counts(path_b)
    if left_ids != right_ids:
        raise ValueError(f"Mismatched IDs between {path_a} and {path_b}")
    keys = sorted(left)
    a = np.stack([left[k] for k in keys])
    b = np.stack([right[k] for k in keys])
    n_pat = sum(k[0] == "patient" for k in keys)
    return _eval_pair(a, b, n_pat, mode=mode)


def compare_cascade(cascade_jsonl: Path, arm_a: str, arm_b: str) -> dict[str, Any]:
    left: dict[tuple[str, Any], np.ndarray] = defaultdict(lambda: np.zeros(3, dtype=np.int64))
    right: dict[tuple[str, Any], np.ndarray] = defaultdict(lambda: np.zeros(3, dtype=np.int64))
    with cascade_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            key = ("document", row["id"]) if row.get("subject_id") is None else ("patient", row["subject_id"])
            truth = set(row["truth"])
            pa = set(row[arm_a])
            pb = set(row[arm_b])
            left[key] += np.array([len(pa & truth), len(pa - truth), len(truth - pa)], dtype=np.int64)
            right[key] += np.array([len(pb & truth), len(pb - truth), len(truth - pb)], dtype=np.int64)
    keys = sorted(left)
    a = np.stack([left[k] for k in keys])
    b = np.stack([right[k] for k in keys])
    n_pat = sum(k[0] == "patient" for k in keys)
    return _eval_pair(a, b, n_pat, mode="f1")


def _summarize_seeds(seed_results: list[dict[str, Any]]) -> dict[str, Any]:
    diffs = [r["difference"] for r in seed_results]
    dzs = [r["effect_size"]["cohens_dz"] for r in seed_results]
    rbs = [r["effect_size"]["rank_biserial"] for r in seed_results]
    wrs = [r["effect_size"]["win_rate_non_ties"] for r in seed_results]
    return {
        "n_seeds": len(seed_results),
        "units": seed_results[0]["units"],
        "patients": seed_results[0]["patients"],
        "metric": seed_results[0]["metric"],
        "mean_difference": float(np.mean(diffs)),
        "sd_difference": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0,
        "seed42_ci_95": seed_results[0]["ci_95"],
        "max_p_value_across_seeds": float(max(r["p_value_two_sided"] for r in seed_results)),
        "mean_cohens_dz": float(np.mean(dzs)),
        "sd_cohens_dz": float(np.std(dzs, ddof=1)) if len(dzs) > 1 else 0.0,
        "mean_rank_biserial": float(np.mean(rbs)),
        "mean_win_rate_non_ties": float(np.mean(wrs)),
        "per_seed": seed_results,
    }


def main() -> None:
    s6_src = ROOT / "runs/stage6-locked-source-test-8688209"
    s6_tr = ROOT / "runs/stage6-locked-transfer-test-8687430"
    s7_ev = ROOT / "runs/stage7-evals-8690841"
    s8_4b = ROOT / "runs/stage8-qwen3-4b-s42-8697149"
    s8_cas = ROOT / "runs/stage8-gemma4-cascade-1000/per_note_cascade.jsonl"

    results: dict[str, Any] = {}

    # 1. Source MIMIC-IV ICD-9 Locked Test (3 seeds)
    results["source_icd9_qwen3_vs_plmicd"] = _summarize_seeds([
        compare_jsonl(
            s6_src / f"isotoken_u22_b05_s{s}/counts_freq10.jsonl",
            s6_src / f"plmicd_u22_s{s}/counts_freq10.jsonl",
        )
        for s in (42, 43, 44)
    ])

    pplx_src_seeds = [
        compare_jsonl(
            s6_src / "isotoken_u22_b05_pplx/counts_freq10.jsonl",
            s6_src / "plmicd_u22_s42/counts_freq10.jsonl",
        )
    ]
    for s, dname in [(43, "stage7-pplx-s43-8690810"), (44, "stage7-pplx-s44-8690811")]:
        p_cnt = ROOT / f"runs/{dname}/source_test/counts_freq10.jsonl"
        if p_cnt.exists():
            pplx_src_seeds.append(compare_jsonl(p_cnt, s6_src / f"plmicd_u22_s{s}/counts_freq10.jsonl"))
    results["source_icd9_pplx_vs_plmicd"] = _summarize_seeds(pplx_src_seeds)

    # 2. MIMIC-IV ICD-10 Locked Test (3 seeds)
    results["icd10_qwen3_vs_crosswalk"] = _summarize_seeds([
        compare_jsonl(
            s6_tr / f"isotoken_u22_b05_s{s}/icd10-w33-pproj50-nms80/predictions.jsonl",
            s6_tr / f"plmicd_u22_s{s}/icd10-xwalk5-crowd-nms80/predictions.jsonl",
        )
        for s in (42, 43, 44)
    ])
    results["icd10_qwen3_vs_cmsgems_nms"] = _summarize_seeds([
        compare_jsonl(
            s6_tr / f"isotoken_u22_b05_s{s}/icd10-w33-pproj50-nms80/predictions.jsonl",
            s7_ev / f"cms_gem/plmicd_s{s}_gem_both_nms80/predictions.jsonl",
        )
        for s in (42, 43, 44)
    ])
    results["icd10_qwen3_4b_vs_06b_s42"] = _summarize_seeds([
        compare_jsonl(
            s8_4b / "eval_locked_test/icd10-auto-v1/predictions.jsonl",
            s6_tr / "isotoken_u22_b05_s42/icd10-w33-pproj50-nms80/predictions.jsonl",
        )
    ])
    s9_base_dirs = sorted(ROOT.glob("runs/*s9-baselines-evals*"))
    if s9_base_dirs:
        biolord_icd10 = s9_base_dirs[-1] / "zeroshot/biolord_2023m/icd10-patient-disjoint-test-crowd/predictions.jsonl"
        if biolord_icd10.exists():
            results["icd10_qwen3_vs_biolord"] = _summarize_seeds([
                compare_jsonl(
                    s6_tr / f"isotoken_u22_b05_s{s}/icd10-w33-pproj50-nms80/predictions.jsonl",
                    biolord_icd10,
                )
                for s in (42, 43, 44)
            ])

    # 3. MIMIC-IV MS-DRG Locked Test (3 seeds)
    results["drg_qwen3_vs_crosswalk"] = _summarize_seeds([
        compare_jsonl(
            s6_tr / f"isotoken_u22_b05_s{s}/drg-w33-mean-crowd/predictions.jsonl",
            s6_tr / f"plmicd_u22_s{s}/drg-xwalk5-bw-crowd/predictions.jsonl",
            mode="acc",
        )
        for s in (42, 43, 44)
    ])
    pplx_drg_paths = [
        (42, s6_tr / "isotoken_u22_b05_pplx/drg-w33-mean-crowd/predictions.jsonl"),
        (43, ROOT / "runs/stage7-pplx-s43-8690810/eval_locked_test/drg-auto-v1/predictions.jsonl"),
        (44, ROOT / "runs/stage7-pplx-s44-8690811/eval_locked_test/drg-auto-v1/predictions.jsonl"),
    ]
    results["drg_pplx_vs_crosswalk"] = _summarize_seeds([
        compare_jsonl(p, s6_tr / f"plmicd_u22_s{s}/drg-xwalk5-bw-crowd/predictions.jsonl", mode="acc")
        for s, p in pplx_drg_paths if p.exists()
    ])
    results["drg_qwen3_4b_vs_06b_s42"] = _summarize_seeds([
        compare_jsonl(
            s8_4b / "eval_locked_test/drg-auto-v1/predictions.jsonl",
            s6_tr / "isotoken_u22_b05_s42/drg-w33-mean-crowd/predictions.jsonl",
            mode="acc",
        )
    ])

    # 4. CLEF 2020 Spanish CodiEsp Locked Test (3 seeds)
    results["codiesp_qwen3_vs_crosswalk"] = _summarize_seeds([
        compare_jsonl(
            s6_tr / f"isotoken_u22_b05_s{s}/codiesp-w33-pproj50-fkg-nms80/predictions.jsonl",
            s6_tr / f"plmicd_u22_s{s}/codiesp-xwalk5-crowd-fkg-nms80/predictions.jsonl",
        )
        for s in (42, 43, 44)
    ])
    results["codiesp_qwen3_4b_vs_06b_s42"] = _summarize_seeds([
        compare_jsonl(
            s8_4b / "eval_locked_test/codiesp-auto-v1/predictions.jsonl",
            s6_tr / "isotoken_u22_b05_s42/codiesp-w33-pproj50-fkg-nms80/predictions.jsonl",
        )
    ])
    unlk_cod_paths = [
        (42, ROOT / "runs/stage7-unlocked-s42-8690812/eval_locked_test/codiesp-auto-v1/predictions.jsonl"),
    ]
    for s in (43, 44):
        ucands = sorted(ROOT.glob(f"runs/*unlocked*s{s}*/eval_locked_test/codiesp-auto-v1/predictions.jsonl"))
        if ucands:
            unlk_cod_paths.append((s, ucands[-1]))
    if len(unlk_cod_paths) == 3:
        results["codiesp_locked_vs_unlocked_3s"] = _summarize_seeds([
            compare_jsonl(
                s6_tr / f"isotoken_u22_b05_s{s}/codiesp-w33-pproj50-fkg-nms80/predictions.jsonl",
                upath,
            )
            for s, upath in unlk_cod_paths
        ])

    # 5. Full-Note System-1 -> System-2 Cascade (n=1,000, 935 unseen patients)
    results["cascade1000_full_vs_system1"] = _summarize_seeds([
        compare_cascade(s8_cas, "cascade_pred", "sys1_pred")
    ])
    results["cascade1000_full_vs_standalone_31b"] = _summarize_seeds([
        compare_cascade(s8_cas, "cascade_pred", "standalone_pred")
    ])
    results["cascade1000_system1_vs_standalone_31b"] = _summarize_seeds([
        compare_cascade(s8_cas, "sys1_pred", "standalone_pred")
    ])

    out_json = ROOT / "reports/remediation/effect_sizes_summary.json"
    out_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_json}")

    cas_pats = int(results["cascade1000_full_vs_system1"]["patients"])
    # Build Supplementary Table S6: Comprehensive Statistical & Effect-Size Summary
    table_specs = [
        ("MIMIC-IV ICD-9 Test ($15{,}425$ pat.)", "Laya-IsoToken (Qwen3, 3s) vs.\\ Laya-PLM-ICD", results["source_icd9_qwen3_vs_plmicd"]),
        ("MIMIC-IV ICD-9 Test ($15{,}425$ pat.)", "Laya-IsoToken (pplx, 3s) vs.\\ Laya-PLM-ICD", results["source_icd9_pplx_vs_plmicd"]),
        *(
            [("MIMIC-IV ICD-10 Test ($8{,}255$ pat.)", "Laya-IsoToken (Qwen3, 3s) vs.\\ ZeroShot BioLORD-2023-M", results["icd10_qwen3_vs_biolord"])]
            if "icd10_qwen3_vs_biolord" in results
            else []
        ),
        ("MIMIC-IV ICD-10 Test ($8{,}255$ pat.)", "Laya-IsoToken (Qwen3, 3s) vs.\\ Semantic Crosswalk", results["icd10_qwen3_vs_crosswalk"]),
        ("MIMIC-IV ICD-10 Test ($8{,}255$ pat.)", "Laya-IsoToken (Qwen3, 3s) vs.\\ Official CMS GEMs+NMS", results["icd10_qwen3_vs_cmsgems_nms"]),
        ("MIMIC-IV ICD-10 Test ($8{,}255$ pat.)", "Laya-IsoToken (Qwen3-4B vs.\\ 0.6B, s42)", results["icd10_qwen3_4b_vs_06b_s42"]),
        ("MIMIC-IV MS-DRG Test ($7{,}340$ pat.)", "Laya-IsoToken (pplx, 3s) vs.\\ Semantic Crosswalk", results["drg_pplx_vs_crosswalk"]),
        ("MIMIC-IV MS-DRG Test ($7{,}340$ pat.)", "Laya-IsoToken (Qwen3-4B vs.\\ 0.6B, s42)", results["drg_qwen3_4b_vs_06b_s42"]),
        ("Spanish CodiEsp Test ($250$ cases)", "Laya-IsoToken (Qwen3, 3s) vs.\\ Semantic Crosswalk", results["codiesp_qwen3_vs_crosswalk"]),
        *(
            [("Spanish CodiEsp Test ($250$ cases)", "Locked Laya-IsoToken vs.\\ Unlocked (3s)", results["codiesp_locked_vs_unlocked_3s"])]
            if "codiesp_locked_vs_unlocked_3s" in results
            else []
        ),
        ("Spanish CodiEsp Test ($250$ cases)", "Laya-IsoToken (Qwen3-4B vs.\\ 0.6B, s42)", results["codiesp_qwen3_4b_vs_06b_s42"]),
        (f"ICD-10 Cascade (${cas_pats}$ pat., $n=1{{,}}000$)", "Gemma-4-31B Cascade vs.\\ System-1 Alone", results["cascade1000_full_vs_system1"]),
        (f"ICD-10 Cascade (${cas_pats}$ pat., $n=1{{,}}000$)", "Gemma-4-31B Cascade vs.\\ Standalone 31B", results["cascade1000_full_vs_standalone_31b"]),
        (f"ICD-10 Cascade (${cas_pats}$ pat., $n=1{{,}}000$)", "System-1 Alone vs.\\ Standalone 31B", results["cascade1000_system1_vs_standalone_31b"]),
    ]

    rows_tex = []
    for bench, comp, st in table_specs:
        d_pct = 100.0 * st["mean_difference"]
        ci_l, ci_h = 100.0 * st["seed42_ci_95"][0], 100.0 * st["seed42_ci_95"][1]
        p_val = st["max_p_value_across_seeds"]
        p_str = "$<0.001$" if p_val < 0.0015 else f"${p_val:.4f}$"
        dz = st["mean_cohens_dz"]
        rb = st["mean_rank_biserial"]
        wr = 100.0 * st["mean_win_rate_non_ties"]
        rows_tex.append(
            f"{bench} & {comp} & {d_pct:+.2f}\\% & $[{ci_l:+.2f}, {ci_h:+.2f}]$ & {p_str} & "
            f"{dz:+.2f} & {rb:+.2f} & {wr:.1f}\\% \\\\"
        )

    tex_lines = [
        "\\begin{table*}[ht!]",
        "\\centering",
        "\\caption{Patient-clustered paired hypothesis tests and standardized effect sizes across locked test benchmarks ($B=2{,}000$ bootstrap replicates; Phipson--Smyth finite-sample two-sided $p$-values). Resampling clusters admissions by unique \\texttt{subject\\_id} where available (and by clinical document on CodiEsp). Paired Cohen's $d_z = \\bar{\\Delta}_u / s_{\\Delta_u}$~\\cite{cohen2013statistical}, paired rank-biserial correlation $r_{\\text{rb}} = (W - L)/(W + L)$, and non-tie win rate $W/(W+L)$ quantify per-patient clinical effect magnitude alongside corpus-level Micro~F1 / Top-1 differences. For 3-seed comparisons, $\\Delta$, $d_z$, $r_{\\text{rb}}$, and win rate report the 3-seed mean, 95\\% CI reports seed 42, and $p$ reports the conservative maximum across seeds 42--44.}",
        "\\label{tab:effect_sizes}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.4pt}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{@{}llcccccc@{}}",
        "\\toprule",
        "Benchmark (Clusters) & Paired Comparison & $\\Delta$ Score & 95\\% Bootstrap CI & Two-sided $p$ & Cohen's $d_z$ & $r_{\\text{rb}}$ & Win Rate \\\\",
        "\\midrule",
        *rows_tex,
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table*}",
    ]
    out_tex = ROOT / "paper/tables/tabS6_effect_sizes.tex"
    out_tex.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_tex}")
    for bench, comp, st in table_specs:
        print(
            f"  {comp:48s} | d={100*st['mean_difference']:+6.2f}% | "
            f"p<={st['max_p_value_across_seeds']:.4f} | dz={st['mean_cohens_dz']:+.3f} | "
            f"r_rb={st['mean_rank_biserial']:+.3f} | win={100*st['mean_win_rate_non_ties']:.1f}%"
        )


if __name__ == "__main__":
    main()
