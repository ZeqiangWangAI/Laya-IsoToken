"""Multi-seed mean ± std aggregation for Stage 5 camera-ready tables."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def mean_std(values: list[float]) -> dict[str, float | int]:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    mu = sum(values) / n
    if n == 1:
        return {"n": 1, "mean": mu, "std": 0.0}
    var = sum((x - mu) ** 2 for x in values) / (n - 1)
    return {"n": n, "mean": mu, "std": math.sqrt(var)}


def fmt_pct(stats: dict[str, float | int], latex: bool = False) -> str:
    n = int(stats["n"])
    if n == 0 or math.isnan(float(stats["mean"])):
        return "---" if latex else "-"
    mu = 100.0 * float(stats["mean"])
    if n == 1:
        return f"{mu:.2f}"
    sd = 100.0 * float(stats["std"])
    return f"{mu:.2f} $\\pm$ {sd:.2f}" if latex else f"{mu:.2f} ± {sd:.2f}"


def fmt_ms(stats: dict[str, float | int], latex: bool = False) -> str:
    n = int(stats["n"])
    if n == 0 or math.isnan(float(stats["mean"])):
        return "---" if latex else "-"
    mu = float(stats["mean"])
    if n == 1:
        return f"{mu:.2f}"
    sd = float(stats["std"])
    return f"{mu:.2f} $\\pm$ {sd:.2f}" if latex else f"{mu:.2f} ± {sd:.2f}"


def load_source_record(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cal_key = "prior_cal" if "prior_cal" in data else "pure"
    cal_f10 = data[cal_key]["freq10"]
    pure_f10 = data["pure"]["freq10"]
    cal_full = data[cal_key]["full"]
    return {
        "f10_micro_f1": float(cal_f10["micro_f1"]),
        "f10_macro_f1_cal": float(cal_f10["macro_f1"]),
        "f10_macro_f1_pure": float(pure_f10["macro_f1"]),
        "f10_micro_auc": float(cal_f10["micro_auc"]),
        "f10_macro_auc": float(cal_f10["macro_auc_supported"]),
        "f10_p_at_5": float(cal_f10["p_at_5"]),
        "f10_p_at_8": float(cal_f10["p_at_8"]),
        "f10_p_at_15": float(cal_f10["p_at_15"]),
        "full_micro_f1": float(cal_full["micro_f1"]),
        "full_macro_auc": float(cal_full["macro_auc_supported"]),
        "ms_per_note": float(data.get("milliseconds_per_note", 0.0)),
    }


def load_transfer_record(path: Path) -> dict[str, float]:
    d = json.loads(path.read_text(encoding="utf-8"))
    single = str(d.get("output_mode", "")).startswith("single_label")
    primary = float(d.get("accuracy", 0.0) if single else d.get("micro_f1", 0.0))
    rk = d.get("ranking") or {}
    cg = d.get("code_groups") or {}
    rare = d.get("evaluation_support_1_to_9") or {}
    diag = cg.get("icd10_diag") or {}
    proc = cg.get("icd10_proc") or {}
    if d.get("macro_f1_supported") is not None:
        macro_sup = float(d["macro_f1_supported"])
    elif diag and proc and (diag.get("supported_labels", 0) + proc.get("supported_labels", 0)) > 0:
        n_d = int(diag.get("supported_labels", 0))
        n_p = int(proc.get("supported_labels", 0))
        macro_sup = (
            float(diag.get("macro_f1_supported") or 0.0) * n_d
            + float(proc.get("macro_f1_supported") or 0.0) * n_p
        ) / (n_d + n_p)
    else:
        macro_sup = float(d.get("macro_f1") or 0.0)
    extrap_path = path.parent / "extrapolation.json"
    spearman = float("nan")
    if extrap_path.exists():
        ex = json.loads(extrap_path.read_text(encoding="utf-8"))
        val = (ex.get("stratification") or {}).get("spearman_rho_vs_f1")
        if val is not None:
            spearman = float(val)
    return {
        "primary": primary,
        "precision": float(d.get("precision") or 0.0),
        "recall": float(d.get("recall") or 0.0),
        "macro_f1": float(d.get("macro_f1") or 0.0),
        "macro_sup": macro_sup,
        "diag_f1": float(diag.get("micro_f1") or 0.0),
        "proc_f1": float(proc.get("micro_f1") or 0.0),
        "rare_f1": float(rare.get("micro_f1") or 0.0),
        "rare_macro": float(rare.get("macro_f1_supported") or 0.0),
        "p_at_5": float(rk["p_at_5"]) if "p_at_5" in rk else float("nan"),
        "r_at_50": float(rk["r_at_50"]) if "r_at_50" in rk else float("nan"),
        "top5": float(rk["top5_accuracy"]) if "top5_accuracy" in rk else float("nan"),
        "spearman_rho": spearman,
        "ms_per_note": float(d.get("milliseconds_per_note") or 0.0),
        "compile_s": float(d.get("compile_seconds") or 0.0),
    }


def aggregate_records(records: list[dict[str, float]]) -> dict[str, dict[str, float | int]]:
    if not records:
        return {}
    keys = list(records[0].keys())
    return {
        k: mean_std([r[k] for r in records if k in r and not math.isnan(float(r[k]))])
        for k in keys
    }


def render_multiseed_markdown(
    source_groups: list[tuple[str, list[Path]]],
    transfer_groups: list[tuple[str, dict[str, list[Path]]]],
    target_order: list[str],
) -> tuple[str, dict[str, Any]]:
    summary: dict[str, Any] = {"source": {}, "transfer": {}}
    lines = ["# Stage 5 Multi-Seed Summary (mean ± std across seeds)", ""]

    if source_groups:
        lines.extend([
            "## Table 1: Source MIMIC-IV ICD-9 (`freq>=10`, 5,646 codes; Full 11,324 codes)",
            "",
            "| Method | Seeds | f10 MiF1 | f10 MaF1 (pure) | f10 MaF1 (cal) | f10 MiAUC | f10 MaAUC | Full MaAUC | P@5 | P@8 | P@15 | ms/note |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for label, paths in source_groups:
            recs = [load_source_record(p) for p in paths if p.exists()]
            agg = aggregate_records(recs)
            summary["source"][label] = agg
            lines.append(
                f"| {label} | {len(recs)} | {fmt_pct(agg['f10_micro_f1'])} | {fmt_pct(agg['f10_macro_f1_pure'])} "
                f"| {fmt_pct(agg['f10_macro_f1_cal'])} | {fmt_pct(agg['f10_micro_auc'])} | {fmt_pct(agg['f10_macro_auc'])} "
                f"| {fmt_pct(agg['full_macro_auc'])} | {fmt_pct(agg['f10_p_at_5'])} | {fmt_pct(agg['f10_p_at_8'])} "
                f"| {fmt_pct(agg['f10_p_at_15'])} | {fmt_ms(agg['ms_per_note'])} |"
            )
        lines.append("")

    if transfer_groups:
        header = "| Method | Seeds | " + " | ".join(target_order) + " |"
        sep = "|---|---:|" + "|".join("---:" for _ in target_order) + "|"
        lines.extend(["## Table 2: Zero-Training Transfer Primary Metrics (Micro-F1 / Top-1 Acc)", "", header, sep])
        for label, per_target in transfer_groups:
            summary["transfer"][label] = {}
            cells = []
            seed_count = 0
            for tgt in target_order:
                paths = [p for p in per_target.get(tgt, []) if p.exists()]
                seed_count = max(seed_count, len(paths))
                recs = [load_transfer_record(p) for p in paths]
                agg = aggregate_records(recs)
                summary["transfer"][label][tgt] = agg
                cells.append(fmt_pct(agg["primary"]) if agg else "-")
            lines.append(f"| {label} | {seed_count} | " + " | ".join(cells) + " |")
        lines.append("")

        lines.extend([
            "## Table 3: MIMIC-IV ICD-10 Fine-Grained Zero-Training Decomposition",
            "",
            "| Method | Seeds | Micro-F1 | Diag-F1 | Proc-F1 | Macro-26K | Mac-Sup | Rare (1-9) | P@5 | R@50 | Top-5 | ms/note |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for label, _ in transfer_groups:
            agg = summary["transfer"][label].get("ICD-10", {})
            if not agg:
                continue
            n = int(agg["primary"]["n"])
            lines.append(
                f"| {label} | {n} | {fmt_pct(agg['primary'])} | {fmt_pct(agg['diag_f1'])} | {fmt_pct(agg['proc_f1'])} "
                f"| {fmt_pct(agg['macro_f1'])} | {fmt_pct(agg['macro_sup'])} | {fmt_pct(agg['rare_f1'])} "
                f"| {fmt_pct(agg['p_at_5'])} | {fmt_pct(agg['r_at_50'])} | {fmt_pct(agg['top5'])} | {fmt_ms(agg['ms_per_note'])} |"
            )
        lines.append("")

        r50_header = "| Method | Seeds | " + " | ".join(target_order) + " |"
        lines.extend(["## Table 4: Threshold-Free Retrieval Recall@50 Across Five Clinical Targets", "", r50_header, sep])
        for label, _ in transfer_groups:
            cells = []
            seed_count = 0
            for tgt in target_order:
                agg = summary["transfer"][label].get(tgt, {})
                if agg:
                    seed_count = max(seed_count, int(agg["r_at_50"]["n"]))
                    cells.append(fmt_pct(agg["r_at_50"]))
                else:
                    cells.append("-")
            lines.append(f"| {label} | {seed_count} | " + " | ".join(cells) + " |")
        lines.append("")

    return "\n".join(lines) + "\n", summary


def render_multiseed_latex(summary: dict[str, Any], target_order: list[str]) -> str:
    blocks = ["% Auto-generated Stage 5 multi-seed booktabs tables", ""]
    if summary.get("source"):
        blocks.extend([
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lccccccc}",
            r"\toprule",
            r"Method & f10 MiF1 & f10 MaF1 (pure) & f10 MiAUC & f10 MaAUC & Full MaAUC & P@5 & ms/note \\",
            r"\midrule",
        ])
        for label, agg in summary["source"].items():
            esc = label.replace("_", r"\_").replace("%", r"\%")
            blocks.append(
                f"{esc} & {fmt_pct(agg['f10_micro_f1'], True)} & {fmt_pct(agg['f10_macro_f1_pure'], True)} "
                f"& {fmt_pct(agg['f10_micro_auc'], True)} & {fmt_pct(agg['f10_macro_auc'], True)} "
                f"& {fmt_pct(agg['full_macro_auc'], True)} & {fmt_pct(agg['f10_p_at_5'], True)} "
                f"& {fmt_ms(agg['ms_per_note'], True)} \\\\"
            )
        blocks.extend([r"\bottomrule", r"\end{tabular}", r"\caption{In-domain MIMIC-IV ICD-9 validation performance (mean $\pm$ std across 3 seeds).}", r"\label{tab:source_multiseed}", r"\end{table*}", ""])

    if summary.get("transfer"):
        cols = "l" + "c" * len(target_order)
        blocks.extend([
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            f"\\begin{{tabular}}{{{cols}}}",
            r"\toprule",
            "Method & " + " & ".join(target_order) + r" \\",
            r"\midrule",
        ])
        for label, tgts in summary["transfer"].items():
            esc = label.replace("_", r"\_").replace("%", r"\%")
            cells = [fmt_pct(tgts[t]["primary"], True) if tgts.get(t) else "---" for t in target_order]
            blocks.append(f"{esc} & " + " & ".join(cells) + r" \\")
        blocks.extend([r"\bottomrule", r"\end{tabular}", r"\caption{Zero-training transfer performance across five clinical targets (mean $\pm$ std across 3 seeds).}", r"\label{tab:transfer_multiseed}", r"\end{table*}", ""])

        blocks.extend([
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lcccccccc}",
            r"\toprule",
            r"Method & Micro-F1 & Diag-F1 & Proc-F1 & Mac-Sup & Rare(1--9) & P@5 & R@50 & ms/note \\",
            r"\midrule",
        ])
        for label, tgts in summary["transfer"].items():
            agg = tgts.get("ICD-10")
            if not agg:
                continue
            esc = label.replace("_", r"\_").replace("%", r"\%")
            blocks.append(
                f"{esc} & {fmt_pct(agg['primary'], True)} & {fmt_pct(agg['diag_f1'], True)} "
                f"& {fmt_pct(agg['proc_f1'], True)} & {fmt_pct(agg['macro_sup'], True)} "
                f"& {fmt_pct(agg['rare_f1'], True)} & {fmt_pct(agg['p_at_5'], True)} "
                f"& {fmt_pct(agg['r_at_50'], True)} & {fmt_ms(agg['ms_per_note'], True)} \\\\"
            )
        blocks.extend([r"\bottomrule", r"\end{tabular}", r"\caption{Fine-grained zero-training MIMIC-IV ICD-10 decomposition (26,085 unseen codes, 0 target notes).}", r"\label{tab:icd10_decomp}", r"\end{table*}", ""])

        blocks.extend([
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            f"\\begin{{tabular}}{{{cols}}}",
            r"\toprule",
            "Method & " + " & ".join(target_order) + r" \\",
            r"\midrule",
        ])
        for label, tgts in summary["transfer"].items():
            esc = label.replace("_", r"\_").replace("%", r"\%")
            cells = [fmt_pct(tgts[t]["r_at_50"], True) if tgts.get(t) else "---" for t in target_order]
            blocks.append(f"{esc} & " + " & ".join(cells) + r" \\")
        blocks.extend([r"\bottomrule", r"\end{tabular}", r"\caption{Threshold-free Recall@50 across five zero-training clinical targets (mean $\pm$ std across 3 seeds).}", r"\label{tab:r50_multiseed}", r"\end{table*}", ""])
    return "\n".join(blocks) + "\n"
