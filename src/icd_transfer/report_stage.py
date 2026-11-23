import json
from pathlib import Path

from .extrapolation import analyze_extrapolation_routing


def _load_json(path):
    return json.loads(Path(path).read_text())


def summarize_run(run_dir):
    root = Path(run_dir)
    train_summary_path = root / "train" / "summary.json"
    source = None
    if train_summary_path.exists():
        summary = _load_json(train_summary_path)
        best = summary.get("best", {})
        best_step = best.get("step")
        val_report = None
        if best_step is not None:
            cand = root / "train" / f"validation-{int(best_step):08d}.json"
            if cand.exists():
                val_report = _load_json(cand)
        args_path = root / "train" / "arguments.json"
        train_args = _load_json(args_path) if args_path.exists() else {}
        atomic = (val_report or {}).get("atomic", {})
        source = {
            "head_mode": train_args.get("head_mode", "mlp"),
            "views": train_args.get("views"),
            "schema_mode": train_args.get("schema_mode"),
            "view_dropout": train_args.get("view_dropout", 0.0),
            "residual_dropout": train_args.get("residual_dropout", 0.0),
            "best_step": best_step,
            "selection_score": best.get("score"),
            "micro_f1": atomic.get("micro_f1"),
            "macro_f1": atomic.get("macro_f1"),
            "macro_f1_supported": atomic.get("macro_f1_supported"),
            "threshold": (val_report or {}).get("threshold"),
        }

    evals = {}
    eval_root = root / "eval"
    if eval_root.exists():
        for eval_dir in sorted(p for p in eval_root.iterdir() if p.is_dir()):
            metrics_path = eval_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            metrics = _load_json(metrics_path)
            entry = {
                "dataset": metrics.get("dataset"),
                "split": metrics.get("split"),
                "micro_f1": metrics.get("micro_f1"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "example_macro_f1": metrics.get("example_macro_f1"),
                "tp": metrics.get("tp"),
                "fp": metrics.get("fp"),
                "fn": metrics.get("fn"),
                "compile_seconds": metrics.get("compile_seconds"),
                "ms_per_note": metrics.get("milliseconds_per_note"),
                "group_algebra": metrics.get("group_algebra"),
                "calibration": metrics.get("calibration"),
                "attention_cast": metrics.get("attention_cast"),
                "subgroups": {
                    k: v.get("micro_f1")
                    for k, v in (metrics.get("subgroups") or {}).items()
                    if isinstance(v, dict)
                },
            }
            if (eval_dir / "geometry.pt").exists() and (eval_dir / "predictions.jsonl").exists():
                routing = analyze_extrapolation_routing(eval_dir)
                quantiles = routing["stratification"]["quantiles"]
                curve20 = next((c for c in routing["escalation_curve"] if abs(c["budget"] - 0.2) < 1e-9), None)
                entry["extrapolation"] = {
                    "supported_codes": routing["stratification"]["supported_codes"],
                    "spearman_rho_vs_f1": routing["stratification"]["spearman_rho_vs_f1"],
                    "rho_q1_mean": quantiles[0]["rho_mean"] if quantiles else None,
                    "rho_q4_mean": quantiles[-1]["rho_mean"] if quantiles else None,
                    "q1_micro_f1": quantiles[0]["micro_f1"] if quantiles else None,
                    "q4_micro_f1": quantiles[-1]["micro_f1"] if quantiles else None,
                    "error_capture_20": curve20["error_capture_rate"] if curve20 else None,
                    "system2_f1_20": curve20["system2_micro_f1"] if curve20 else None,
                    "random_f1_20": curve20["random_micro_f1"] if curve20 else None,
                }
            evals[eval_dir.name] = entry

    return {"run": root.name, "source": source, "evals": evals}


def summarize_runs(run_dirs):
    return [summarize_run(path) for path in run_dirs]


def format_markdown_summary(reports):
    lines = ["| Run | Head | Source Micro | Source MacroSup | Eval Slice | Micro-F1 / Acc | Precision | Recall | FP | ms/note | Q1->Q4 F1 | Cap@20% |",
             "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|"]
    for report in reports:
        src = report["source"] or {}
        head = src.get("head_mode", "-")
        src_micro = f"{src['micro_f1']:.4f}" if src.get("micro_f1") is not None else "-"
        src_macro = f"{src['macro_f1_supported']:.4f}" if src.get("macro_f1_supported") is not None else "-"
        for name, ev in report["evals"].items():
            score = ev["accuracy"] if ev.get("accuracy") is not None else ev.get("micro_f1")
            score_s = f"{score:.4f}" if score is not None else "-"
            prec_s = f"{ev['precision']:.4f}" if ev.get("precision") is not None else "-"
            rec_s = f"{ev['recall']:.4f}" if ev.get("recall") is not None else "-"
            fp_s = str(ev.get("fp", "-"))
            ms_s = f"{ev['ms_per_note']:.1f}" if ev.get("ms_per_note") is not None else "-"
            ext = ev.get("extrapolation")
            if ext and ext.get("q1_micro_f1") is not None:
                q_s = f"{ext['q1_micro_f1']:.3f}->{ext['q4_micro_f1']:.3f}"
                cap_s = f"{100.0 * ext['error_capture_20']:.1f}%" if ext.get("error_capture_20") is not None else "-"
            else:
                q_s, cap_s = "-", "-"
            lines.append(
                f"| {report['run']} | {head} | {src_micro} | {src_macro} | {name} | {score_s} | {prec_s} | {rec_s} | {fp_s} | {ms_s} | {q_s} | {cap_s} |"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize Stage 1/1.5/3 run artifacts.")
    parser.add_argument("run_dirs", nargs="+", help="One or more pulled run directories.")
    parser.add_argument("--json-out", default=None, help="Optional JSON summary output path.")
    args = parser.parse_args()
    reports = summarize_runs(args.run_dirs)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(reports, indent=2))
    print(format_markdown_summary(reports))
