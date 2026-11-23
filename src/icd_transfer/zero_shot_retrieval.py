import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .calibration import compile_calibration, fit_codebook_alignment
from .codebook import load_codebook
from .data import Notes, load_tokenizer, targets_for
from .eval_isotoken import _align_raw_vectors
from .evaluate_transfer import analysis_groups, prediction_keys, transfer_data
from .extrapolation import analyze_extrapolation_routing
from .inference import autocast, sync
from .metrics import Counts
from .schema_algebra import apply_group_algebra, compile_group_index
from .semantic_encoder import load_semantic_backbone, load_semantic_cache, pool_semantic_hidden
from .transfer_metrics import TransferCounts


def _decode_row_chunks(rows, source_tokenizer):
    texts = []
    counts = []
    for row in rows:
        chunks = row["chunks"]
        counts.append(len(chunks))
        decoded = source_tokenizer.batch_decode(chunks, skip_special_tokens=True)
        texts.extend([t.strip() or " " for t in decoded])
    return texts, counts


@torch.inference_mode()
def encode_note_chunk_embeddings(
    rows,
    source_tokenizer,
    sem_tokenizer,
    sem_model,
    pooling,
    device,
    chunk_batch_size=128,
    max_length=320,
):
    """Encode all chunks in `rows` with the semantic backbone, returning normalized [N_chunks, D] + counts."""
    texts, counts = _decode_row_chunks(rows, source_tokenizer)
    max_pos = getattr(getattr(sem_model, "config", None), "max_position_embeddings", None)
    eff_max_len = min(max_length, max(int(max_pos) - 2, 32)) if max_pos is not None else max_length
    out = []
    for start in range(0, len(texts), chunk_batch_size):
        batch = texts[start:start + chunk_batch_size]
        encoded = sem_tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=eff_max_len,
            return_tensors="pt",
        )
        ids = encoded["input_ids"].to(device)
        mask = encoded["attention_mask"].to(device)
        with autocast(device):
            hidden = sem_model(input_ids=ids, attention_mask=mask).last_hidden_state
            pooled = pool_semantic_hidden(hidden, mask, pooling)
        out.append(F.normalize(pooled.float(), dim=-1, eps=1e-6).cpu())
    return torch.cat(out, dim=0), counts


def score_chunk_embeddings(chunk_embs, chunk_counts, prototypes, note_pool="max", label_chunk_size=4096, device="cpu"):
    """Compute [B, C] cosine similarity between note chunks and unit-normalized target prototypes."""
    proto = prototypes.to(device=device)
    embs = chunk_embs.to(device=device)
    use_bf16 = embs.is_cuda and torch.cuda.is_bf16_supported()
    embs_mm = embs.bfloat16() if use_bf16 else embs
    proto_mm = proto.bfloat16() if use_bf16 else proto

    num_notes = len(chunk_counts)
    total_codes = len(proto)
    out_chunks = []
    for start in range(0, total_codes, label_chunk_size):
        p_chunk = proto_mm[start:start + label_chunk_size]
        sim = torch.matmul(embs_mm, p_chunk.T).float()  # [N_chunks, C_chunk]
        note_scores = torch.empty((num_notes, sim.shape[1]), device=device, dtype=torch.float32)
        offset = 0
        for i, count in enumerate(chunk_counts):
            slice_sim = sim[offset:offset + count]
            note_scores[i] = slice_sim.max(dim=0).values if note_pool == "max" else slice_sim.mean(dim=0)
            offset += count
        out_chunks.append(note_scores)
    return torch.cat(out_chunks, dim=1)


def _calibrate_source_threshold(
    source_val_dir,
    source_vecs,
    source_tokenizer,
    sem_tokenizer,
    sem_model,
    pooling,
    device,
    note_pool="max",
    scale=15.0,
    limit=2000,
    chunk_batch_size=128,
    max_length=320,
    label_chunk_size=4096,
):
    val = Notes(source_val_dir, "validation")
    if val.meta["icd_version"] != 9:
        raise ValueError("Source calibration must use ICD-9 validation only")
    codes = load_codebook(Path(source_val_dir) / "codebook.json")
    counts_meta = val.meta.get("label_counts", {})
    freq10_indices = np.asarray([i for i, c in enumerate(codes) if int(counts_meta.get(c.key, 0)) >= 10], dtype=np.int64)
    proto = F.normalize(source_vecs.float(), dim=-1, eps=1e-6).to(device)
    n_eval = min(len(val), limit) if limit else len(val)
    cos_grid = np.arange(0.15, 0.86, 0.025)
    counts = Counts(cos_grid, len(codes))
    batch_notes = 64
    for start in range(0, n_eval, batch_notes):
        rows = [val[i] for i in range(start, min(n_eval, start + batch_notes))]
        target = targets_for(rows, codes, torch.device("cpu")).numpy()
        embs, ch_counts = encode_note_chunk_embeddings(
            rows, source_tokenizer, sem_tokenizer, sem_model, pooling, device,
            chunk_batch_size=chunk_batch_size, max_length=max_length,
        )
        # Center chunk embeddings with the exact same source mean only if raw; standard cosine uses unit norm directly.
        scores = score_chunk_embeddings(
            embs, ch_counts, proto, note_pool=note_pool, label_chunk_size=label_chunk_size, device=device,
        ).cpu().numpy()
        counts.update(scores, target)
    best = counts.best()
    cos_thr = float(counts.thresholds[best])
    res = counts.result(best)
    if len(freq10_indices):
        f10 = counts.result(best, indices=freq10_indices)
        res["freq10_micro_f1"] = f10["micro_f1"]
        res["freq10_macro_f1"] = f10["macro_f1"]
    return {
        "cos_threshold": cos_thr,
        "scale": float(scale),
        "bias": float(-scale * cos_thr),
        "prob_threshold": 0.5,
        "source_val_examples": n_eval,
        "source_val_metrics": res,
        "source_subjects": sorted(set(json.loads((Path(source_val_dir) / "subjects.json").read_text()))),
    }


def evaluate_zeroshot_retrieval(args):
    """Run zero-training pure dense semantic retrieval baseline (e.g. Qwen3, MedCPT, BioLORD)."""
    device = torch.device(args.device)
    source_tokenizer = load_tokenizer(args.model_dir)
    note_model_spec = getattr(args, "note_semantic_model", None) or args.semantic_model
    sem_tokenizer, sem_model, pooling = load_semantic_backbone(note_model_spec, device, pooling=args.semantic_pooling)
    source_codes = load_codebook(Path(args.source_val_data) / "codebook.json")
    source_vecs, sem_meta = load_semantic_cache(args.source_semantic_cache, source_codes, device=device)

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    cal_file = out_root / "source_calibration.json"
    if cal_file.exists():
        src_cal = json.loads(cal_file.read_text())
    else:
        src_cal = _calibrate_source_threshold(
            args.source_val_data,
            source_vecs,
            source_tokenizer,
            sem_tokenizer,
            sem_model,
            pooling,
            device,
            note_pool=args.note_pool,
            scale=args.logit_scale,
            limit=args.source_val_limit,
            chunk_batch_size=args.chunk_batch_size,
            max_length=args.max_length,
            label_chunk_size=args.label_chunk_size,
        )
        # Also exclude source training patients if provided.
        if args.source_train_subjects:
            train_subj = set(json.loads(Path(args.source_train_subjects).read_text()))
            src_cal["source_subjects"] = sorted(set(src_cal["source_subjects"]) | train_subj)
        cal_file.write_text(json.dumps(src_cal, indent=2) + "\n")

    if not args.data:
        print(json.dumps(src_cal, indent=2))
        return src_cal

    data_dir = Path(args.data)
    split = args.split
    metadata, table = transfer_data(data_dir, split, args.approve_test)

    codes = load_codebook(data_dir / "codebook.json")
    group_file = data_dir / "groups.json"
    compositions = json.loads(group_file.read_text()) if group_file.exists() else []
    target_vecs_raw, _ = load_semantic_cache(args.target_semantic_cache, codes, device=device)

    excluded = set(src_cal["source_subjects"])
    eligible_rows = []
    skipped = 0
    for row in table.to_pylist():
        if metadata.get("patient_id_namespace") == "mimiciv" and row["subject_id"] in excluded:
            skipped += 1
            continue
        eligible_rows.append(row)
        if args.validation_limit and len(eligible_rows) >= args.validation_limit:
            break
    if not eligible_rows:
        raise ValueError("No eligible target examples remained")

    sync(device)
    enc_started = time.perf_counter()
    note_batches = []
    for start in range(0, len(eligible_rows), args.batch_size):
        batch_rows = eligible_rows[start:start + args.batch_size]
        embs, ch_counts = encode_note_chunk_embeddings(
            batch_rows, source_tokenizer, sem_tokenizer, sem_model, pooling, device,
            chunk_batch_size=args.chunk_batch_size, max_length=args.max_length,
        )
        note_batches.append((batch_rows, embs, ch_counts))
    sync(device)
    encode_seconds = time.perf_counter() - enc_started

    mode = metadata.get("output_mode", "multilabel")
    scale = float(src_cal["scale"])
    bias = float(src_cal["bias"])
    threshold = float(src_cal["prob_threshold"])

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    results = {}
    for tag in modes:
        align_mode = "none"
        prior_shift = 0.0
        crowding_alpha = 0.0
        group_alg = "none"
        if tag == "raw":
            pass
        elif tag == "crowd":
            crowding_alpha = 1.0
        elif tag == "full":
            prior_shift = 1.0
            crowding_alpha = 1.0
        elif tag == "mean-crowd":
            align_mode = "mean"
            crowding_alpha = 1.0
        elif tag == "bw-crowd":
            align_mode = "bures_wasserstein"
            crowding_alpha = 1.0
        elif tag == "bw-full":
            align_mode = "bures_wasserstein"
            prior_shift = 1.0
            crowding_alpha = 1.0
        elif tag == "bw-crowd-fkg":
            align_mode = "bures_wasserstein"
            crowding_alpha = 1.0
            group_alg = "frechet_clamp"
        elif tag == "bw-full-fkg":
            align_mode = "bures_wasserstein"
            prior_shift = 1.0
            crowding_alpha = 1.0
            group_alg = "frechet_clamp"
        else:
            raise ValueError(f"Unknown zero-shot eval mode tag: {tag}")

        sync(device)
        t_comp = time.perf_counter()
        target_vecs = target_vecs_raw
        alignment = None
        if align_mode != "none":
            alignment = fit_codebook_alignment(source_vecs, target_vecs_raw, mode=align_mode, reg=args.codebook_align_reg)
            target_vecs = _align_raw_vectors(target_vecs_raw, alignment)
        prototypes = F.normalize(target_vecs.float(), dim=-1, eps=1e-6).to(device)
        calibration = compile_calibration(
            source_vecs,
            target_vecs,
            mode,
            source_count=len(source_codes),
            target_count=len(codes),
            device=device,
            prior_shift_scale=prior_shift,
            crowding_alpha=crowding_alpha,
            crowding_k=args.crowding_k,
            whiten_eps=args.whiten_eps,
            residual_ridge=args.residual_ridge,
            record_geometry=True,
            residual_target_vectors=target_vecs_raw,
        )
        compiled_groups = compile_group_index(codes, compositions) if (compositions and group_alg != "none") else None
        sync(device)
        compile_seconds = time.perf_counter() - t_comp

        counts = TransferCounts([code.key for code in codes], mode)
        sub_out = out_root / f"{data_dir.name}-{split}-{tag}"
        sub_out.mkdir(parents=True, exist_ok=True)
        patients = set()
        missing_patient = 0
        sync(device)
        t_score = time.perf_counter()
        with (sub_out / "predictions.jsonl").open("w") as stream:
            for batch_rows, embs, ch_counts in note_batches:
                cos_sim = score_chunk_embeddings(
                    embs, ch_counts, prototypes,
                    note_pool=args.note_pool,
                    label_chunk_size=args.label_chunk_size,
                    device=device,
                )
                logits = scale * cos_sim + bias + calibration.logit_bias.unsqueeze(0)
                if compiled_groups is not None:
                    logits = apply_group_algebra(
                        logits, codes, compositions, mode=group_alg, compiled_groups=compiled_groups,
                    )
                probs_gpu = logits if mode.startswith("single_label") else torch.sigmoid(logits)
                top_k = min(50, len(codes))
                top_idx_batch = torch.topk(probs_gpu, k=top_k, dim=-1).indices.cpu().tolist()
                probs = probs_gpu.cpu()
                for row, row_probs, row_top in zip(batch_rows, probs, top_idx_batch, strict=True):
                    predicted = prediction_keys(row_probs, codes, mode, threshold)
                    tp, fp, fn = counts.update(predicted, row)
                    counts.update_ranking([codes[idx].key for idx in row_top[:50]], row)
                    missing_patient += row["subject_id"] is None
                    patients.add(row["subject_id"])
                    stream.write(json.dumps({
                        "id": row["id"],
                        "subject_id": row["subject_id"],
                        "predicted": predicted,
                        "truth": row["labels"],
                        "label_groups": row.get("label_groups", []),
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                    }, ensure_ascii=False) + "\n")
        sync(device)
        score_seconds = time.perf_counter() - t_score
        total_infer_seconds = encode_seconds + score_seconds

        metrics = {
            **counts.result(),
            "model": "zeroshot_retrieval_v1",
            "arch": "zeroshot_retrieval",
            "kind": "zeroshot_retrieval",
            "head_mode": f"zeroshot_{args.note_pool}",
            "semantic_model": str(args.semantic_model),
            "dataset": metadata["dataset"],
            "split": split,
            "output_mode": mode,
            "threshold": threshold,
            "cos_threshold": src_cal["cos_threshold"],
            "examples": counts.examples,
            "patients": len(patients - {None}),
            "missing_patient_examples": missing_patient,
            "excluded_source_patient_examples": skipped,
            "compile_seconds": compile_seconds,
            "encode_seconds": encode_seconds,
            "score_seconds": score_seconds,
            "inference_seconds": total_infer_seconds,
            "ms_per_note": 1000.0 * total_infer_seconds / max(counts.examples, 1),
            "group_algebra": group_alg,
            "codebook_align": align_mode,
            "calibration": calibration.summary,
            "codebook_alignment": alignment.summary if alignment is not None else None,
            "note_semantic_model": str(note_model_spec),
        }
        if mode == "multilabel":
            metrics["code_groups"] = {name: counts.label_result(keys) for name, keys in analysis_groups(codes).items()}
            bench_groups_path = getattr(args, "benchmark_groups", None)
            if bench_groups_path:
                bg_file = Path(bench_groups_path)
                if not bg_file.exists():
                    raise FileNotFoundError(f"benchmark_groups file not found: {bg_file}")
                bg = json.loads(bg_file.read_text())
                key_set = {c.key for c in codes}
                metrics["benchmark_groups"] = {
                    name: counts.label_result([k for k in gkeys if k in key_set])
                    for name, gkeys in bg.items()
                }
        torch.save({
            "keys": [code.key for code in codes],
            "crowding": calibration.crowding.cpu() if calibration.crowding is not None else None,
            "residual": calibration.residual.cpu() if calibration.residual is not None else None,
            "logit_bias": calibration.logit_bias.cpu(),
            "summary": calibration.summary,
        }, sub_out / "geometry.pt")
        (sub_out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        if calibration.residual is not None:
            routing = analyze_extrapolation_routing(sub_out)
            (sub_out / "extrapolation.json").write_text(json.dumps(routing, indent=2) + "\n")
            metrics["extrapolation"] = routing
            (sub_out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        results[tag] = {
            "primary": metrics.get("accuracy") if mode.startswith("single_label") else metrics["micro_f1"],
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "tp": metrics.get("tp"),
            "fp": metrics.get("fp"),
            "ms_per_note": metrics["ms_per_note"],
        }
    print(json.dumps({"target": data_dir.name, "split": split, "results": results}, indent=2))
    return results
