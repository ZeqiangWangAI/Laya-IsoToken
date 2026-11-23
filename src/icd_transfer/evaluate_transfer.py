import json
import time
from pathlib import Path

import pyarrow.feather as feather
import torch

from .attention_cast import compile_attention_cast, fit_attention_cast
from .baseline_transfer import baseline_logits, compile_baseline
from .calibration import (
    apply_codebook_alignment,
    compile_calibration,
    fit_codebook_alignment,
    primary_semantic_vectors,
    project_out_note_shift,
    residual_buckets,
)
from .codebook import Code, load_codebook
from .data import load_tokenizer
from .inference import autocast, embed_codebook, score_chunks, sync
from .label_views import encode_views, membership_matrix
from .models import build_model
from .schema_algebra import apply_group_algebra, compile_group_index
from .transfer_data import save_json
from .transfer_metrics import TransferCounts


def transfer_data(directory, split, approved):
    if split == "test" and not approved:
        raise PermissionError("First test evaluation requires explicit user approval")
    root = Path(directory)
    metadata = json.loads((root / "metadata.json").read_text())
    if (metadata["role"] != "target_evaluation_only" or metadata["model_fit_allowed"]
            or metadata["threshold_selection_allowed"]):
        raise ValueError("Frozen evaluation requires a target-only dataset")
    if split not in metadata["splits"]:
        raise ValueError("Requested split is absent from the prepared target")
    return metadata, feather.read_table(root / f"{split}.arrow", memory_map=True)


def source_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "feature_metadata" not in state:
        best = state["source_selection"]
        if best is None or best["step"] != state["step"] or state["threshold"] != best["threshold"]:
            raise ValueError("Evaluate the source-selected checkpoint identified in summary.json")
        if best["training_icd_version"] != 9 or best["validation_icd_version"] != 9:
            raise ValueError("Transfer requires source-only model selection")
        return state, best["threshold"]
    best = state["progress"]["best"]
    if best is None or best["step"] != state["progress"]["step"]:
        raise ValueError("Evaluate the source-selected checkpoint identified in summary.json")
    if state["feature_metadata"]["role"] != "source_frozen_encoder_features":
        raise ValueError("Transfer requires a source-only model checkpoint")
    return state, best["threshold"]


def load_transfer_model(state, codes, args):
    source_codes = [Code(**{**row, "ancestors": tuple(row.get("ancestors", []))}) for row in state["source_codes"]]
    if "feature_metadata" in state:
        args.label_max_len = state["feature_metadata"]["label_max_len"]
        head_mode = state["args"].get("head_mode", "mlp")
        residual_dropout = state["args"].get("residual_dropout", 0.25)
        view_dropout = state["args"].get("view_dropout", 0.5)
        model = build_model(
            "multiview",
            args.model_dir,
            source_codes,
            state["args"]["projection_dim"],
            head_mode=head_mode,
            residual_dropout=residual_dropout,
            view_dropout=view_dropout,
        )
        model.head.load_state_dict(state["model"], strict=True)
        return model, "multiview", source_codes, state["feature_metadata"]["encoder"], state["feature_metadata"]["source_subjects"]
    settings = state["args"]
    for key in ("head_max_len", "label_max_len", "ridge"):
        setattr(args, key, settings[key])
    model = build_model(settings["kind"], args.model_dir, source_codes, settings["projection_dim"], initialize=False)
    model.load_state_dict(state["model"], strict=True)
    return model, settings["kind"], source_codes, state["tokenizer_origin"], state["source_subjects"]


def semantic_vectors_for_calibration(model, kind, labels, codes, source_codes, compositions, tokenizer, args):
    if kind == "native":
        return None, None
    if kind == "multiview":
        source_views = encode_views(model, source_codes, [], tokenizer, args)
        return primary_semantic_vectors(source_views), primary_semantic_vectors(labels)
    source_vecs = embed_codebook(model, source_codes, tokenizer, args)
    if kind != "linear":
        return source_vecs, labels
    positions = {code.key: i for i, code in enumerate(codes)}
    composed = {group["key"] for group in compositions}
    atomic = [i for i, code in enumerate(codes) if code.key not in composed]
    embeddings = embed_codebook(model, [codes[i] for i in atomic], tokenizer, args)
    target_vecs = embeddings.new_zeros((len(codes), embeddings.shape[-1]))
    target_vecs[atomic] = embeddings
    if compositions:
        matrix = membership_matrix(codes, compositions, args.device)
        with torch.autocast(device_type=matrix.device.type, enabled=False):
            pooled = torch.sparse.mm(matrix, target_vecs.float())
        target_vecs[[positions[group["key"]] for group in compositions]] = pooled.to(target_vecs.dtype)
    return source_vecs, target_vecs


def primary_views(labels):
    missing = ~labels.available[:, 0]
    labels.vectors[missing, 0] = labels.vectors[missing, 2]
    labels.available[:] = False
    labels.available[:, 0] = True
    labels.operators.zero_()
    return labels


def prediction_keys(probabilities, codes, mode, threshold):
    if mode.startswith("single_label"):
        return [codes[int(probabilities.argmax())].key]
    indices = torch.nonzero(probabilities >= threshold).flatten().tolist()
    return [codes[i].key for i in indices]


def analysis_groups(codes):
    groups = {}
    for code in codes:
        groups.setdefault(code.system, []).append(code.key)
        if code.system.endswith("_proc") and len(code.code) in (4, 7):
            groups.setdefault(f"{code.system}_{len(code.code)}_characters", []).append(code.key)
    return groups


@torch.inference_mode()
def evaluate_transfer(args):
    metadata, table = transfer_data(args.data, args.split, args.test_approved)
    state, threshold = source_checkpoint(args.checkpoint)
    origin = json.loads((Path(args.model_dir) / "origin.json").read_text())
    codes = load_codebook(Path(args.data) / "codebook.json")
    group_file = Path(args.data) / "groups.json"
    compositions = json.loads(group_file.read_text()) if group_file.exists() else []
    model, kind, source_codes, encoder_origin, source_subjects = load_transfer_model(state, codes, args)
    if metadata["tokenizer"] != origin or encoder_origin != origin:
        raise ValueError("Source model and target tokens must use the same fixed encoder revision")
    model.to(args.device).requires_grad_(False).eval()
    tokenizer = load_tokenizer(args.model_dir)
    mode = metadata.get("output_mode", "multilabel")
    prior_shift_scale = getattr(args, "prior_shift_scale", 0.0)
    crowding_alpha = getattr(args, "crowding_alpha", 0.0)
    crowding_k = getattr(args, "crowding_k", 10)
    whiten_eps = getattr(args, "whiten_eps", 1e-4)
    residual_ridge = getattr(args, "residual_ridge", 1.0)
    record_geometry = getattr(args, "record_geometry", False)
    group_algebra = getattr(args, "group_algebra", "none")
    cast_scale = getattr(args, "cast_scale", 1.0)
    cast_ridge = getattr(args, "cast_ridge", 0.0)
    cast_rank = getattr(args, "cast_rank", 0)
    cast_weighting = getattr(args, "cast_weighting", "none")
    codebook_align = getattr(args, "codebook_align", "none")
    codebook_align_reg = getattr(args, "codebook_align_reg", 0.05)
    note_shift_proj = getattr(args, "note_shift_proj", False)
    compiled_groups = compile_group_index(codes, compositions) if (compositions and group_algebra != "none") else None
    calibration = None
    cast_summary = None
    alignment = None
    sync(args.device)
    started = time.perf_counter()
    with autocast(args.device):
        if kind == "multiview":
            labels = encode_views(model, codes, compositions, tokenizer, args)
            if state["args"]["views"] == "definition":
                labels = primary_views(labels)
        else:
            labels = compile_baseline(model, kind, codes, source_codes, compositions, tokenizer, args)
        has_cast = kind == "multiview" and hasattr(model.head, "delta_query")
        need_cast = has_cast and cast_scale != 0.0
        need_align = kind == "multiview" and (codebook_align != "none" or note_shift_proj)
        need_cal_vecs = crowding_alpha != 0.0 or record_geometry
        source_vecs, target_vecs_raw = (None, None)
        if need_cast or need_align or need_cal_vecs:
            source_vecs, target_vecs_raw = semantic_vectors_for_calibration(
                model, kind, labels, codes, source_codes, compositions, tokenizer, args,
            )
        target_vecs = target_vecs_raw
        if need_align:
            alignment = fit_codebook_alignment(
                source_vecs, target_vecs_raw, mode=codebook_align, reg=codebook_align_reg,
            )
            labels = apply_codebook_alignment(labels, alignment)
            target_vecs = primary_semantic_vectors(labels)
        if has_cast:
            if need_cast:
                operator = fit_attention_cast(
                    source_vecs,
                    model.head.delta_query,
                    model.head.delta_value,
                    support_counts=state.get("source_label_counts"),
                    ridge=cast_ridge if cast_ridge > 0.0 else None,
                    whiten_eps=whiten_eps,
                    weighting=cast_weighting,
                    rank=cast_rank,
                )
                dq_tgt, dv_tgt = compile_attention_cast(operator, target_vecs, scale=cast_scale)
                labels = labels.with_residuals(dq_tgt, dv_tgt)
                cast_summary = {**operator.summary, "scale": float(cast_scale), "enabled": True}
            else:
                cast_summary = {"scale": 0.0, "enabled": False}
        if prior_shift_scale != 0.0 or crowding_alpha != 0.0 or record_geometry:
            calibration = compile_calibration(
                source_vecs,
                target_vecs,
                mode,
                source_count=len(source_codes),
                target_count=len(codes),
                device=args.device,
                prior_shift_scale=prior_shift_scale,
                crowding_alpha=crowding_alpha,
                crowding_k=crowding_k,
                whiten_eps=whiten_eps,
                residual_ridge=residual_ridge,
                record_geometry=record_geometry,
                residual_target_vectors=target_vecs_raw,
            )
    sync(args.device)
    compile_seconds = time.perf_counter() - started
    counts = TransferCounts([code.key for code in codes], mode)
    excluded = set(source_subjects)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    skipped, missing_patient, patients = 0, 0, set()
    sync(args.device)
    started = time.perf_counter()
    with (output / "predictions.jsonl").open("x") as stream:
        for batch in table.to_batches(max_chunksize=args.batch_size):
            if args.validation_limit and counts.examples >= args.validation_limit:
                break
            rows = batch.to_pylist()
            if metadata.get("patient_id_namespace") == "mimiciv":
                eligible = [row for row in rows if row["subject_id"] not in excluded]
                skipped += len(rows) - len(eligible)
                rows = eligible
            if args.validation_limit:
                rows = rows[:args.validation_limit - counts.examples]
            if not rows:
                continue
            with autocast(args.device):
                if kind == "multiview":
                    notes = model.encode_notes(rows, tokenizer.pad_token_id, args.device)
                    hidden_batch = [
                        project_out_note_shift(hidden, alignment) if (note_shift_proj and alignment is not None) else hidden
                        for hidden, _ in notes
                    ]
                    if hasattr(model.head, "score_batch"):
                        batched = torch.cat(
                            [model.head.score_batch(hidden_batch, labels[start:start + args.label_batch_size])
                             for start in range(0, len(labels), args.label_batch_size)],
                            dim=1,
                        )
                        scores = [batched[i] for i in range(len(notes))]
                    else:
                        scores = [score_chunks(model, hidden, labels, args.label_batch_size) for hidden in hidden_batch]
                else:
                    scores = baseline_logits(model, kind, rows, labels, tokenizer, args)
                if calibration is not None:
                    scores = [score + calibration.logit_bias for score in scores]
                if compiled_groups is not None:
                    stacked = apply_group_algebra(
                        torch.stack(scores, dim=0), codes, compositions, mode=group_algebra, compiled_groups=compiled_groups,
                    )
                    scores = [stacked[i] for i in range(len(scores))]
                probabilities = scores if mode.startswith("single_label") else [score.sigmoid() for score in scores]
            for row, probability in zip(rows, probabilities, strict=True):
                predicted = prediction_keys(probability, codes, mode, threshold)
                tp, fp, fn = counts.update(predicted, row)
                missing_patient += row["subject_id"] is None
                patients.add(row["subject_id"])
                record = {"id": row["id"], "subject_id": row["subject_id"], "predicted": predicted,
                          "truth": row["labels"], "label_groups": row["label_groups"], "tp": tp, "fp": fp, "fn": fn}
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    sync(args.device)
    elapsed = time.perf_counter() - started
    result = {**counts.result(), "dataset": metadata["dataset"], "split": args.split, "output_mode": mode,
              "checkpoint": str(args.checkpoint), "threshold": threshold, "threshold_source": "ICD-9 validation",
              "source_patient_rows_excluded": skipped, "rows_without_patient_id": missing_patient,
              "patients_with_id": len(patients - {None}), "codebook_seconds": compile_seconds,
              "inference_seconds": elapsed, "milliseconds_per_note": 1000 * elapsed / max(counts.examples, 1),
              "model_fit_on_target": False, "kind": kind, "views": state["args"].get("views", "baseline"),
              "head_mode": state["args"].get("head_mode", "mlp") if kind == "multiview" else "baseline"}
    if mode == "multilabel":
        result["code_groups"] = {key: counts.label_result(keys) for key, keys in analysis_groups(codes).items()}
    if group_algebra != "none":
        result["group_algebra"] = group_algebra
    if alignment is not None:
        result["codebook_alignment"] = {**alignment.summary, "note_shift_proj": bool(note_shift_proj)}
    if cast_summary is not None:
        result["attention_cast"] = cast_summary
    if calibration is not None:
        result["calibration"] = calibration.summary
        if calibration.residual is not None:
            torch.save(
                {"keys": [code.key for code in codes], "crowding": calibration.crowding.cpu(),
                 "residual": calibration.residual.cpu()},
                output / "geometry.pt",
            )
            if mode == "multilabel":
                result["residual_buckets"] = residual_buckets(counts, [code.key for code in codes], calibration.residual)
    save_json(output / "metrics.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result
