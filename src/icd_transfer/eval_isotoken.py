import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .attention_cast import compile_attention_cast, fit_attention_cast
from .calibration import (
    apply_span_semantic_nms,
    compile_calibration,
    compile_cms_gem_crosswalk,
    fit_codebook_alignment,
    residual_buckets,
    resolve_zero_target_policy,
    semantic_prior_projection,
    wiener_spherical_gain,
)
from .codebook import Code, load_codebook
from .data import Notes, load_tokenizer
from .evaluate_transfer import analysis_groups, prediction_keys, transfer_data
from .extrapolation import analyze_extrapolation_routing
from .inference import autocast, sync
from .isotoken import build_isotoken_model
from .metrics import (
    CalibrationTracker,
    Counts,
    binary_auprc_exact,
    binary_roc_auc_exact,
    histogram_auprc,
    histogram_roc_auc,
)
from .schema_algebra import apply_group_algebra, compile_group_index
from .semantic_encoder import encode_codebook_semantic, load_semantic_cache, save_semantic_cache
from .transfer_data import save_json
from .transfer_metrics import TransferCounts


def _load_or_encode_semantic(cache_path, codes, compositions, semantic_model, text_mode, pooling, batch_size, max_len, device):
    if cache_path and Path(cache_path).exists():
        vecs, _ = load_semantic_cache(cache_path, codes, device=device)
        return vecs
    vecs = encode_codebook_semantic(
        codes,
        compositions,
        model_spec=semantic_model,
        device=device,
        text_mode=text_mode,
        batch_size=batch_size,
        max_length=max_len,
        pooling=pooling,
    )
    if cache_path:
        save_semantic_cache(cache_path, codes, vecs, model_spec=semantic_model, text_mode=text_mode, pooling=pooling)
    return vecs


def _align_raw_vectors(target_raw, alignment):
    if alignment is None or alignment.mode == "none":
        return target_raw
    mu_a = alignment.source_mean.to(device=target_raw.device, dtype=torch.float32)
    mu_b = alignment.target_mean.to(device=target_raw.device, dtype=torch.float32)
    t_map = None if alignment.linear_map is None else alignment.linear_map.to(device=target_raw.device, dtype=torch.float32)
    with torch.autocast(device_type=target_raw.device.type, enabled=False):
        centered = target_raw.float() - mu_b
        mapped = centered if t_map is None else centered @ t_map
        return mu_a + mapped


def compile_semantic_crosswalk(source_vecs, target_vecs, k=1, temp=0.1, chunk_size=4096):
    # Offline top-k cosine crosswalk from target codebook B to source codebook A (zero target labels).
    k = max(1, min(int(k), source_vecs.size(0)))
    with torch.autocast(device_type=target_vecs.device.type, enabled=False):
        mu = source_vecs.float().mean(dim=0, keepdim=True)
        z_a = F.normalize(source_vecs.float() - mu, dim=-1)
        z_b = F.normalize(target_vecs.float() - mu, dim=-1)
        idx_chunks, sim_chunks = [], []
        for start in range(0, z_b.size(0), chunk_size):
            sim = z_b[start:start + chunk_size] @ z_a.T
            vals, idx = torch.topk(sim, k=k, dim=-1)
            idx_chunks.append(idx)
            sim_chunks.append(vals.clamp(min=0.0, max=1.0))
        top_idx = torch.cat(idx_chunks, dim=0)
        top_sim = torch.cat(sim_chunks, dim=0)
        if k == 1:
            weights = top_sim
        else:
            attn = torch.softmax(top_sim / max(float(temp), 1e-4), dim=-1)
            weights = attn * top_sim
        summary = {
            "mode": "semantic_crosswalk",
            "k": k,
            "temp": float(temp),
            "mean_top1_sim": float(top_sim[:, 0].mean().item()),
            "median_top1_sim": float(top_sim[:, 0].median().item()),
        }
        return top_idx, weights, summary


def apply_semantic_crosswalk(source_logits, top_idx, weights, eps=1e-6, chunk_size=16384):
    # Map calibrated source probabilities p_A(x) -> target logits logit(sum_j w_bj p_{a_j}(x)).
    src_probs = torch.sigmoid(source_logits.float())
    out = []
    for start in range(0, top_idx.size(0), chunk_size):
        idx = top_idx[start:start + chunk_size]
        w = weights[start:start + chunk_size]
        gathered = src_probs[:, idx]
        probs = (gathered * w.unsqueeze(0)).sum(dim=-1).clamp(min=eps, max=1.0 - eps)
        out.append(torch.logit(probs))
    return torch.cat(out, dim=1)


def _load_isotoken_checkpoint(checkpoint_path, model_dir, device, label_chunk_size=4096):
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if state.get("format") != "isotoken_v1":
        raise ValueError("Expected an isotoken_v1 checkpoint")
    train_args = state["args"]
    semantic_meta = state["semantic_meta"]
    source_codes = [Code(**{**row, "ancestors": tuple(row.get("ancestors", []))}) for row in state["source_codes"]]
    model = build_isotoken_model(
        model_dir=model_dir,
        arch=state["arch"],
        label_dim=int(semantic_meta["dim"]),
        source_size=len(source_codes),
        span_size=int(train_args["span_size"]),
        trainable_encoder_layers=int(train_args["trainable_encoder_layers"]),
        head_hidden_dim=int(train_args["head_hidden_dim"]),
        attn_dim=int(train_args["attn_dim"]),
        dropout=0.0,
        label_chunk_size=label_chunk_size,
        prior_tau=float(train_args.get("prior_tau", 0.0)),
        initialize=False,
    )
    model.load_state_dict(state["model"], strict=True)
    model.to(device).requires_grad_(False).eval()
    return model, state


@torch.inference_mode()
def evaluate_isotoken_transfer(args):
    metadata, table = transfer_data(args.data, args.split, args.test_approved)
    device = torch.device(args.device)
    model, state = _load_isotoken_checkpoint(args.checkpoint, args.model_dir, device, label_chunk_size=args.label_chunk_size)
    best = state["source_selection"]
    if best is None or best["step"] != state["step"]:
        raise ValueError("Evaluate the source-selected checkpoint identified in summary.json")
    if best["training_icd_version"] != 9 or best["validation_icd_version"] != 9:
        raise ValueError("Transfer requires source-only model selection")
    threshold = float(best["threshold"])

    origin = json.loads((Path(args.model_dir) / "origin.json").read_text())
    if metadata["tokenizer"] != origin or state["tokenizer_origin"] != origin:
        raise ValueError("Source model and target tokens must use the same fixed encoder revision")

    codes = load_codebook(Path(args.data) / "codebook.json")
    group_file = Path(args.data) / "groups.json"
    compositions = json.loads(group_file.read_text()) if group_file.exists() else []
    source_codes = [Code(**{**row, "ancestors": tuple(row.get("ancestors", []))}) for row in state["source_codes"]]
    train_args = state["args"]
    semantic_meta = state["semantic_meta"]
    semantic_model = args.semantic_model or semantic_meta["model_spec"]
    text_mode = semantic_meta.get("text_mode", "description")
    pooling = semantic_meta.get("pooling", "auto")
    label_max_len = int(train_args.get("label_max_len", 128))
    tokenizer = load_tokenizer(args.model_dir)
    specials = tuple(i for i in (tokenizer.cls_token_id, tokenizer.sep_token_id) if i is not None)

    mode = metadata.get("output_mode", "multilabel")
    operator_policy_summary = None
    if getattr(args, "operator_policy", "none") == "auto_v1":
        policy = resolve_zero_target_policy(
            mode=mode,
            codes=codes,
            source_count=len(source_codes),
            has_groups=bool(compositions),
            source_prior_tau=float(train_args.get("prior_tau", 0.0)),
        )
        args.codebook_align = policy["codebook_align"]
        args.crowding_alpha = policy["crowding_alpha"]
        args.wiener_lambda = policy["wiener_lambda"]
        args.prior_proj_tau = policy["prior_proj_tau"]
        args.prior_shift_scale = policy["prior_shift_scale"]
        args.span_nms_cos = policy["span_nms_cos"]
        args.group_algebra = policy["group_algebra"]
        operator_policy_summary = policy

    compiled_groups = compile_group_index(codes, compositions) if (compositions and args.group_algebra != "none") else None
    is_iso = model.arch in ("isotoken", "isotoken_unlocked")

    sync(device)
    compile_started = time.perf_counter()
    source_vecs = _load_or_encode_semantic(
        args.source_semantic_cache,
        source_codes,
        [],
        semantic_model,
        text_mode,
        pooling,
        args.semantic_batch_size,
        label_max_len,
        device,
    )
    target_vecs_raw = _load_or_encode_semantic(
        args.target_semantic_cache,
        codes,
        compositions,
        semantic_model,
        text_mode,
        pooling,
        args.semantic_batch_size,
        label_max_len,
        device,
    )

    alignment = None
    target_vecs = target_vecs_raw
    if args.codebook_align != "none":
        alignment = fit_codebook_alignment(
            source_vecs,
            target_vecs_raw,
            mode=args.codebook_align,
            reg=args.codebook_align_reg,
        )
        target_vecs = _align_raw_vectors(target_vecs_raw, alignment)

    cast_summary = None
    crosswalk_summary = None
    crosswalk_idx = crosswalk_weights = None
    prototypes = None
    custom_query = custom_weight = custom_bias = None
    plmicd_transfer = getattr(args, "plmicd_transfer", "cast")
    if is_iso:
        prototypes = model.head.prepare_prototypes(target_vecs)
    elif plmicd_transfer == "cms_gem":
        gem_dir = getattr(args, "gem_dir", None)
        if not gem_dir:
            raise ValueError("--plmicd-transfer cms_gem requires --gem-dir")
        crosswalk_idx, crosswalk_weights, crosswalk_summary = compile_cms_gem_crosswalk(
            source_codes,
            codes,
            gem_dir=gem_dir,
            direction=getattr(args, "gem_direction", "both"),
            device=device,
        )
    elif plmicd_transfer == "crosswalk":
        crosswalk_idx, crosswalk_weights, crosswalk_summary = compile_semantic_crosswalk(
            source_vecs,
            target_vecs,
            k=int(getattr(args, "crosswalk_k", 1)),
            temp=float(getattr(args, "crosswalk_temp", 0.1)),
        )
    else:
        # Closed-form CAST extension (Heyden et al., 2026) of PLM-ICD attention queries and classifier weights.
        operator = fit_attention_cast(
            source_vecs,
            model.head.second_linear.weight.detach(),
            model.head.classifier_weight.detach(),
            support_counts=state.get("source_label_counts"),
            ridge=args.cast_ridge if args.cast_ridge > 0.0 else None,
            whiten_eps=args.whiten_eps,
            weighting=args.cast_weighting,
            rank=args.cast_rank,
        )
        custom_query, custom_weight = compile_attention_cast(operator, target_vecs, scale=1.0)
        src_counts = torch.as_tensor(state.get("source_label_counts") or [1] * len(source_codes), device=device)
        supported_mask = src_counts > 0
        norm_matched = False
        if bool(getattr(args, "cast_norm_match", False)):
            src_q_norm = model.head.second_linear.weight.detach().to(device)[supported_mask].float().norm(dim=-1).median()
            src_w_norm = model.head.classifier_weight.detach().to(device)[supported_mask].float().norm(dim=-1).median()
            custom_query = F.normalize(custom_query.float(), dim=-1, eps=1e-6) * src_q_norm
            custom_weight = F.normalize(custom_weight.float(), dim=-1, eps=1e-6) * src_w_norm
            norm_matched = True
        base_bias = float(model.head.classifier_bias.detach().to(device)[supported_mask].median().item())
        custom_bias = torch.full((len(codes),), base_bias, device=device, dtype=torch.float32)
        cast_summary = {**operator.summary, "base_bias_median": base_bias, "norm_matched": norm_matched}

    prior_proj_bias = None
    prior_proj_summary = None
    prior_proj_tau = float(getattr(args, "prior_proj_tau", 0.0))
    prior_proj_k = int(getattr(args, "prior_proj_k", 1))
    prior_proj_temp = float(getattr(args, "prior_proj_temp", 0.05))
    if prior_proj_tau > 0.0 and is_iso:
        src_delta = getattr(model.head, "source_prior_delta", None)
        if src_delta is None and state.get("source_label_counts"):
            counts_t = torch.as_tensor(state["source_label_counts"], dtype=torch.float64)
            n_src = float(max(len(state.get("source_subjects", [])) * 2, int(counts_t.max().item()) * 2, 1))
            pi = ((counts_t + 0.5) / (n_src + 1.0)).clamp(min=1e-7, max=1.0 - 1e-7)
            ell = torch.log(pi / (1.0 - pi))
            bar_pi = pi.mean().clamp(min=1e-7, max=1.0 - 1e-7)
            src_delta = (ell - torch.log(bar_pi / (1.0 - bar_pi))).float()
        if src_delta is not None:
            prior_proj_bias, prior_proj_summary = semantic_prior_projection(
                source_vecs,
                target_vecs,
                src_delta,
                tau=prior_proj_tau,
                top_k=prior_proj_k,
                temp=prior_proj_temp,
                source_counts=state.get("source_label_counts"),
            )
            train_tau = float(train_args.get("prior_tau", 0.0))
            if train_tau > 0.0 and best.get("prior_cal_threshold") is not None:
                alpha = min(max(prior_proj_tau / train_tau, 0.0), 1.0)
                threshold = (1.0 - alpha) * float(best["threshold"]) + alpha * float(best["prior_cal_threshold"])
                prior_proj_summary["interpolated_threshold"] = threshold

    wiener_gain = None
    wiener_summary = None
    wiener_lambda = float(getattr(args, "wiener_lambda", 0.0))
    if wiener_lambda > 0.0 and is_iso:
        wiener_gain, wiener_summary = wiener_spherical_gain(
            source_vecs, target_vecs, wiener_lambda=wiener_lambda,
        )

    span_nms_cos = float(getattr(args, "span_nms_cos", 0.0))
    span_nms_window = int(getattr(args, "span_nms_window", 1))
    use_span_nms = mode == "multilabel" and span_nms_cos > 0.0

    calibration = None
    if args.prior_shift_scale != 0.0 or args.crowding_alpha != 0.0 or args.record_geometry:
        calibration = compile_calibration(
            source_vecs,
            target_vecs,
            mode,
            source_count=len(source_codes),
            target_count=len(codes),
            device=device,
            prior_shift_scale=args.prior_shift_scale,
            crowding_alpha=args.crowding_alpha,
            crowding_k=args.crowding_k,
            whiten_eps=args.whiten_eps,
            residual_ridge=args.residual_ridge,
            record_geometry=args.record_geometry,
            residual_target_vectors=target_vecs_raw,
        )
    sync(device)
    compile_seconds = time.perf_counter() - compile_started

    counts = TransferCounts([code.key for code in codes], mode)
    excluded = set(state["source_subjects"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    skipped, missing_patient, patients = 0, 0, set()

    sync(device)
    infer_started = time.perf_counter()
    save_topk_limit = int(getattr(args, "save_topk_candidates", 0) or 0)
    save_topk_k = int(getattr(args, "save_topk_k", 50) or 50)
    save_topk_no_spans = bool(getattr(args, "save_topk_no_spans", False))
    need_evidence = use_span_nms or (save_topk_limit > 0 and not save_topk_no_spans)
    if use_span_nms:
        if prototypes is not None:
            prototypes_dev = prototypes.detach().to(device)
        else:
            prototypes_dev = F.normalize(target_vecs.float() - source_vecs.float().mean(dim=0, keepdim=True), dim=-1).to(device)
    else:
        prototypes_dev = None
    record_cal = bool(getattr(args, "record_calibration", False)) and mode == "multilabel"
    cal_tracker = CalibrationTracker(n_bins=15) if record_cal else None
    hist_bins = 65536
    hist_lo, hist_hi = -30.0, 30.0
    hist_scale = hist_bins / (hist_hi - hist_lo)
    pos_hist = np.zeros(hist_bins, dtype=np.int64) if record_cal else None
    neg_hist = np.zeros(hist_bins, dtype=np.int64) if record_cal else None
    code_to_idx = {c.key: i for i, c in enumerate(codes)} if record_cal else None

    topk_stream = (output / "topk_candidates.jsonl").open("w", encoding="utf-8") if save_topk_limit > 0 else None
    topk_written = 0
    residual_arr = calibration.residual.cpu().tolist() if calibration is not None else None
    if save_topk_limit > 0:
        from .evidence import _decode_span_window
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
            top_spans = None
            with autocast(device):
                if is_iso:
                    if need_evidence:
                        logits, top_spans = model(
                            rows,
                            tokenizer.pad_token_id,
                            device,
                            prototypes=prototypes,
                            special_ids=specials,
                            label_chunk_size=args.label_chunk_size,
                            return_evidence=True,
                        )
                        logits = logits.float()
                    else:
                        logits = model(
                            rows,
                            tokenizer.pad_token_id,
                            device,
                            prototypes=prototypes,
                            special_ids=specials,
                            label_chunk_size=args.label_chunk_size,
                        ).float()
                    if wiener_gain is not None:
                        base_bias = model.head.bias.detach().float()
                        logits = base_bias + wiener_gain.unsqueeze(0) * (logits - base_bias)
                elif crosswalk_idx is not None:
                    if need_evidence:
                        src_logits, src_spans = model(
                            rows,
                            tokenizer.pad_token_id,
                            device,
                            special_ids=specials,
                            label_chunk_size=args.label_chunk_size,
                            return_evidence=True,
                        )
                        src_logits = src_logits.float()
                        top_spans = src_spans[:, crosswalk_idx[:, 0]]
                    else:
                        src_logits = model(
                            rows,
                            tokenizer.pad_token_id,
                            device,
                            special_ids=specials,
                            label_chunk_size=args.label_chunk_size,
                        ).float()
                    logits = apply_semantic_crosswalk(src_logits, crosswalk_idx, crosswalk_weights)
                else:
                    if need_evidence:
                        logits, top_spans = model(
                            rows,
                            tokenizer.pad_token_id,
                            device,
                            special_ids=specials,
                            custom_query=custom_query,
                            custom_weight=custom_weight,
                            custom_bias=custom_bias,
                            label_chunk_size=args.label_chunk_size,
                            return_evidence=True,
                        )
                        logits = logits.float()
                    else:
                        logits = model(
                            rows,
                            tokenizer.pad_token_id,
                            device,
                            special_ids=specials,
                            custom_query=custom_query,
                            custom_weight=custom_weight,
                            custom_bias=custom_bias,
                            label_chunk_size=args.label_chunk_size,
                        ).float()
            if prior_proj_bias is not None:
                logits = logits + prior_proj_bias.unsqueeze(0)
            if calibration is not None:
                logits = logits + calibration.logit_bias.unsqueeze(0)
            if compiled_groups is not None:
                logits = apply_group_algebra(
                    logits, codes, compositions, mode=args.group_algebra, compiled_groups=compiled_groups,
                )
            top_k = min(max(50, save_topk_k if save_topk_limit > 0 else 50), len(codes))
            top_indices = torch.topk(logits, k=top_k, dim=-1).indices.cpu().tolist()
            probs_dev = logits if mode.startswith("single_label") else torch.sigmoid(logits)
            if record_cal:
                y_batch = np.zeros((len(rows), len(codes)), dtype=np.bool_)
                for r_i, row in enumerate(rows):
                    for key in row.get("labels", []):
                        c_i = code_to_idx.get(key)
                        if c_i is not None:
                            y_batch[r_i, c_i] = True
                logits_np = logits.detach().cpu().numpy()
                probs_np = probs_dev.detach().cpu().numpy()
                cal_tracker.update(probs_np, y_batch)
                bins = np.clip(((logits_np - hist_lo) * hist_scale).astype(np.int64), 0, hist_bins - 1)
                pos_hist += np.bincount(bins[y_batch], minlength=hist_bins)
                neg_hist += np.bincount(bins[~y_batch], minlength=hist_bins)

            probs_cpu = None if use_span_nms else probs_dev.cpu()
            for idx_in_batch, (row, row_top) in enumerate(zip(rows, top_indices, strict=True)):
                if use_span_nms:
                    predicted = apply_span_semantic_nms(
                        probs_dev[idx_in_batch],
                        top_spans[idx_in_batch],
                        prototypes_dev,
                        codes,
                        threshold,
                        cos_threshold=span_nms_cos,
                        span_window=span_nms_window,
                    )
                else:
                    predicted = prediction_keys(probs_cpu[idx_in_batch], codes, mode, threshold)
                tp, fp, fn = counts.update(predicted, row)
                counts.update_ranking([codes[idx].key for idx in row_top[:50]], row)
                missing_patient += row["subject_id"] is None
                patients.add(row["subject_id"])
                entry = {
                    "id": row["id"],
                    "subject_id": row["subject_id"],
                    "predicted": predicted,
                    "truth": row["labels"],
                    "label_groups": row.get("label_groups", []),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                }
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
                if topk_stream is not None and topk_written < save_topk_limit:
                    row_p_src = probs_dev[idx_in_batch] if use_span_nms else probs_cpu[idx_in_batch]
                    cands = []
                    for c_idx in row_top[:save_topk_k]:
                        sp_idx = (
                            int(top_spans[idx_in_batch, c_idx].item())
                            if (top_spans is not None and not save_topk_no_spans)
                            else -1
                        )
                        span_txt, _ = (
                            _decode_span_window(tokenizer, row["chunks"], sp_idx, model.span_size, context_tokens=4)
                            if sp_idx >= 0
                            else ("", "")
                        )
                        cands.append({
                            "key": codes[c_idx].key,
                            "code": codes[c_idx].code,
                            "system": codes[c_idx].system,
                            "description": codes[c_idx].description,
                            "prob": float(row_p_src[c_idx].item()),
                            "logit": float(logits[idx_in_batch, c_idx].item()),
                            "rho_b": float(residual_arr[c_idx]) if residual_arr is not None else 0.0,
                            "span_idx": sp_idx,
                            "span_text": span_txt,
                        })
                    topk_stream.write(json.dumps({
                        "id": row["id"],
                        "subject_id": row["subject_id"],
                        "predicted": predicted,
                        "truth": row["labels"],
                        "threshold": threshold,
                        "candidates": cands,
                    }, ensure_ascii=False) + "\n")
                    topk_written += 1
    if topk_stream is not None:
        topk_stream.close()
    sync(device)
    inference_seconds = time.perf_counter() - infer_started
    if counts.examples == 0:
        raise ValueError("No eligible target examples remained for evaluation")

    metrics = {
        **counts.result(),
        "model": f"{model.arch}_v1",
        "arch": model.arch,
        "kind": model.arch,
        "head_mode": model.arch,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "dataset": metadata["dataset"],
        "split": args.split,
        "output_mode": mode,
        "threshold": threshold,
        "threshold_source": "ICD-9 validation",
        "model_fit_on_target": False,
        "source_patient_rows_excluded": skipped,
        "rows_without_patient_id": missing_patient,
        "patients_with_id": len(patients - {None}),
        "codebook_seconds": compile_seconds,
        "compile_seconds": compile_seconds,
        "inference_seconds": inference_seconds,
        "milliseconds_per_note": 1000.0 * inference_seconds / max(counts.examples, 1),
        "group_algebra": args.group_algebra,
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
    if operator_policy_summary is not None:
        metrics["operator_policy"] = operator_policy_summary
    if record_cal and cal_tracker is not None:
        cal_res = cal_tracker.result()
        cal_res["micro_roc_auc"] = histogram_roc_auc(pos_hist, neg_hist)
        cal_res["micro_auprc"] = histogram_auprc(pos_hist, neg_hist)
        metrics["probability_calibration"] = cal_res
    if alignment is not None:
        metrics["codebook_alignment"] = alignment.summary
    if cast_summary is not None:
        metrics["cast"] = cast_summary
    if crosswalk_summary is not None:
        metrics["crosswalk"] = crosswalk_summary
    if prior_proj_summary is not None:
        metrics["prior_projection"] = prior_proj_summary
    if wiener_summary is not None:
        metrics["wiener_gain"] = wiener_summary
    if use_span_nms:
        metrics["span_nms"] = {"cos_threshold": span_nms_cos, "span_window": span_nms_window}
    if calibration is not None:
        metrics["calibration"] = calibration.summary
        if calibration.residual is not None:
            torch.save(
                {
                    "keys": [code.key for code in codes],
                    "residual": calibration.residual.cpu(),
                    "crowding": calibration.crowding.cpu(),
                    "logit_bias": calibration.logit_bias.cpu(),
                },
                output / "geometry.pt",
            )
            if mode == "multilabel":
                metrics["residual_buckets"] = residual_buckets(
                    counts, [code.key for code in codes], calibration.residual,
                )
    save_json(output / "metrics.json", metrics)
    if calibration is not None and calibration.residual is not None and args.record_geometry:
        routing = analyze_extrapolation_routing(output)
        save_json(output / "extrapolation.json", routing)
        metrics["extrapolation"] = routing
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def _compute_source_suite(
    logits_fp16,
    truth_bool,
    freq10_indices,
    thresholds,
    benchmark_freq_indices=None,
    hist_bins=65536,
    hist_range=(-30.0, 30.0),
):
    n_examples, n_labels = logits_fp16.shape
    counts = Counts(thresholds, n_labels)
    lo, hi = hist_range
    scale = hist_bins / (hi - lo)
    pos_hist_full = np.zeros(hist_bins, dtype=np.int64)
    neg_hist_full = np.zeros(hist_bins, dtype=np.int64)
    pos_hist_f10 = np.zeros(hist_bins, dtype=np.int64)
    neg_hist_f10 = np.zeros(hist_bins, dtype=np.int64)
    pos_hist_bf = np.zeros(hist_bins, dtype=np.int64)
    neg_hist_bf = np.zeros(hist_bins, dtype=np.int64)
    cal_full = CalibrationTracker(n_bins=15)
    cal_f10 = CalibrationTracker(n_bins=15)
    cal_bf = CalibrationTracker(n_bins=15)
    p_at_full = {5: 0.0, 8: 0.0, 15: 0.0}
    p_at_f10 = {5: 0.0, 8: 0.0, 15: 0.0}
    p_at_bf = {5: 0.0, 8: 0.0, 15: 0.0}
    f10_idx = np.asarray(freq10_indices, dtype=np.int64)
    bf_idx = np.asarray(benchmark_freq_indices, dtype=np.int64) if benchmark_freq_indices else np.asarray([], dtype=np.int64)

    batch_size = 512
    for start in range(0, n_examples, batch_size):
        end = min(start + batch_size, n_examples)
        block = logits_fp16[start:end].astype(np.float32)
        y = truth_bool[start:end]
        probs = 1.0 / (1.0 + np.exp(-np.clip(block, -30.0, 30.0)))
        counts.update(probs, y)
        cal_full.update(probs, y)

        bins = np.clip(((block - lo) * scale).astype(np.int64), 0, hist_bins - 1)
        pos_hist_full += np.bincount(bins[y], minlength=hist_bins)
        neg_hist_full += np.bincount(bins[~y], minlength=hist_bins)

        if len(f10_idx) > 0:
            bins_f = bins[:, f10_idx]
            y_f = y[:, f10_idx]
            pos_hist_f10 += np.bincount(bins_f[y_f], minlength=hist_bins)
            neg_hist_f10 += np.bincount(bins_f[~y_f], minlength=hist_bins)
            cal_f10.update(probs[:, f10_idx], y_f)

        if len(bf_idx) > 0:
            bins_b = bins[:, bf_idx]
            y_b = y[:, bf_idx]
            pos_hist_bf += np.bincount(bins_b[y_b], minlength=hist_bins)
            neg_hist_bf += np.bincount(bins_b[~y_b], minlength=hist_bins)
            cal_bf.update(probs[:, bf_idx], y_b)

        top15 = np.argpartition(-block, min(15, n_labels - 1), axis=1)[:, : min(15, n_labels)]
        row_arange = np.arange(end - start)[:, None]
        top15_scores = block[row_arange, top15]
        order15 = np.argsort(-top15_scores, axis=1)
        sorted_top15 = top15[row_arange, order15]
        hits15 = y[row_arange, sorted_top15]
        for k in (5, 8, 15):
            kk = min(k, n_labels)
            p_at_full[k] += float(hits15[:, :kk].sum()) / kk

        if len(f10_idx) > 0:
            block_f = block[:, f10_idx]
            nf = block_f.shape[1]
            top15_f = np.argpartition(-block_f, min(15, nf - 1), axis=1)[:, : min(15, nf)]
            scores_f = block_f[row_arange, top15_f]
            order_f = np.argsort(-scores_f, axis=1)
            sorted_f = top15_f[row_arange, order_f]
            hits_f = y_f[row_arange, sorted_f]
            for k in (5, 8, 15):
                kk = min(k, nf)
                p_at_f10[k] += float(hits_f[:, :kk].sum()) / kk

        if len(bf_idx) > 0:
            block_b = block[:, bf_idx]
            nb = block_b.shape[1]
            top15_b = np.argpartition(-block_b, min(15, nb - 1), axis=1)[:, : min(15, nb)]
            scores_b = block_b[row_arange, top15_b]
            order_b = np.argsort(-scores_b, axis=1)
            sorted_b = top15_b[row_arange, order_b]
            hits_b = y_b[row_arange, sorted_b]
            for k in (5, 8, 15):
                kk = min(k, nb)
                p_at_bf[k] += float(hits_b[:, :kk].sum()) / kk

    best_full = counts.best()
    full_res = counts.result(best_full)
    # Exact tie-aware Macro ROC-AUC and Macro AUPRC across all supported labels.
    aucs = np.full(n_labels, np.nan, dtype=np.float64)
    auprcs = np.full(n_labels, np.nan, dtype=np.float64)
    pos_counts = truth_bool.sum(axis=0)
    for col in np.flatnonzero((pos_counts > 0) & (pos_counts < n_examples)):
        col_scores = logits_fp16[:, col]
        col_y = truth_bool[:, col]
        aucs[col] = binary_roc_auc_exact(col_scores, col_y)
        auprcs[col] = binary_auprc_exact(col_scores, col_y)
    valid_full = ~np.isnan(aucs)
    full_res.update({
        "threshold": float(thresholds[best_full]),
        "micro_auc": histogram_roc_auc(pos_hist_full, neg_hist_full),
        "macro_auc_supported": float(np.nanmean(aucs[valid_full])) if valid_full.any() else None,
        "micro_auprc": histogram_auprc(pos_hist_full, neg_hist_full),
        "macro_auprc_supported": float(np.nanmean(auprcs[valid_full])) if valid_full.any() else None,
        "calibration": cal_full.result(),
        "p_at_5": p_at_full[5] / max(n_examples, 1),
        "p_at_8": p_at_full[8] / max(n_examples, 1),
        "p_at_15": p_at_full[15] / max(n_examples, 1),
    })

    freq10_res = {}
    if len(f10_idx) > 0:
        f10_scores = [counts.result(i, f10_idx)["micro_f1"] for i in range(len(thresholds))]
        best_f10 = max(range(len(thresholds)), key=lambda i: (f10_scores[i], -abs(thresholds[i] - 0.5)))
        freq10_res = counts.result(best_f10, f10_idx)
        aucs_f10 = aucs[f10_idx]
        auprcs_f10 = auprcs[f10_idx]
        valid_f10 = ~np.isnan(aucs_f10)
        freq10_res.update({
            "threshold": float(thresholds[best_f10]),
            "micro_auc": histogram_roc_auc(pos_hist_f10, neg_hist_f10),
            "macro_auc_supported": float(np.nanmean(aucs_f10[valid_f10])) if valid_f10.any() else None,
            "micro_auprc": histogram_auprc(pos_hist_f10, neg_hist_f10),
            "macro_auprc_supported": float(np.nanmean(auprcs_f10[valid_f10])) if valid_f10.any() else None,
            "calibration": cal_f10.result(),
            "p_at_5": p_at_f10[5] / max(n_examples, 1),
            "p_at_8": p_at_f10[8] / max(n_examples, 1),
            "p_at_15": p_at_f10[15] / max(n_examples, 1),
        })

    out = {"full": full_res, "freq10": freq10_res}
    if len(bf_idx) > 0:
        bf_scores = [counts.result(i, bf_idx)["micro_f1"] for i in range(len(thresholds))]
        best_bf = max(range(len(thresholds)), key=lambda i: (bf_scores[i], -abs(thresholds[i] - 0.5)))
        bf_res = counts.result(best_bf, bf_idx)
        aucs_bf = aucs[bf_idx]
        auprcs_bf = auprcs[bf_idx]
        valid_bf = ~np.isnan(aucs_bf)
        bf_res.update({
            "threshold": float(thresholds[best_bf]),
            "micro_auc": histogram_roc_auc(pos_hist_bf, neg_hist_bf),
            "macro_auc_supported": float(np.nanmean(aucs_bf[valid_bf])) if valid_bf.any() else None,
            "micro_auprc": histogram_auprc(pos_hist_bf, neg_hist_bf),
            "macro_auprc_supported": float(np.nanmean(auprcs_bf[valid_bf])) if valid_bf.any() else None,
            "calibration": cal_bf.result(),
            "p_at_5": p_at_bf[5] / max(n_examples, 1),
            "p_at_8": p_at_bf[8] / max(n_examples, 1),
            "p_at_15": p_at_bf[15] / max(n_examples, 1),
        })
        out["benchmark_frequent"] = bf_res
    return out


def _write_source_counts(path: Path, row_ids: list[tuple[str, str]], logits_fp16: np.ndarray,
                         truth_bool: np.ndarray, col_indices: np.ndarray | None, threshold: float) -> None:
    logit_thr = float(np.log(max(threshold, 1e-6) / max(1.0 - threshold, 1e-6)))
    with path.open("w", encoding="utf-8") as stream:
        for i, (hadm_id, subject_id) in enumerate(row_ids):
            row_logits = logits_fp16[i] if col_indices is None else logits_fp16[i, col_indices]
            row_truth = truth_bool[i] if col_indices is None else truth_bool[i, col_indices]
            pred = row_logits >= logit_thr
            tp = int(np.logical_and(pred, row_truth).sum())
            fp = int(np.logical_and(pred, ~row_truth).sum())
            fn = int(np.logical_and(~pred, row_truth).sum())
            stream.write(json.dumps({"id": hadm_id, "subject_id": subject_id, "tp": tp, "fp": fp, "fn": fn}) + "\n")


def evaluate_isotoken_source(args):
    split = getattr(args, "split", "validation")
    test_approved = bool(getattr(args, "test_approved", False))
    if (split == "test" or Path(args.validation_data).name == "test") and not test_approved:
        raise PermissionError("Source test split evaluation requires explicit user approval (--test-approved)")
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    tokenizer = load_tokenizer(args.model_dir)
    specials = [i for i in (tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id) if i is not None]
    val_dir = Path(args.validation_data)
    if split == "test" and not (val_dir / "metadata.json").exists() and (val_dir / "test" / "metadata.json").exists():
        val_dir = val_dir / "test"
    validation = Notes(val_dir, split, test_approved=test_approved)
    codes = load_codebook(val_dir / "codebook.json")
    train_meta = json.loads((Path(args.train_data) / "meta.json").read_text()) if (Path(args.train_data) / "meta.json").exists() else {}
    label_counts_map = train_meta.get("label_counts", {})
    if not label_counts_map:
        train_ds = Notes(args.train_data, "train")
        label_counts_map = train_ds.meta.get("label_counts", {})
    if not label_counts_map:
        index_map = {code.key: idx for idx, code in enumerate(codes)}
        counts_list = [0] * len(codes)
        for i in range(len(train_ds)):
            for key in train_ds[i]["labels"]:
                if key in index_map:
                    counts_list[index_map[key]] += 1
        label_counts_map = {codes[i].key: counts_list[i] for i in range(len(codes))}
    freq10_indices = [i for i, code in enumerate(codes) if int(label_counts_map.get(code.key, 0)) >= 10]
    bench_freq_indices = None
    bench_groups_path = getattr(args, "benchmark_groups", None)
    if bench_groups_path:
        bg_file = Path(bench_groups_path)
        if not bg_file.exists():
            raise FileNotFoundError(f"benchmark_groups file not found: {bg_file}")
        bg = json.loads(bg_file.read_text())
        bf_set = set(bg.get("benchmark_frequent", []))
        bench_freq_indices = [i for i, code in enumerate(codes) if code.key in bf_set]

    model, state = _load_isotoken_checkpoint(args.checkpoint, args.model_dir, device)
    is_iso = model.arch in ("isotoken", "isotoken_unlocked")
    saved_keys = [row["key"] for row in state.get("source_codes", [])]
    if saved_keys and saved_keys != [c.key for c in codes]:
        raise ValueError("Checkpoint source codebook does not match validation codebook")
    source_vecs = _load_or_encode_semantic(
        args.source_semantic_cache, codes, [],
        state.get("semantic_model") or args.semantic_model,
        args.semantic_text_mode, args.semantic_pooling, 256, 128, device,
    )
    if is_iso:
        prototypes = model.head.prepare_prototypes(source_vecs)
    else:
        prototypes = None

    n_examples = min(len(validation), args.limit) if args.limit else len(validation)
    n_labels = len(codes)
    code_to_idx = {code.key: idx for idx, code in enumerate(codes)}
    truth_bool = np.zeros((n_examples, n_labels), dtype=np.bool_)
    row_ids: list[tuple[str, str]] = []
    pure_logits_fp16 = np.empty((n_examples, n_labels), dtype=np.float16)
    cal_logits_fp16 = np.empty((n_examples, n_labels), dtype=np.float16) if is_iso else None

    sync(device)
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, n_examples, args.batch_size):
            end = min(start + args.batch_size, n_examples)
            rows = [validation[i] for i in range(start, end)]
            for r_off, row in enumerate(rows):
                row_ids.append((str(row["id"]), str(row["subject_id"])))
                for key in row["labels"]:
                    idx = code_to_idx.get(key)
                    if idx is not None:
                        truth_bool[start + r_off, idx] = True
            with autocast(device):
                if is_iso:
                    pure, cal = model(
                        rows, tokenizer.pad_token_id, device,
                        prototypes=prototypes, special_ids=specials,
                        label_chunk_size=args.label_chunk_size, return_calibrated=True,
                    )
                    pure_logits_fp16[start:end] = pure.float().cpu().numpy().astype(np.float16)
                    cal_logits_fp16[start:end] = cal.float().cpu().numpy().astype(np.float16)
                else:
                    pure = model(
                        rows, tokenizer.pad_token_id, device,
                        special_ids=specials, label_chunk_size=args.label_chunk_size,
                    )
                    pure_logits_fp16[start:end] = pure.float().cpu().numpy().astype(np.float16)
    sync(device)
    infer_seconds = time.perf_counter() - started

    if split == "test":
        # Zero test-set threshold tuning: strictly use frozen validation-selected thresholds.
        sel = state.get("source_selection") or {}
        pure_thr = float(sel.get("threshold", state.get("threshold", 0.5)))
        cal_thr = float(sel.get("prior_cal_threshold") or pure_thr)
        pure_thresholds = np.array([pure_thr], dtype=np.float64)
        cal_thresholds = np.array([cal_thr], dtype=np.float64)
    else:
        pure_thresholds = np.linspace(0.05, 0.95, 19)
        cal_thresholds = pure_thresholds
    report = {
        "arch": model.arch,
        "split": split,
        "threshold_locked": split == "test",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "examples": n_examples,
        "labels": n_labels,
        "freq10_labels": len(freq10_indices),
        "benchmark_frequent_labels": len(bench_freq_indices) if bench_freq_indices else 0,
        "inference_seconds": infer_seconds,
        "milliseconds_per_note": 1000.0 * infer_seconds / max(n_examples, 1),
        "pure": _compute_source_suite(
            pure_logits_fp16, truth_bool, freq10_indices, pure_thresholds,
            benchmark_freq_indices=bench_freq_indices,
        ),
    }
    if cal_logits_fp16 is not None:
        report["prior_cal"] = _compute_source_suite(
            cal_logits_fp16, truth_bool, freq10_indices, cal_thresholds,
            benchmark_freq_indices=bench_freq_indices,
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "source_metrics.json", report)
    if getattr(args, "save_counts", False):
        f10_arr = np.asarray(freq10_indices, dtype=np.int64)
        main_logits = cal_logits_fp16 if cal_logits_fp16 is not None else pure_logits_fp16
        main_key = "prior_cal" if cal_logits_fp16 is not None else "pure"
        _write_source_counts(output / "counts_freq10.jsonl", row_ids, main_logits, truth_bool, f10_arr, report[main_key]["freq10"]["threshold"])
        _write_source_counts(output / "counts_full.jsonl", row_ids, main_logits, truth_bool, None, report[main_key]["full"]["threshold"])
        if bench_freq_indices:
            bf_arr = np.asarray(bench_freq_indices, dtype=np.int64)
            _write_source_counts(output / "counts_benchmark_frequent.jsonl", row_ids, main_logits, truth_bool, bf_arr, report[main_key]["benchmark_frequent"]["threshold"])
    print(json.dumps(report, indent=2), flush=True)
    return report

