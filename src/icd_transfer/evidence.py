"""Extract zero-training 4-token clinical evidence spans from IsoToken checkpoints."""

import json
from pathlib import Path

import torch

from .calibration import compile_calibration, fit_codebook_alignment, primary_semantic_vectors
from .codebook import load_codebook
from .data import load_tokenizer
from .eval_isotoken import _align_raw_vectors, _load_isotoken_checkpoint
from .semantic_encoder import load_semantic_cache
from .evaluate_transfer import transfer_data


def _special_token_ids(tokenizer):
    return tuple(i for i in (tokenizer.cls_token_id, tokenizer.sep_token_id) if i is not None)


def _decode_span_window(tokenizer, input_ids, span_idx, span_size, context_tokens=8):
    """Decode the exact top-attended span matching `pool_note_spans` indexing."""
    specials = set(_special_token_ids(tokenizer))
    pad_id = tokenizer.pad_token_id
    span_tokens = []
    span_chunk_pos = []
    for c_idx, chunk in enumerate(input_ids):
        toks = [int(t) for t in chunk]
        valid = [t != pad_id for t in toks]
        content = [v and (t not in specials) for t, v in zip(toks, valid)]
        if not any(content):
            content = valid
        rem = len(toks) % int(span_size)
        if rem:
            pad_len = int(span_size) - rem
            toks.extend([pad_id] * pad_len)
            content.extend([False] * pad_len)
        for s in range(0, len(toks), int(span_size)):
            win_t = toks[s:s + int(span_size)]
            win_m = content[s:s + int(span_size)]
            if any(win_m):
                span_tokens.append([t for t, m in zip(win_t, win_m) if m])
                span_chunk_pos.append((c_idx, s, s + int(span_size)))
    if not span_tokens:
        return "", ""
    idx = min(max(0, int(span_idx)), len(span_tokens) - 1)
    c_idx, s_start, s_end = span_chunk_pos[idx]
    chunk_toks = [int(t) for t in input_ids[c_idx]]
    ctx_start = max(0, s_start - int(context_tokens))
    ctx_end = min(len(chunk_toks), s_end + int(context_tokens))
    span_text = tokenizer.decode(span_tokens[idx], skip_special_tokens=True).strip()
    ctx_text = tokenizer.decode(chunk_toks[ctx_start:ctx_end], skip_special_tokens=True).strip()
    return span_text, ctx_text


def inspect_evidence(args):
    """Extract top predicted codes and their highest-attended 4-token clinical spans."""
    device = torch.device(args.device)
    model, ckpt = _load_isotoken_checkpoint(args.checkpoint, args.model_dir, device)
    if model.arch != "isotoken":
        raise ValueError("inspect-evidence requires an isotoken checkpoint")
    tokenizer = load_tokenizer(args.model_dir)
    specials = _special_token_ids(tokenizer)

    metadata, table = transfer_data(args.data, args.split, False)
    target_codes = load_codebook(Path(args.data) / "codebook.json")
    target_vecs_raw, _ = load_semantic_cache(args.target_semantic_cache, target_codes, device=device)
    source_codes = load_codebook(Path(args.source_codebook))
    source_vecs, _ = load_semantic_cache(args.source_semantic_cache, source_codes, device=device)

    target_vecs = target_vecs_raw
    if args.codebook_align != "none":
        alignment = fit_codebook_alignment(source_vecs, target_vecs_raw, mode=args.codebook_align, reg=args.codebook_align_reg)
        target_vecs = _align_raw_vectors(target_vecs_raw, alignment)
    cal = compile_calibration(
        source_vecs,
        target_vecs,
        mode=metadata.get("output_mode", "multilabel"),
        prior_shift_scale=args.prior_shift_scale,
        crowding_alpha=args.crowding_alpha,
        crowding_k=args.crowding_k,
    )
    prototypes = model.head.prepare_prototypes(target_vecs).to(device)
    bias_shift = cal.logit_bias.to(device)

    threshold = float(args.threshold) if args.threshold is not None else float(ckpt["source_selection"]["threshold"])
    n_notes = min(table.num_rows, int(args.limit))
    rows = table.slice(0, n_notes).to_pylist()
    records = []
    with torch.no_grad():
        for i, row in enumerate(rows):
            logits, top_spans = model(
                [row],
                tokenizer.pad_token_id,
                device,
                prototypes=prototypes,
                special_ids=specials,
                return_evidence=True,
            )
            probs = (logits[0].float() + bias_shift).sigmoid().cpu()
            spans_i = top_spans[0].cpu()
            truth = set(row.get("labels", []))
            if not truth and row.get("label_groups"):
                for g in row["label_groups"]:
                    truth.update(g)
            picked = (probs >= threshold).nonzero(as_tuple=False).flatten().tolist()
            if not picked:
                picked = torch.topk(probs, k=min(args.top_k, len(probs))).indices.tolist()
            picked = sorted(picked, key=lambda idx: float(probs[idx]), reverse=True)[: args.top_k]
            items = []
            for idx in picked:
                code = target_codes[idx]
                span_text, ctx_text = _decode_span_window(
                    tokenizer, row["chunks"], int(spans_i[idx]), model.span_size, args.context_tokens
                )
                items.append({
                    "code": code.key,
                    "description": code.description,
                    "prob": round(float(probs[idx]), 4),
                    "is_truth": code.key in truth,
                    "span_index": int(spans_i[idx]),
                    "span_text": span_text,
                    "context_text": ctx_text,
                })
            records.append({
                "note_id": row.get("note_id", str(i)),
                "num_truth": len(truth),
                "predictions": items,
            })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"notes": len(records), "output": str(out_path)}, indent=2))
    return records


_NEGATION_RE = None
_FAMILY_RE = None


def _negation_flags(context_text):
    import re

    global _NEGATION_RE, _FAMILY_RE
    if _NEGATION_RE is None:
        _NEGATION_RE = re.compile(
            r"\b(no|not|denies|denied|without|negative\s+for|ruled\s+out|rule\s+out|r/o|free\s+of|absent|unlikely|resolved|history\s+of\s+no)\b",
            re.IGNORECASE,
        )
        _FAMILY_RE = re.compile(
            r"\b(family\s+history|fhx|mother|father|sister|brother|maternal|paternal)\b",
            re.IGNORECASE,
        )
    return bool(_NEGATION_RE.search(context_text)), bool(_FAMILY_RE.search(context_text))


@torch.inference_mode()
def audit_span_faithfulness(args):
    """Quantitative 4-token span-deletion faithfulness & clinical negation audit on locked transfer notes."""
    import random

    import numpy as np

    from .calibration import (
        apply_span_semantic_nms,
        semantic_prior_projection,
        wiener_spherical_gain,
    )

    device = torch.device(args.device)
    model, ckpt = _load_isotoken_checkpoint(args.checkpoint, args.model_dir, device)
    if model.arch not in ("isotoken", "isotoken_unlocked"):
        raise ValueError("audit-span-faithfulness requires an isotoken checkpoint")
    tokenizer = load_tokenizer(args.model_dir)
    specials = _special_token_ids(tokenizer)

    metadata, table = transfer_data(args.data, args.split, bool(getattr(args, "test_approved", False)))
    target_codes = load_codebook(Path(args.data) / "codebook.json")
    source_codes = load_codebook(Path(args.source_codebook))
    target_vecs_raw, _ = load_semantic_cache(args.target_semantic_cache, target_codes, device=device)
    source_vecs, _ = load_semantic_cache(args.source_semantic_cache, source_codes, device=device)

    target_vecs = target_vecs_raw
    if args.codebook_align != "none":
        alignment = fit_codebook_alignment(source_vecs, target_vecs_raw, mode=args.codebook_align, reg=args.codebook_align_reg)
        target_vecs = _align_raw_vectors(target_vecs_raw, alignment)

    cal = compile_calibration(
        source_vecs, target_vecs, mode=metadata.get("output_mode", "multilabel"),
        prior_shift_scale=args.prior_shift_scale, crowding_alpha=args.crowding_alpha, crowding_k=args.crowding_k,
    )
    prototypes = model.head.prepare_prototypes(target_vecs).to(device)
    total_bias = cal.logit_bias.to(device)
    best = ckpt["source_selection"]
    threshold = float(best["threshold"])
    train_args = ckpt["args"]
    prior_proj_tau = float(getattr(args, "prior_proj_tau", 0.0))
    if prior_proj_tau > 0.0 and getattr(model.head, "source_prior_delta", None) is not None:
        pp_bias, _ = semantic_prior_projection(
            source_vecs, target_vecs, model.head.source_prior_delta, tau=prior_proj_tau,
        )
        total_bias = total_bias + pp_bias.to(device)
        train_tau = float(train_args.get("prior_tau", 0.0))
        if train_tau > 0.0 and best.get("prior_cal_threshold") is not None:
            alpha = min(max(prior_proj_tau / train_tau, 0.0), 1.0)
            threshold = (1.0 - alpha) * float(best["threshold"]) + alpha * float(best["prior_cal_threshold"])
    wiener_gain = None
    if float(getattr(args, "wiener_lambda", 0.0)) > 0.0:
        wiener_gain, _ = wiener_spherical_gain(source_vecs, target_vecs, wiener_lambda=float(args.wiener_lambda))
        wiener_gain = wiener_gain.to(device)
    base_bias = model.head.bias.detach().float().to(device)

    code_to_idx = {c.key: i for i, c in enumerate(target_codes)}
    excluded = set(ckpt.get("source_subjects", []))
    rng = random.Random(42)
    rows = []
    for batch in table.to_batches(max_chunksize=64):
        for row in batch.to_pylist():
            if metadata.get("patient_id_namespace") == "mimiciv" and row["subject_id"] in excluded:
                continue
            rows.append(row)
            if len(rows) >= int(args.limit):
                break
        if len(rows) >= int(args.limit):
            break

    top_drops, rand_drops = [], []
    top_flips, rand_flips = 0, 0
    tp_neg, tp_fam, tp_tot = 0, 0, 0
    fp_neg, fp_fam, fp_tot = 0, 0, 0
    span_counts = []

    for row in rows:
        spans, span_mask = model.encode_spans([row], tokenizer.pad_token_id, device, special_ids=specials)
        n_valid = int(span_mask[0].sum().item())
        if n_valid <= 2:
            continue
        span_counts.append(n_valid)
        key, val = model.head.project_spans(spans, span_mask)
        attn_scale = model.head.attn_log_scale.exp().clamp(min=1.0, max=100.0)
        scale = model.head.log_scale.exp().clamp(min=1.0, max=100.0)
        raw_logits, top_spans = model.head.score_spans(spans, span_mask, prototypes, return_evidence=True)
        logits = raw_logits[0].float()
        if wiener_gain is not None:
            logits = base_bias + wiener_gain * (logits - base_bias)
        logits = logits + total_bias
        probs = torch.sigmoid(logits)
        if float(getattr(args, "span_nms_cos", 0.0)) > 0.0:
            pred_keys = apply_span_semantic_nms(
                probs, top_spans[0], prototypes, target_codes, threshold,
                cos_threshold=float(args.span_nms_cos), span_window=1,
            )
        else:
            pred_keys = [target_codes[i].key for i in (probs >= threshold).nonzero(as_tuple=False).flatten().tolist()]
        if not pred_keys:
            continue
        truth = set(row.get("labels", []))
        pred_idx = torch.tensor([code_to_idx[k] for k in pred_keys], dtype=torch.long, device=device)
        p_sub = prototypes[pred_idx]  # [M, d_z]
        base_probs = probs[pred_idx].cpu().numpy()
        chosen_spans = top_spans[0, pred_idx].cpu().tolist()

        # Precompute [S, M] attention logits & value dots strictly for the M predicted codes.
        k0 = key[0, :n_valid]  # [S, d_z]
        v0 = val[0, :n_valid]  # [S, d_z]
        a_logits = (k0 @ p_sub.T) * attn_scale  # [S, M]
        v_dots = v0 @ p_sub.T                   # [S, M]
        w_sub = wiener_gain[pred_idx] if wiener_gain is not None else None
        b_sub = total_bias[pred_idx]

        # Exact span deletion for top-attended span s*(b) vs matched random span s_rand != s*(b).
        a_top = a_logits.clone()
        a_rand = a_logits.clone()
        for m_i, sp in enumerate(chosen_spans):
            a_top[sp, m_i] = -torch.inf
            r_sp = rng.randrange(n_valid - 1)
            if r_sp >= sp:
                r_sp += 1
            a_rand[r_sp, m_i] = -torch.inf

        def _masked_probs(am):
            ev = (am.softmax(dim=0) * v_dots).sum(dim=0)
            lg = scale * ev + base_bias
            if w_sub is not None:
                lg = base_bias + w_sub * (lg - base_bias)
            return torch.sigmoid(lg + b_sub).cpu().numpy()

        p_after_top = _masked_probs(a_top)
        p_after_rand = _masked_probs(a_rand)
        for m_i, (k_str, sp) in enumerate(zip(pred_keys, chosen_spans)):
            p0 = float(base_probs[m_i])
            pt = float(p_after_top[m_i])
            pr = float(p_after_rand[m_i])
            top_drops.append(p0 - pt)
            rand_drops.append(p0 - pr)
            if pt < threshold:
                top_flips += 1
            if pr < threshold:
                rand_flips += 1
            _, ctx = _decode_span_window(tokenizer, row["chunks"], int(sp), model.span_size, context_tokens=12)
            has_neg, has_fam = _negation_flags(ctx)
            if k_str in truth:
                tp_tot += 1
                tp_neg += int(has_neg)
                tp_fam += int(has_fam)
            else:
                fp_tot += 1
                fp_neg += int(has_neg)
                fp_fam += int(has_fam)

    total_preds = len(top_drops)
    report = {
        "notes_audited": len(span_counts),
        "total_predictions_audited": total_preds,
        "mean_spans_per_note": float(np.mean(span_counts)) if span_counts else 0.0,
        "threshold": threshold,
        "faithfulness": {
            "top_span_mean_prob_drop": float(np.mean(top_drops)) if top_drops else 0.0,
            "top_span_median_prob_drop": float(np.median(top_drops)) if top_drops else 0.0,
            "top_span_flip_rate": top_flips / max(total_preds, 1),
            "random_span_mean_prob_drop": float(np.mean(rand_drops)) if rand_drops else 0.0,
            "random_span_median_prob_drop": float(np.median(rand_drops)) if rand_drops else 0.0,
            "random_span_flip_rate": rand_flips / max(total_preds, 1),
            "drop_ratio_top_vs_random": (
                float(np.mean(top_drops)) / max(float(np.mean(rand_drops)), 1e-8) if top_drops else 0.0
            ),
        },
        "context_negation_audit": {
            "tp_total": tp_tot,
            "tp_negation_rate": tp_neg / max(tp_tot, 1),
            "tp_family_history_rate": tp_fam / max(tp_tot, 1),
            "fp_total": fp_tot,
            "fp_negation_rate": fp_neg / max(fp_tot, 1),
            "fp_family_history_rate": fp_fam / max(fp_tot, 1),
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report
