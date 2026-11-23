import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from .codebook import load_codebook
from .data import Notes, load_tokenizer, targets_for
from .evaluate import check_metadata
from .inference import autocast, sync
from .isotoken import build_isotoken_model
from .metrics import Counts
from .semantic_encoder import encode_codebook_semantic, load_semantic_cache, save_semantic_cache
from .train import seed_everything


def _dist_info():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return world_size, rank, local_rank


def _init_distributed(device_arg):
    world_size, rank, local_rank = _dist_info()
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        return world_size, rank, local_rank, device
    return 1, 0, 0, torch.device(device_arg)


def _cleanup_distributed(world_size):
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def _source_datasets(args):
    train = Notes(args.train_data, "train")
    validation = Notes(args.validation_data, "validation")
    if train.meta["icd_version"] != 9 or validation.meta["icd_version"] != 9:
        raise ValueError("Only ICD-9 source examples may enter training or model selection")
    origin = json.loads((Path(args.model_dir) / "origin.json").read_text())
    check_metadata(train, origin)
    check_metadata(validation, origin)
    codes = load_codebook(Path(args.train_data) / "codebook.json")
    if codes != load_codebook(Path(args.validation_data) / "codebook.json"):
        raise ValueError("Source training and validation codebooks must have identical order")
    train_subjects = set(json.loads((Path(args.train_data) / "subjects.json").read_text()))
    val_subjects = set(json.loads((Path(args.validation_data) / "subjects.json").read_text()))
    if train_subjects & val_subjects:
        raise ValueError("Source training and validation contain overlapping patients")
    counts = train.meta.get("label_counts", {})
    if counts:
        label_counts = [int(counts.get(code.key, 0)) for code in codes]
    else:
        index = {code.key: idx for idx, code in enumerate(codes)}
        label_counts = [0] * len(codes)
        for i in range(len(train)):
            for key in train[i]["labels"]:
                if key in index:
                    label_counts[index[key]] += 1
    freq10_indices = np.asarray([i for i, c in enumerate(label_counts) if c >= 10], dtype=np.int64)
    return train, validation, codes, label_counts, freq10_indices, origin, sorted(train_subjects | val_subjects)


def _load_or_build_source_semantic(args, codes, device, rank, world_size):
    cache_path = Path(args.semantic_cache) if args.semantic_cache else None
    if cache_path is not None and cache_path.exists():
        vectors, meta = load_semantic_cache(cache_path, codes, device=device)
        return vectors, meta
    if rank == 0:
        vectors = encode_codebook_semantic(
            codes,
            [],
            model_spec=args.semantic_model,
            device=device,
            text_mode=args.semantic_text_mode,
            batch_size=args.semantic_batch_size,
            max_length=args.label_max_len,
            pooling=args.semantic_pooling,
        )
        if cache_path is not None:
            save_semantic_cache(
                cache_path,
                codes,
                vectors,
                model_spec=args.semantic_model,
                text_mode=args.semantic_text_mode,
                pooling=args.semantic_pooling,
            )
    if world_size > 1:
        dist.barrier()
        if cache_path is not None:
            vectors, meta = load_semantic_cache(cache_path, codes, device=device)
            return vectors, meta
        raise ValueError("Multi-GPU training requires --semantic-cache path when compiling semantic labels")
    meta = {
        "model_spec": str(args.semantic_model),
        "text_mode": args.semantic_text_mode,
        "pooling": args.semantic_pooling,
        "dim": int(vectors.shape[-1]),
    }
    return vectors, meta


def _special_ids(tokenizer):
    return tuple(i for i in (tokenizer.cls_token_id, tokenizer.sep_token_id) if i is not None)


def _optimizer_groups(model, encoder_lr, head_lr, weight_decay):
    enc_decay, enc_nodecay, head_decay, head_nodecay = [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_no_decay = param.ndim < 2 or "bias" in name or "norm" in name.lower() or "log_scale" in name
        if name.startswith("encoder."):
            (enc_nodecay if is_no_decay else enc_decay).append(param)
        else:
            (head_nodecay if is_no_decay else head_decay).append(param)
    groups = []
    if enc_decay:
        groups.append({"params": enc_decay, "lr": encoder_lr, "weight_decay": weight_decay})
    if enc_nodecay:
        groups.append({"params": enc_nodecay, "lr": encoder_lr, "weight_decay": 0.0})
    if head_decay:
        groups.append({"params": head_decay, "lr": head_lr, "weight_decay": weight_decay})
    if head_nodecay:
        groups.append({"params": head_nodecay, "lr": head_lr, "weight_decay": 0.0})
    return groups


def _lr_factor(step, warmup_steps, total_steps, min_ratio=0.1):
    if total_steps <= 0:
        return 1.0
    if step < warmup_steps:
        return float(step + 1) / float(max(warmup_steps, 1))
    progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
    progress = min(max(progress, 0.0), 1.0)
    return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))


def _reduce_counts(counts, world_size, device):
    if world_size <= 1:
        return counts
    for attr in ("tp", "fp", "fn"):
        tensor = torch.as_tensor(getattr(counts, attr), device=device, dtype=torch.int64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        setattr(counts, attr, tensor.cpu().numpy())
    return counts


@torch.inference_mode()
def validate_isotoken(
    model,
    validation,
    codes,
    freq10_indices,
    prototypes,
    tokenizer,
    args,
    device,
    rank,
    world_size,
):
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.eval()
    thresholds = np.concatenate(([0.02], np.arange(0.05, 0.96, 0.05), [0.98]))
    pure_counts = Counts(thresholds, len(codes))
    has_prior_cal = raw_model.arch in ("isotoken", "isotoken_unlocked") and (
        raw_model.head.source_bias is not None or raw_model.head.prior_tau > 0.0
    )
    cal_counts = Counts(thresholds, len(codes)) if has_prior_cal else None
    specials = _special_ids(tokenizer)
    limit = min(len(validation), args.validation_limit) if args.validation_limit else len(validation)
    indices = list(range(rank, limit, world_size))
    sync(device)
    started = time.perf_counter()
    for start in range(0, len(indices), args.eval_batch_size):
        batch_idx = indices[start:start + args.eval_batch_size]
        rows = [validation[i] for i in batch_idx]
        target = targets_for(rows, codes, device).cpu().numpy()
        with autocast(device):
            logits = raw_model(
                rows,
                tokenizer.pad_token_id,
                device,
                prototypes=prototypes,
                special_ids=specials,
                label_chunk_size=args.label_chunk_size,
            )
        probs = torch.sigmoid(logits.float()).cpu().numpy()
        pure_counts.update(probs, target)
        if has_prior_cal:
            cal_probs = torch.sigmoid(raw_model.head.detached_source_logits(logits.float())).cpu().numpy()
            cal_counts.update(cal_probs, target)
    _reduce_counts(pure_counts, world_size, device)
    if has_prior_cal:
        _reduce_counts(cal_counts, world_size, device)
    sync(device)
    elapsed = time.perf_counter() - started
    best = pure_counts.best()
    threshold = float(pure_counts.thresholds[best])
    result = {
        "threshold": threshold,
        **pure_counts.result(best),
        "validation_seconds": elapsed,
        "validation_examples": limit,
    }
    if len(freq10_indices):
        f10 = pure_counts.result(best, indices=freq10_indices)
        result["freq10_micro_f1"] = f10["micro_f1"]
        result["freq10_macro_f1"] = f10["macro_f1"]
        result["freq10_precision"] = f10["precision"]
        result["freq10_recall"] = f10["recall"]
        result["freq10_labels"] = f10["labels"]
    if has_prior_cal:
        cal_best = cal_counts.best()
        cal_res = cal_counts.result(cal_best)
        result["prior_cal_threshold"] = float(cal_counts.thresholds[cal_best])
        result["prior_cal_micro_f1"] = cal_res["micro_f1"]
        result["prior_cal_macro_f1"] = cal_res["macro_f1"]
        result["prior_cal_macro_f1_supported"] = cal_res["macro_f1_supported"]
        if len(freq10_indices):
            cal_f10 = cal_counts.result(cal_best, indices=freq10_indices)
            result["prior_cal_freq10_micro_f1"] = cal_f10["micro_f1"]
            result["prior_cal_freq10_macro_f1"] = cal_f10["macro_f1"]
    return result


def _save_isotoken_checkpoint(
    path,
    raw_model,
    optimizer,
    scheduler,
    args,
    epoch,
    step,
    threshold,
    codes,
    label_counts,
    origin,
    subjects,
    semantic_meta,
    selection,
):
    payload = {
        "format": "isotoken_v1",
        "arch": raw_model.arch,
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": vars(args),
        "epoch": epoch,
        "step": step,
        "threshold": threshold,
        "source_codes": [asdict(code) for code in codes],
        "source_label_counts": label_counts,
        "tokenizer_origin": origin,
        "source_subjects": subjects,
        "semantic_meta": semantic_meta,
        "source_selection": selection,
    }
    torch.save(payload, path)


def train_isotoken(args):
    world_size, rank, local_rank, device = _init_distributed(args.device)
    is_main = rank == 0
    try:
        seed_everything(args.seed + rank)
        train_set, validation, codes, label_counts, freq10_indices, origin, subjects = _source_datasets(args)
        tokenizer = load_tokenizer(args.model_dir)
        specials = _special_ids(tokenizer)
        semantic_vecs, semantic_meta = _load_or_build_source_semantic(args, codes, device, rank, world_size)
        raw_model = build_isotoken_model(
            model_dir=args.model_dir,
            arch=args.arch,
            label_dim=int(semantic_vecs.shape[-1]),
            source_size=len(codes),
            span_size=args.span_size,
            trainable_encoder_layers=args.trainable_encoder_layers,
            head_hidden_dim=args.head_hidden_dim,
            attn_dim=args.attn_dim,
            dropout=args.dropout,
            label_chunk_size=args.label_chunk_size,
            prior_tau=getattr(args, "prior_tau", 0.0),
            initialize=not bool(args.resume),
        ).to(device)
        if args.gradient_checkpointing and hasattr(raw_model.encoder, "gradient_checkpointing_enable"):
            raw_model.encoder.gradient_checkpointing_enable()
        if raw_model.arch in ("isotoken", "isotoken_unlocked"):
            raw_model.head.set_label_mean(semantic_vecs.mean(0))
            if raw_model.head.prior_tau > 0.0:
                raw_model.head.set_source_prior(label_counts, len(train_set))
            prototypes = semantic_vecs.detach() if raw_model.arch == "isotoken_unlocked" else raw_model.head.prepare_prototypes(semantic_vecs).detach()
        else:
            prototypes = None

        groups = _optimizer_groups(raw_model, args.encoder_lr, args.head_lr, args.weight_decay)
        optimizer = torch.optim.AdamW(groups)
        per_rank_examples = math.ceil(len(train_set) / world_size)
        steps_per_epoch = math.ceil(per_rank_examples / (args.batch_size * args.grad_accum))
        total_steps = args.max_steps if args.max_steps else args.epochs * steps_per_epoch
        warmup_steps = args.warmup_steps if args.warmup_steps >= 0 else max(1, int(0.05 * total_steps))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda s: _lr_factor(s, warmup_steps, total_steps, min_ratio=0.1),
        )

        start_epoch, step, selection, stale = 0, 0, None, 0
        if args.resume:
            ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
            raw_model.load_state_dict(ckpt["model"], strict=True)
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = int(ckpt["epoch"])
            step = int(ckpt["step"])
            selection = ckpt.get("source_selection")
            if raw_model.arch in ("isotoken", "isotoken_unlocked"):
                prototypes = semantic_vecs.detach() if raw_model.arch == "isotoken_unlocked" else raw_model.head.prepare_prototypes(semantic_vecs).detach()

        model = DDP(raw_model, device_ids=[local_rank] if device.type == "cuda" else None) if world_size > 1 else raw_model
        output = Path(args.output)
        if is_main:
            output.mkdir(parents=True, exist_ok=True)
            (output / "arguments.json").write_text(json.dumps({**vars(args), "world_size": world_size}, indent=2))
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        sync(device)
        started = time.perf_counter()
        log_path = output / "training.jsonl"
        log_stream = log_path.open("a" if args.resume and log_path.exists() else "w") if is_main else None
        try:
            for epoch in range(start_epoch, args.epochs):
                order = list(range(len(train_set)))
                random.Random(args.seed + epoch).shuffle(order)
                # Ensure identical step count across DDP ranks.
                usable = (len(order) // world_size) * world_size
                rank_order = order[rank:usable:world_size]
                model.train()
                optimizer.zero_grad(set_to_none=True)
                running_loss, micro_batches, updates = 0.0, 0, 0
                epoch_start = time.perf_counter()
                for b_idx, start in enumerate(range(0, len(rank_order), args.batch_size)):
                    rows = [train_set[i] for i in rank_order[start:start + args.batch_size]]
                    target = targets_for(rows, codes, device)
                    is_accum_step = (micro_batches + 1) % args.grad_accum != 0 and (start + args.batch_size) < len(rank_order)
                    ctx = model.no_sync() if (isinstance(model, DDP) and is_accum_step) else _null_context()
                    with ctx:
                        with autocast(device):
                            logits, cal_logits = model(
                                rows,
                                tokenizer.pad_token_id,
                                device,
                                prototypes=prototypes,
                                special_ids=specials,
                                return_calibrated=True,
                                label_chunk_size=args.label_chunk_size,
                            )
                            if raw_model.arch in ("isotoken", "isotoken_unlocked") and raw_model.head.prior_tau > 0.0:
                                loss = F.binary_cross_entropy_with_logits(cal_logits.float(), target)
                                pure_loss = loss
                            else:
                                pure_loss = F.binary_cross_entropy_with_logits(logits.float(), target)
                                loss = pure_loss
                                if cal_logits is not None:
                                    loss = loss + F.binary_cross_entropy_with_logits(cal_logits, target)
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f"Non-finite training loss at step {step + 1}")
                        (loss / args.grad_accum).backward()
                    running_loss += float(pure_loss.item())
                    micro_batches += 1
                    if not is_accum_step:
                        norm = torch.nn.utils.clip_grad_norm_(
                            [p for p in raw_model.parameters() if p.requires_grad],
                            args.clip_grad,
                            error_if_nonfinite=True,
                        )
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        step += 1
                        updates += 1
                        if is_main and (step % args.log_every == 0 or updates == 1):
                            rec = {
                                "step": step,
                                "epoch": epoch,
                                "loss": running_loss / max(micro_batches, 1),
                                "grad_norm": float(norm),
                                "lr_enc": optimizer.param_groups[0]["lr"],
                                "lr_head": optimizer.param_groups[-1]["lr"],
                                "elapsed_s": round(time.perf_counter() - epoch_start, 2),
                            }
                            if raw_model.arch in ("isotoken", "isotoken_unlocked"):
                                rec["scale"] = round(float(raw_model.head.log_scale.exp().item()), 3)
                                rec["attn_scale"] = round(float(raw_model.head.attn_log_scale.exp().item()), 3)
                                rec["bias"] = round(float(raw_model.head.bias.item()), 3)
                            log_stream.write(json.dumps(rec) + "\n")
                            log_stream.flush()
                            print(json.dumps(rec), flush=True)
                        if args.max_steps and step >= args.max_steps:
                            break
                with torch.no_grad():
                    raw_model.eval()
                    eval_prototypes = (
                        raw_model.head.prepare_prototypes(semantic_vecs).detach()
                        if raw_model.arch == "isotoken_unlocked"
                        else prototypes
                    )
                val = validate_isotoken(
                    model,
                    validation,
                    codes,
                    freq10_indices,
                    eval_prototypes,
                    tokenizer,
                    args,
                    device,
                    rank,
                    world_size,
                )
                val["epoch"] = epoch
                val["step"] = step
                val["train_loss"] = running_loss / max(micro_batches, 1)
                val["epoch_seconds"] = time.perf_counter() - epoch_start
                if is_main:
                    ckpt_path = output / f"step-{step:08d}.pt"
                    sel_key = "prior_cal_micro_f1" if (raw_model.arch in ("isotoken", "isotoken_unlocked") and raw_model.head.prior_tau > 0.0) else "micro_f1"
                    improved = selection is None or val[sel_key] > selection.get(sel_key, selection["micro_f1"])
                    if improved:
                        stale = 0
                        selection = {
                            "epoch": epoch,
                            "step": step,
                            "micro_f1": val["micro_f1"],
                            "macro_f1": val["macro_f1"],
                            "macro_f1_supported": val["macro_f1_supported"],
                            "freq10_micro_f1": val.get("freq10_micro_f1"),
                            "freq10_macro_f1": val.get("freq10_macro_f1"),
                            "prior_cal_micro_f1": val.get("prior_cal_micro_f1"),
                            "prior_cal_freq10_micro_f1": val.get("prior_cal_freq10_micro_f1"),
                            "prior_cal_freq10_macro_f1": val.get("prior_cal_freq10_macro_f1"),
                            "threshold": val["threshold"],
                            "prior_cal_threshold": val.get("prior_cal_threshold"),
                            "selection_metric": sel_key,
                            "checkpoint": str(ckpt_path.resolve()),
                            "training_icd_version": 9,
                            "validation_icd_version": 9,
                        }
                        _save_isotoken_checkpoint(
                            ckpt_path,
                            raw_model,
                            optimizer,
                            scheduler,
                            args,
                            epoch + 1,
                            step,
                            val["threshold"],
                            codes,
                            label_counts,
                            origin,
                            subjects,
                            semantic_meta,
                            selection,
                        )
                    else:
                        stale += 1
                    _save_isotoken_checkpoint(
                        output / "latest.pt",
                        raw_model,
                        optimizer,
                        scheduler,
                        args,
                        epoch + 1,
                        step,
                        val["threshold"],
                        codes,
                        label_counts,
                        origin,
                        subjects,
                        semantic_meta,
                        selection,
                    )
                    (output / f"validation-{step:08d}.json").write_text(json.dumps(val, indent=2))
                    print(json.dumps(val), flush=True)
                stop_flag = torch.tensor([1 if (args.patience and stale >= args.patience) or (args.max_steps and step >= args.max_steps) else 0], device=device)
                if world_size > 1:
                    dist.broadcast(stop_flag, src=0)
                if int(stop_flag.item()) == 1:
                    break
        finally:
            if log_stream is not None:
                log_stream.close()

        sync(device)
        if is_main:
            summary = {
                "format": "isotoken_v1",
                "arch": raw_model.arch,
                "steps": step,
                "world_size": world_size,
                "wall_seconds": time.perf_counter() - started,
                "peak_gpu_mb": (torch.cuda.max_memory_allocated(device) / (1024 ** 2)) if device.type == "cuda" else 0.0,
                "semantic_meta": semantic_meta,
                "best": selection,
            }
            (output / "summary.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(summary), flush=True)
    finally:
        _cleanup_distributed(world_size)


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
