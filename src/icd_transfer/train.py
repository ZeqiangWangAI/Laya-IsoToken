import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .codebook import load_codebook
from .data import Notes, load_tokenizer, targets_for
from .evaluate import check_metadata, score_dataset
from .inference import autocast, label_tokens, sync
from .models import build_model, native_prompt, native_scores
from .sampling import merged_labels, sample_candidates


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def source_data(args):
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
    active = [code for code in codes if train.meta["label_counts"].get(code.key, 0) > 0]
    return train, validation, codes, active, origin, sorted(train_subjects | val_subjects)


def batch_loss(model, rows, source_codes, tokenizer, args, rng):
    indices, weights = sample_candidates(rows, source_codes, args.candidates, rng, args.device)
    codes = [source_codes[i] for i in indices]
    target = targets_for(rows, codes, args.device)
    if args.kind == "native":
        prompts = [native_prompt(tokenizer, code.text, args.head_max_len) for code in codes]
        logits = native_scores(model, rows, prompts, tokenizer.pad_token_id, args.device, args.pair_batch_size)
        extra = logits.new_zeros(())
    else:
        notes = model.encode_notes(rows, tokenizer.pad_token_id, args.device)
        logits, extra = factored_loss(model, notes, codes, target, indices, tokenizer, args, rng)
    loss = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * weights).sum()
    return loss / (len(rows) * len(source_codes)) + args.merge_weight * extra


def factored_loss(model, notes, codes, target, indices, tokenizer, args, rng):
    if args.kind == "linear":
        logits = torch.stack([model.score_source(hidden, indices) for hidden, _ in notes])
        return logits, logits.new_zeros(())
    texts = [code.text for code in codes]
    count = args.merges if args.kind == "evidence" else 0
    merges, union = merged_labels(codes, target, count, rng)
    tokens = label_tokens(tokenizer, texts + merges, args.label_max_len)
    labels = model.encode_labels(tokens, tokenizer.pad_token_id, args.device)
    scores = torch.stack([model.score(hidden, labels) for hidden, _ in notes])
    extra = F.binary_cross_entropy_with_logits(scores[:, len(codes):], union) if merges else scores.new_zeros(())
    return scores[:, :len(codes)], extra


def save_checkpoint(path, model, optimizer, scheduler, args, epoch, next_batch, step,
                    threshold, source_codes, origin, source_subjects, sampling_rng, selection=None):
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "scheduler": scheduler.state_dict(), "args": vars(args),
               "epoch": epoch, "next_batch": next_batch, "step": step, "threshold": threshold,
               "source_codes": [asdict(code) for code in source_codes], "tokenizer_origin": origin,
               "source_subjects": source_subjects, "rng": torch.get_rng_state(),
               "sampling_rng": sampling_rng.getstate(), "source_selection": selection,
               "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    with path.open("xb") as stream:
        torch.save(payload, stream)


def restore(path, model, optimizer, scheduler, args, codes, origin, subjects, rng):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    stable = ("kind", "projection_dim", "seed", "batch_size", "candidates", "merges", "merge_weight",
              "lr", "weight_decay", "epochs", "warmup_steps", "ridge", "label_max_len", "head_max_len",
              "clip_grad", "pair_batch_size", "gradient_checkpointing")
    if any(checkpoint["args"][key] != getattr(args, key) for key in stable):
        raise ValueError("Resume must retain training hyperparameters")
    if checkpoint["source_codes"] != [asdict(code) for code in codes] or checkpoint["tokenizer_origin"] != origin:
        raise ValueError("Resume source codebook or tokenizer differs")
    if checkpoint["source_subjects"] != subjects:
        raise ValueError("Resume requires the same source patients")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    rng.setstate(checkpoint["sampling_rng"])
    torch.set_rng_state(checkpoint["rng"])
    if checkpoint["cuda_rng"]:
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    return checkpoint["epoch"], checkpoint["next_batch"], checkpoint["step"], checkpoint.get("source_selection")


def run_epoch(model, optimizer, scheduler, train, codes, tokenizer, args, epoch, skip, step, rng, log, save_progress):
    order = list(range(len(train)))
    random.Random(args.seed + epoch).shuffle(order)
    model.train()
    total_loss, updates = 0.0, 0
    for batch_index, start in enumerate(range(0, len(order), args.batch_size)):
        if batch_index < skip:
            continue
        rows = [train[i] for i in order[start:start + args.batch_size]]
        optimizer.zero_grad(set_to_none=True)
        with autocast(args.device):
            loss = batch_loss(model, rows, codes, tokenizer, args, rng)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        step += 1
        total_loss += loss.item()
        updates += 1
        record = {"step": step, "epoch": epoch, "batch": batch_index, "loss": loss.item(),
                  "gradient_norm": float(norm), "lr": scheduler.get_last_lr()[0]}
        log.write(json.dumps(record) + "\n")
        log.flush()
        if step % 10 == 0 or updates == 1:
            print(json.dumps(record), flush=True)
        if args.checkpoint_every and step % args.checkpoint_every == 0:
            save_progress(epoch, batch_index + 1, step)
        if args.max_steps and step >= args.max_steps:
            return step, batch_index + 1, total_loss / updates
    return step, math.ceil(len(train) / args.batch_size), total_loss / max(updates, 1)


def train(args):
    seed_everything(args.seed)
    train_set, validation, all_codes, source_codes, origin, subjects = source_data(args)
    tokenizer = load_tokenizer(args.model_dir)
    model = build_model(args.kind, args.model_dir, source_codes, args.projection_dim, not bool(args.resume))
    model.to(args.device)
    if str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    if args.gradient_checkpointing and args.kind != "native":
        for module in model.modules():
            if hasattr(module, "gradient_checkpointing_enable"):
                module.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * math.ceil(len(train_set) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:
        min((step + 1) / max(args.warmup_steps, 1), max(total_steps - step, 0) /
            max(total_steps - args.warmup_steps, 1)))
    rng = random.Random(args.seed)
    epoch, skip, step, selection = (0, 0, 0, None)
    if args.resume:
        epoch, skip, step, selection = restore(args.resume, model, optimizer, scheduler, args,
                                    source_codes, origin, subjects, rng)
        if epoch >= args.epochs or (args.max_steps and step >= args.max_steps):
            raise ValueError("The checkpoint has already reached the requested training endpoint")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "arguments.json").write_text(json.dumps(vars(args), indent=2))
    def save_progress(current, next_batch, current_step):
        save_checkpoint(output / f"train-step-{current_step:08d}.pt", model, optimizer, scheduler,
                        args, current, next_batch, current_step, None, source_codes, origin, subjects, rng, selection)
    sync(args.device)
    started = time.perf_counter()
    with (output / "training.jsonl").open("x") as log:
        for current in range(epoch, args.epochs):
            step, next_batch, loss = run_epoch(model, optimizer, scheduler, train_set, source_codes,
                                               tokenizer, args, current, skip, step, rng, log, save_progress)
            counts, timing = score_dataset(model, args.kind, validation, all_codes, source_codes,
                                           tokenizer, args, np.arange(0.05, 1.0, 0.05))
            best = counts.best()
            threshold = float(counts.thresholds[best])
            result = {"epoch": current, "step": step, "train_loss": loss,
                      "threshold": threshold, **counts.result(best), **timing}
            if selection is None or result["micro_f1"] > selection["micro_f1"]:
                selection = {"step": step, "micro_f1": result["micro_f1"], "threshold": threshold,
                             "checkpoint": str((output / f"step-{step:08d}.pt").resolve()),
                             "training_icd_version": 9, "validation_icd_version": 9}
            completed = next_batch == math.ceil(len(train_set) / args.batch_size)
            saved_epoch, saved_batch = (current + 1, 0) if completed else (current, next_batch)
            save_checkpoint(output / f"step-{step:08d}.pt", model, optimizer, scheduler, args,
                            saved_epoch, saved_batch, step, threshold, source_codes, origin, subjects, rng, selection)
            (output / f"validation-{step:08d}.json").write_text(json.dumps(result, indent=2))
            print(json.dumps(result), flush=True)
            skip = 0
            if args.max_steps and step >= args.max_steps:
                break
    sync(args.device)
    summary = {"steps": step, "wall_seconds": time.perf_counter() - started,
               "peak_gpu_bytes": torch.cuda.max_memory_allocated(args.device) if str(args.device).startswith("cuda") else 0,
               "dataset_version": train_set.meta["dataset_version"], "model_origin": origin, "best": selection}
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary

