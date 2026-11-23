import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .calibration import primary_semantic_vectors
from .codebook import load_codebook
from .data import targets_for
from .inference import autocast, score_chunks, sync
from .metrics import Counts
from .models import build_multiview_head
from .sampling import sample_candidates
from .source_features import SourceFeatures, load_label_features
from .source_schemas import sample_schemas, schema_targets
from .train import seed_everything
from .transfer_data import save_json


def load_inputs(args):
    root = Path(args.features)
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata["role"] != "source_frozen_encoder_features":
        raise ValueError("Training requires frozen source-only feature data")
    train, validation = SourceFeatures(root, "train"), SourceFeatures(root, "validation")
    codes = load_codebook(root / "codebook.json")
    groups = json.loads((root / "groups.json").read_text())
    labels = load_label_features(root, args.device, args.views)
    return metadata, train, validation, codes, groups, labels


def score_notes(model, hidden, vectors):
    if hasattr(model, "score_batch"):
        return model.score_batch(hidden, vectors)
    return torch.stack([model.score(note, vectors) for note in hidden])


def training_loss(model, hidden, rows, active, active_labels, active_source_indices, positions, labels, groups, args, atomic_rng, schema_rng):
    if args.candidates <= 0 or args.candidates >= len(active):
        vectors = model.attach_source_residuals(active_labels, active_source_indices) if hasattr(model, "attach_source_residuals") else active_labels
        logits = score_notes(model, hidden, vectors)
        truth = targets_for(rows, active, args.device)
        atomic = F.binary_cross_entropy_with_logits(logits, truth)
    else:
        min_negatives = getattr(args, "min_negatives", 1)
        indices, weights = sample_candidates(rows, active, args.candidates, atomic_rng, args.device, min_negatives)
        candidates = [active[i] for i in indices]
        cand_positions = [positions[c.key] for c in candidates]
        vectors = labels[cand_positions]
        if hasattr(model, "attach_source_residuals"):
            vectors = model.attach_source_residuals(vectors, cand_positions)
        logits = score_notes(model, hidden, vectors)
        truth = targets_for(rows, candidates, args.device)
        atomic = (F.binary_cross_entropy_with_logits(logits, truth, reduction="none") * weights).sum()
        atomic = atomic / (len(rows) * len(active))
    if args.schema_mode == "none":
        return atomic
    selected = sample_schemas(rows, groups, args.schema_candidates, schema_rng)
    vectors = labels[[positions[group["key"]] for group in selected]]
    scores = score_notes(model, hidden, vectors)
    target = schema_targets(rows, selected, args.device)
    return atomic + args.schema_weight * F.binary_cross_entropy_with_logits(scores, target)


def score_batch_chunks(model, notes, labels, label_batch_size):
    if len(labels) == 0:
        return torch.empty((len(notes), 0), device=notes[0].device)
    return torch.cat(
        [model.score_batch(notes, labels[start:start + label_batch_size])
         for start in range(0, len(labels), label_batch_size)],
        dim=1,
    )


@torch.no_grad()
def validate(model, validation, source_codes, groups, labels, positions, args):
    model.eval()
    thresholds = np.arange(0.05, 1.0, 0.05)
    atomic = Counts(thresholds, len(source_codes))
    selected = [group for group in groups if group["partition"] == "selection"]
    structural = Counts(thresholds, len(selected))
    atomic_indices = [positions[c.key] for c in source_codes]
    atomic_labels = labels[atomic_indices]
    if hasattr(model, "attach_source_residuals"):
        atomic_labels = model.attach_source_residuals(atomic_labels, atomic_indices)
    schema_labels = labels[[positions[group["key"]] for group in selected]]
    limit = min(len(validation), args.validation_limit or len(validation))
    started = time.perf_counter()
    if hasattr(model, "score_batch"):
        for start in range(0, limit, args.batch_size):
            stop = min(start + args.batch_size, limit)
            rows = validation.rows[start:stop]
            with autocast(args.device):
                notes = [validation.hidden(i, args.device) for i in range(start, stop)]
                probabilities = score_batch_chunks(model, notes, atomic_labels, args.label_batch_size).sigmoid()
                schema_prob = score_batch_chunks(model, notes, schema_labels, args.label_batch_size).sigmoid()
            atomic.update(probabilities.cpu().numpy(), targets_for(rows, source_codes, "cpu").numpy())
            structural.update(schema_prob.cpu().numpy(), schema_targets(rows, selected, "cpu").numpy())
    else:
        for index in range(limit):
            row = validation.rows[index]
            with autocast(args.device):
                hidden = validation.hidden(index, args.device)
                probabilities = score_chunks(model, hidden, atomic_labels, args.label_batch_size).sigmoid()
                schema_prob = score_chunks(model, hidden, schema_labels, args.label_batch_size).sigmoid()
            atomic.update(probabilities.cpu().numpy()[None], targets_for([row], source_codes, "cpu").numpy())
            structural.update(schema_prob.cpu().numpy()[None], schema_targets([row], selected, "cpu").numpy())
    best = atomic.best()
    result = {"atomic": atomic.result(best), "threshold": float(thresholds[best]), "examples": limit}
    for operator in ("OR", "AND"):
        indices = [i for i, group in enumerate(selected) if group["operator"] == operator]
        result[operator] = structural.result(best, indices)
    score_keys = {"none": ("atomic",), "or": ("atomic", "OR"), "or_and": ("atomic", "OR", "AND")}[args.schema_mode]
    result["selection_score"] = sum(result[key]["micro_f1"] for key in score_keys) / len(score_keys)
    sync(args.device)
    result["validation_seconds"] = time.perf_counter() - started
    return result


def save_state(path, model, optimizer, scheduler, progress, args, metadata, codes, groups, rngs, source_label_counts=None):
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "progress": progress, "args": vars(args), "feature_metadata": metadata,
             "source_codes": [asdict(code) for code in codes], "groups": groups,
             "source_label_counts": source_label_counts,
             "rng": torch.get_rng_state(), "sampling_rngs": [rng.getstate() for rng in rngs],
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    with path.open("xb") as stream:
        torch.save(state, stream)


def restore_state(path, model, optimizer, scheduler, args, metadata, codes, groups, rngs):
    state = torch.load(path, map_location="cpu", weights_only=False)
    stable = ("seed", "projection_dim", "batch_size", "candidates", "schema_candidates", "schema_mode",
              "schema_weight", "views", "lr", "weight_decay", "epochs", "warmup_steps", "clip_grad",
              "patience", "validation_limit")
    if any(state["args"][key] != getattr(args, key) for key in stable):
        raise ValueError("Resume must preserve training and model-selection settings")
    head_mode = getattr(args, "head_mode", "mlp")
    if (state["args"].get("head_mode", "mlp") != head_mode or
            state["args"].get("min_negatives", 1) != getattr(args, "min_negatives", 1)):
        raise ValueError("Resume must preserve head_mode and min_negatives settings")
    if head_mode in ("attention_cast", "isometric_cast") and state["args"].get("residual_dropout", 0.25) != getattr(args, "residual_dropout", 0.25):
        raise ValueError("Resume must preserve residual_dropout setting")
    if head_mode in ("isometric", "isometric_cast") and state["args"].get("view_dropout", 0.5) != getattr(args, "view_dropout", 0.5):
        raise ValueError("Resume must preserve view_dropout setting")
    if (state["feature_metadata"] != metadata or state["groups"] != groups or
            state["source_codes"] != [asdict(code) for code in codes]):
        raise ValueError("Resume source features or schema definitions differ")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["rng"])
    for rng, saved in zip(rngs, state["sampling_rngs"], strict=True):
        rng.setstate(saved)
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["progress"]


def run_epoch(model, optimizer, scheduler, train, active, active_labels, active_source_indices, positions, labels, groups, args, progress, rngs, log, save_progress):
    order = list(range(len(train)))
    random.Random(args.seed + progress["epoch"]).shuffle(order)
    model.train()
    start_batch = progress["next_batch"]
    for batch in range(start_batch, math.ceil(len(order) / args.batch_size)):
        indices = order[batch * args.batch_size:(batch + 1) * args.batch_size]
        rows = [train.rows[i] for i in indices]
        hidden = [train.hidden(i, args.device) for i in indices]
        optimizer.zero_grad(set_to_none=True)
        with autocast(args.device):
            loss = training_loss(model, hidden, rows, active, active_labels, active_source_indices, positions, labels, groups, args, *rngs)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite source training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        progress["step"] += 1
        progress["next_batch"] = batch + 1
        record = {"step": progress["step"], "epoch": progress["epoch"], "batch": batch,
                  "loss": loss.item(), "gradient_norm": float(norm), "lr": scheduler.get_last_lr()[0]}
        log.write(json.dumps(record) + "\n")
        log.flush()
        if progress["step"] % 100 == 0 or batch == start_batch:
            print(json.dumps(record), flush=True)
        if args.checkpoint_every and progress["step"] % args.checkpoint_every == 0:
            save_progress()
        if args.max_steps and progress["step"] >= args.max_steps:
            break


def finish_epoch(model, validation, codes, groups, labels, positions, args, progress, output):
    result = validate(model, validation, codes, groups, labels, positions, args)
    result.update(epoch=progress["epoch"], step=progress["step"])
    save_json(output / f"validation-{progress['step']:08d}.json", result)
    previous = progress["best"]
    improved = previous is None or result["selection_score"] > previous["selection_score"]
    progress["bad_epochs"] = 0 if improved else progress["bad_epochs"] + 1
    progress["epoch"] += 1
    progress["next_batch"] = 0
    if improved:
        progress["best"] = result | {"checkpoint": str((output / f"step-{progress['step']:08d}.pt").resolve())}
    print(json.dumps(result), flush=True)


def train_multiview(args):
    seed_everything(args.seed)
    metadata, train, validation, codes, groups, labels = load_inputs(args)
    source_codes = codes[:metadata["source_codes"]]
    source_label_counts = [int(train.meta["label_counts"].get(c.key, 0)) for c in source_codes]
    active = [code for code in source_codes if train.meta["label_counts"].get(code.key, 0) > 0]
    positions = {code.key: i for i, code in enumerate(codes)}
    active_source_indices = [positions[c.key] for c in active]
    active_labels = labels[active_source_indices]
    training_groups = [g for g in groups if g["partition"] == "train" and
                       (args.schema_mode == "or_and" or g["operator"] == "OR")]
    head_mode = getattr(args, "head_mode", "mlp")
    residual_dropout = getattr(args, "residual_dropout", 0.25)
    view_dropout = getattr(args, "view_dropout", 0.5)
    model = build_multiview_head(
        metadata["hidden_size"],
        args.projection_dim,
        head_mode,
        source_size=len(source_codes),
        residual_dropout=residual_dropout,
        view_dropout=view_dropout,
    ).to(args.device)
    if hasattr(model, "set_label_mean"):
        source_all_labels = labels[[positions[c.key] for c in source_codes]]
        model.set_label_mean(primary_semantic_vectors(source_all_labels).float().mean(0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * math.ceil(len(train) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:
        min((step + 1) / max(args.warmup_steps, 1), max(total_steps - step, 0) / max(total_steps - args.warmup_steps, 1)))
    rngs = random.Random(args.seed), random.Random(args.seed + 1000003)
    progress = {"epoch": 0, "next_batch": 0, "step": 0, "bad_epochs": 0, "best": None}
    if args.resume:
        progress = restore_state(args.resume, model, optimizer, scheduler, args, metadata, source_codes, groups, rngs)
    pending_validation = progress["next_batch"] == math.ceil(len(train) / args.batch_size)
    if (progress["epoch"] >= args.epochs or (args.max_steps and progress["step"] >= args.max_steps and not pending_validation)
            or (args.patience and progress["bad_epochs"] >= args.patience)):
        raise ValueError("The checkpoint has reached the requested endpoint")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "arguments.json", vars(args))
    def save_progress():
        save_state(output / f"train-step-{progress['step']:08d}.pt", model, optimizer, scheduler, progress,
                   args, metadata, source_codes, groups, rngs, source_label_counts=source_label_counts)
    sync(args.device)
    started = time.perf_counter()
    with (output / "training.jsonl").open("x") as log:
        while progress["epoch"] < args.epochs:
            run_epoch(model, optimizer, scheduler, train, active, active_labels, active_source_indices, positions, labels,
                      training_groups, args, progress, rngs, log, save_progress)
            completed = progress["next_batch"] == math.ceil(len(train) / args.batch_size)
            if completed:
                finish_epoch(model, validation, source_codes, groups, labels, positions, args, progress, output)
            save_state(output / f"step-{progress['step']:08d}.pt", model, optimizer, scheduler, progress,
                       args, metadata, source_codes, groups, rngs, source_label_counts=source_label_counts)
            stopped = args.max_steps and progress["step"] >= args.max_steps
            if stopped or (args.patience and progress["bad_epochs"] >= args.patience):
                break
    sync(args.device)
    summary = {**progress, "wall_seconds": time.perf_counter() - started,
               "peak_gpu_bytes": torch.cuda.max_memory_allocated(args.device) if str(args.device).startswith("cuda") else 0,
               "encoder_trainable": False, "selection_source": "ICD-9 validation only"}
    save_json(output / "summary.json", summary)
    return summary
