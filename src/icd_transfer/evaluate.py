import json
import time
from pathlib import Path

import numpy as np
import torch

from .codebook import Code, load_codebook
from .data import Notes, load_tokenizer, targets_for
from .inference import compile_codebook, predict, sync
from .metrics import Counts
from .models import build_model


def score_dataset(model, kind, notes, codes, source_codes, tokenizer, args, thresholds, stream=None):
    compiled, compile_seconds = compile_codebook(model, kind, codes, source_codes, tokenizer, args)
    counts = Counts(thresholds, len(codes))
    limit = min(len(notes), args.validation_limit or len(notes))
    sync(args.device)
    start = time.perf_counter()
    for offset in range(0, limit, args.batch_size):
        rows = [notes[i] for i in range(offset, min(offset + args.batch_size, limit))]
        probabilities = predict(model, kind, rows, compiled, tokenizer, args).cpu().numpy()
        truth = targets_for(rows, codes, "cpu").numpy().astype(bool)
        counts.update(probabilities, truth)
        if stream is not None:
            write_predictions(stream, rows, codes, probabilities >= thresholds[0], truth)
    sync(args.device)
    seconds = time.perf_counter() - start
    return counts, {"examples": limit, "codebook_seconds": compile_seconds,
                    "inference_seconds": seconds, "milliseconds_per_note": seconds * 1000 / limit}


def write_predictions(stream, rows, codes, predicted, truth):
    for row, pred, actual in zip(rows, predicted, truth):
        result = {"id": row["id"], "subject_id": row["subject_id"],
                  "predicted": [code.key for code, keep in zip(codes, pred) if keep],
                  "truth": row["labels"], "tp": int((pred & actual).sum()),
                  "fp": int((pred & ~actual).sum()), "fn": int((~pred & actual).sum())}
        stream.write(json.dumps(result) + "\n")


def check_metadata(notes, tokenizer_origin):
    if notes.meta["tokenizer"] != tokenizer_origin:
        raise ValueError("Prepared tokens and model checkpoint use different tokenizers")


def evaluate(args):
    if args.split == "test" and not args.test_approved:
        raise PermissionError("First test evaluation needs user approval; then pass --test-approved")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["threshold"] is None:
        raise ValueError("This progress checkpoint has no source-validation threshold; resume source training first")
    notes = Notes(args.data, args.split, args.test_approved)
    check_metadata(notes, checkpoint["tokenizer_origin"])
    if args.split == "validation" and notes.meta["icd_version"] != 9:
        raise ValueError("Development validation is restricted to source ICD-9")
    model_args = checkpoint["args"]
    for key in ("head_max_len", "label_max_len", "ridge"):
        setattr(args, key, model_args[key])
    codes = load_codebook(Path(args.data) / "codebook.json")
    source_codes = [Code(**row) for row in checkpoint["source_codes"]]
    model = build_model(model_args["kind"], args.model_dir, source_codes,
                        model_args["projection_dim"], initialize=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(args.device).requires_grad_(False)
    tokenizer = load_tokenizer(args.model_dir)
    origin = json.loads((Path(args.model_dir) / "origin.json").read_text())
    check_metadata(notes, origin)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    threshold = checkpoint["threshold"]
    with (output / "predictions.jsonl").open("x") as stream:
        counts, timing = score_dataset(model, model_args["kind"], notes, codes, source_codes,
                                       tokenizer, args, [threshold], stream)
    result = {"split": args.split, "kind": model_args["kind"], "threshold": threshold,
              "threshold_source": "ICD-9 validation", **counts.result(0), **timing}
    result["evaluation_support_1_to_9"] = counts.result(0, np.flatnonzero((counts.tp[0] + counts.fn[0] > 0) &
                                                                 (counts.tp[0] + counts.fn[0] < 10)))
    train_subjects = set(checkpoint["source_subjects"])
    evaluation_subjects = set(json.loads((Path(args.data) / "subjects.json").read_text()))
    result["overlapping_source_patients"] = len(train_subjects & evaluation_subjects)
    result["patient_disjoint"] = patient_disjoint_metrics(output / "predictions.jsonl", codes, train_subjects)
    if args.groups:
        result["groups"] = group_metrics(counts, codes, args.groups)
    with (output / "metrics.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


def patient_disjoint_metrics(path, codes, train_subjects):
    counts = Counts([0.5], len(codes))
    keys, examples = [code.key for code in codes], 0
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["subject_id"] in train_subjects:
                continue
            pred, truth = set(row["predicted"]), set(row["truth"])
            counts.update(np.array([[key in pred for key in keys]]), np.array([[key in truth for key in keys]]))
            examples += 1
    return {"examples": examples, **counts.result(0)}


def group_metrics(counts, codes, path):
    positions = {code.key: i for i, code in enumerate(codes)}
    groups = json.loads(Path(path).read_text())
    return {name: counts.result(0, [positions[key] for key in keys]) for name, keys in groups.items()}

