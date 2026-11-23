import json
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.feather as feather
import torch

from .codebook import load_codebook
from .data import load_tokenizer
from .inference import autocast, sync
from .label_views import LabelViews, encode_views
from .models import build_model
from .train import source_data
from .transfer_data import save_json


@torch.no_grad()
def encode_source_split(model, notes, output, tokenizer, args):
    lengths = pc.list_value_length(notes.table.column("chunks")).to_numpy().astype(np.int64)
    offsets = np.concatenate(([0], lengths.cumsum()))
    np.save(output / "offsets.npy", offsets)
    features = np.lib.format.open_memmap(output / "notes.npy", mode="w+", dtype="float32",
                                        shape=(int(offsets[-1]), model.encoder.config.hidden_size))
    started = time.perf_counter()
    for start in range(0, len(notes), args.batch_size):
        rows = [notes[i] for i in range(start, min(start + args.batch_size, len(notes)))]
        with autocast(args.device):
            pieces = model.encode_notes(rows, tokenizer.pad_token_id, args.device)
        features[offsets[start]:offsets[start + len(rows)]] = torch.cat([h for h, _ in pieces]).float().cpu().numpy()
        if start % (100 * args.batch_size) == 0:
            print(json.dumps({"split": notes.meta["split"], "encoded": start + len(rows), "total": len(notes)}), flush=True)
    features.flush()
    feather.write_feather(notes.table.select(["id", "subject_id", "labels"]), output / "rows.arrow")
    save_json(output / "metadata.json", notes.meta)
    sync(args.device)
    return {"rows": len(notes), "chunks": int(offsets[-1]), "encoding_seconds": time.perf_counter() - started}


@torch.no_grad()
def cache_source_features(args):
    train, validation, codes, _, origin, subjects = source_data(args)
    schema_root = Path(args.schemas)
    schema_meta = json.loads((schema_root / "metadata.json").read_text())
    if schema_meta["role"] != "source_structural_training":
        raise ValueError("Feature preparation requires source-only schemas")
    compiled_codes = load_codebook(schema_root / "codebook.json")
    if codes != compiled_codes[:len(codes)]:
        raise ValueError("Feature and schema source codebooks must match including hierarchy")
    groups = json.loads((schema_root / "groups.json").read_text())
    if str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    model = build_model("multiview", args.model_dir, codes, args.projection_dim).to(args.device).eval()
    tokenizer = load_tokenizer(args.model_dir)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    sync(args.device)
    label_started = time.perf_counter()
    with autocast(args.device):
        labels = encode_views(model, compiled_codes, groups, tokenizer, args)
    sync(args.device)
    label_seconds = time.perf_counter() - label_started
    torch.save({"vectors": labels.vectors.float().cpu(), "available": labels.available.cpu(),
                "operators": labels.operators.cpu()}, output / "labels.pt")
    results = {}
    for split, notes in (("train", train), ("validation", validation)):
        directory = output / split
        directory.mkdir()
        results[split] = encode_source_split(model, notes, directory, tokenizer, args)
    for name in ("codebook.json", "groups.json"):
        shutil.copyfile(schema_root / name, output / name)
    save_json(output / "metadata.json", {
        "role": "source_frozen_encoder_features", "tokenizer": origin, "encoder": origin,
        "hidden_size": model.encoder.config.hidden_size, "source_codes": len(codes),
        "source_subjects": subjects, "note_pooling": "masked mean per chunk",
        "source": {"train": str(args.train_data), "validation": str(args.validation_data)},
        "schemas": schema_meta, "label_max_len": args.label_max_len, "results": results,
        "label_encoding_seconds": label_seconds,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(args.device) if str(args.device).startswith("cuda") else 0,
    })
    print(json.dumps(results), flush=True)


class SourceFeatures:
    def __init__(self, root, split):
        if split not in ("train", "validation"):
            raise PermissionError("Source feature loading accepts train and validation only")
        directory = Path(root) / split
        self.meta = json.loads((directory / "metadata.json").read_text())
        if self.meta["icd_version"] != 9 or self.meta["split"] != split:
            raise ValueError("Only source ICD-9 features enter training and model selection")
        self.rows = feather.read_table(directory / "rows.arrow", memory_map=True).to_pylist()
        self.offsets = np.load(directory / "offsets.npy", mmap_mode="r")
        self.vectors = np.load(directory / "notes.npy", mmap_mode="r")

    def __len__(self):
        return len(self.rows)

    def hidden(self, index, device):
        values = self.vectors[self.offsets[index]:self.offsets[index + 1]]
        return torch.tensor(values, device=device)


def load_label_features(root, device, views):
    values = torch.load(Path(root) / "labels.pt", map_location=device, weights_only=True)
    labels = LabelViews(**values)
    if views == "definition":
        labels.available[:, 1:] = False
        if not labels.available[:, 0].all():
            raise ValueError("Definition-only source ablation requires descriptions for every schema")
    return labels
