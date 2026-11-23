import csv
import gzip
import json
import sys
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather

from .codebook import load_codebook, write_codebook
from .data import SCHEMA, chunk_tokens, load_tokenizer
from .transfer_data import clinical_text, save_json


def source_annotations(root, codes):
    records, metadata = {}, {}
    for split in ("train", "validation"):
        directory = root / split
        meta = json.loads((directory / "metadata.json").read_text())
        if meta["split"] != split or meta["icd_version"] != 9:
            raise ValueError("Clinical source preparation requires ICD-9 train and validation")
        original = load_codebook(directory / "codebook.json")
        if [c.key for c in original] != [c.key for c in codes]:
            raise ValueError("Clinical source must preserve the benchmark candidate order")
        rows = feather.read_table(directory / f"{split}.arrow", columns=["id", "subject_id", "labels"]).to_pylist()
        for row in rows:
            if row["id"] in records:
                raise ValueError("Source splits contain overlapping admissions")
            records[row["id"]] = (split, row)
        metadata[split] = meta
    return records, metadata


def write_clinical_batch(batch, writers, tokenizer, args, counts, stats, subjects):
    tokens = tokenizer([row["text"] for _, row, _ in batch], add_special_tokens=False)["input_ids"]
    outputs = {"train": [], "validation": []}
    for (split, row, dropped_words), ids in zip(batch, tokens):
        chunks, discarded = chunk_tokens(ids, tokenizer, args.chunk_size, args.max_chunks)
        outputs[split].append({k: row[k] for k in ("id", "subject_id", "labels")} | {"chunks": chunks})
        subjects[split].add(row["subject_id"])
        counts[split].update(row["labels"])
        stats[split].update(rows=1, tokens=len(ids), chunks=len(chunks),
                            truncated_rows=int(discarded > 0), discarded_tokens=discarded,
                            word_truncated_rows=int(dropped_words > 0), discarded_words=dropped_words,
                            unknown_tokens=ids.count(tokenizer.unk_token_id))
    for split, rows in outputs.items():
        if rows:
            writers[split].write_table(pa.Table.from_pylist(rows, schema=SCHEMA))


def stream_clinical(args, annotations, writers, tokenizer, counts, stats, subjects):
    seen, batch = set(), []
    csv.field_size_limit(sys.maxsize)
    with gzip.open(args.discharge, "rt", newline="") as stream:
        for raw in csv.DictReader(stream):
            identifier = raw["hadm_id"]
            if identifier not in annotations:
                continue
            split, row = annotations[identifier]
            if identifier in seen or raw["subject_id"] != row["subject_id"]:
                raise ValueError("Source admission is duplicated or its patient does not match")
            seen.add(identifier)
            words = clinical_text(raw["text"]).split()
            batch.append((split, row | {"text": " ".join(words[:args.max_words])}, max(0, len(words) - args.max_words)))
            if len(batch) == 128:
                write_clinical_batch(batch, writers, tokenizer, args, counts, stats, subjects)
                batch.clear()
            if len(seen) % 10000 == 0:
                print(json.dumps({"rows": len(seen), "expected": len(annotations)}), flush=True)
    if batch:
        write_clinical_batch(batch, writers, tokenizer, args, counts, stats, subjects)
    if seen != annotations.keys():
        raise ValueError(f"Source notes missing for {len(annotations.keys() - seen)} admissions")


def prepare_source_clinical(args):
    codes = load_codebook(args.codebook)
    annotations, metadata = source_annotations(Path(args.source), codes)
    tokenizer = load_tokenizer(args.model_dir)
    origin = json.loads((Path(args.model_dir) / "origin.json").read_text())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    counts = {split: Counter() for split in metadata}
    stats = {split: Counter() for split in metadata}
    subjects = {split: set() for split in metadata}
    with ExitStack() as stack:
        writers = {}
        for split in metadata:
            (output / split).mkdir()
            sink = stack.enter_context(pa.OSFile(str(output / split / f"{split}.arrow"), "wb"))
            writers[split] = stack.enter_context(pa.ipc.new_file(sink, SCHEMA))
        stream_clinical(args, annotations, writers, tokenizer, counts, stats, subjects)
    if subjects["train"] & subjects["validation"]:
        raise ValueError("Source train and validation patients overlap")
    for split, original in metadata.items():
        if dict(counts[split]) != original["label_counts"]:
            raise ValueError("Clinical source labels differ from the aligned benchmark")
        write_codebook(codes, output / split / "codebook.json")
        save_json(output / split / "subjects.json", sorted(subjects[split]))
        save_json(output / split / "metadata.json", {
            **original, **stats[split], "label_counts": dict(counts[split]),
            "dataset_version": "mimiciv-2.2-edin-full-clinical-multilingual", "tokenizer": origin,
            "max_words": args.max_words, "chunk_size": args.chunk_size, "max_chunks": args.max_chunks,
            "pad_id": tokenizer.pad_token_id, "text_normalization": "NFC; whitespace; Unicode punctuation and numbers retained",
            "source_prepared": str(Path(args.source) / split), "role": "source_development",
        })
    result = {split: dict(stats[split]) | {"patients": len(subjects[split])} for split in metadata}
    save_json(output / "summary.json", result | {"patient_overlap": 0})
    print(json.dumps(result), flush=True)
