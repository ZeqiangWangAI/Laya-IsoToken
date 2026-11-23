import json
from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

from .codebook import load_codebook, normalize_code, write_codebook
from .data import SCHEMA, chunk_tokens, iter_input, load_tokenizer


def published_code(code):
    value = normalize_code(code.code)
    if code.system == "icd10_proc":
        return value
    if code.system == "icd10_diag":
        return value[:3] + "." + value[3:]
    width = 2 if code.system == "icd9_proc" else 4 if value.startswith("E") else 3
    return value[:width] + "." + value[width:] if len(value) > width else value


def fixed_codebooks(root, references, output):
    scopes = json.loads((references / "data/code_scopes.json").read_text())
    descriptions = json.loads((references / "data/code_descriptions.json").read_text())
    books = {}
    for version in (9, 10):
        codes = load_codebook(root / f"icd{version}-codebook.json")
        lookup = {published_code(code): code for code in codes}
        if len(lookup) != len(codes):
            raise ValueError("Published code notation must identify exactly one code")
        for scope in ("full", "frequent"):
            indices = scopes[f"dataset:mimiciv_icd{version}"][f"code_scope:{scope}"]
            names = sorted(indices, key=indices.get)
            book = [replace(lookup[name], description=descriptions[f"icd{version}"][name]) for name in names]
            write_codebook(book, output / f"icd{version}-{scope}.json")
            books[version, scope] = book
        frequent = {code.key for code in books[version, "frequent"]}
        full = {code.key for code in books[version, "full"]}
        if not frequent <= full:
            raise ValueError("Frequent codes must be a subset of the full benchmark")
        groups = {"benchmark_frequent": sorted(frequent), "benchmark_rare": sorted(full - frequent)}
        (output / f"icd{version}-groups.json").write_text(json.dumps(groups, indent=2))
    return books


def source_ids(root, references):
    expected = json.loads((root / "splits/edin-icd9/source_ids.json").read_text())
    table = feather.read_table(references / "data/splits/mimiciv_icd9_split.feather")
    for split, name in (("train", "train"), ("validation", "val")):
        public = set(map(str, table.filter(pc.equal(table["split"], name))["_id"].to_pylist()))
        if public != set(map(str, expected[split])):
            raise ValueError(f"Edin and DualLAAT source IDs differ: {split}")
    return {split: set(map(str, ids)) for split, ids in expected.items()}


def align_record(raw, prepared, full_keys, tokenizer):
    labels = {f"{kind}:{normalize_code(code)}" for kind in ("icd9_diag", "icd9_proc") for code in raw[kind]}
    if str(raw["_id"]) != prepared["id"] or str(raw["subject_id"]) != prepared["subject_id"]:
        raise ValueError("Source text and prepared rows must refer to the same admission")
    if labels != set(prepared["labels"]) or not labels <= full_keys:
        raise ValueError("Source labels must agree and belong to the published full scope")
    words = raw["text"].split()
    record = dict(prepared)
    discarded_tokens = 0
    # Reuse existing tokens unless the shared word limit or previous chunk limit can affect them.
    if len(words) > 4000 or len(prepared["chunks"]) == 32:
        tokens = tokenizer(" ".join(words[:4000]), add_special_tokens=False)["input_ids"]
        record["chunks"], discarded_tokens = chunk_tokens(tokens, tokenizer, 256, 32)
    return record, max(0, len(words) - 4000), discarded_tokens


def write_batch(writers, buffer):
    for scope, rows in buffer.items():
        writers[scope].write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
        rows.clear()


def append_scopes(record, keys, counts, buffer):
    for scope, allowed in keys.items():
        labels = sorted(set(record["labels"]) & allowed)
        if not labels:
            raise ValueError("Published source cohort contains a label-empty admission")
        counts[scope].update(labels)
        buffer[scope].append({**record, "labels": labels})


def prepare_split(root, output, split, expected_ids, books, tokenizer):
    keys = {scope: {code.key for code in books[9, scope]} for scope in ("full", "frequent")}
    counts = {scope: Counter() for scope in keys}
    buffer = {scope: [] for scope in keys}
    stats = Counter()
    ids, subjects = set(), set()
    raw_rows = iter_input(root / f"source-text/{split}.jsonl", ["_id", "subject_id", "text", "icd9_diag", "icd9_proc"])
    old_rows = iter_input(root / f"prepared/source/{split}/{split}.arrow", SCHEMA.names)
    with ExitStack() as stack:
        writers = {}
        for scope in keys:
            directory = output / scope / "source" / split
            directory.mkdir(parents=True)
            sink = stack.enter_context(pa.OSFile(str(directory / f"{split}.arrow"), "wb"))
            writers[scope] = stack.enter_context(pa.ipc.new_file(sink, SCHEMA))
        for raw, old in zip(raw_rows, old_rows, strict=True):
            record, words, tokens = align_record(raw, old, keys["full"], tokenizer)
            if record["id"] in ids:
                raise ValueError("Duplicate source admission")
            ids.add(record["id"])
            subjects.add(record["subject_id"])
            stats.update(rows=1, word_truncated_rows=int(words > 0), discarded_words=words,
                         truncated_rows=int(tokens > 0), discarded_tokens=tokens,
                         tokens=sum(len(chunk) - 2 for chunk in record["chunks"]) + tokens,
                         chunks=len(record["chunks"]))
            append_scopes(record, keys, counts, buffer)
            if len(buffer["full"]) == 128:
                write_batch(writers, buffer)
        if buffer["full"]:
            write_batch(writers, buffer)
    if ids != expected_ids:
        raise ValueError(f"Prepared {split} IDs differ from the published source cohort")
    original = json.loads((root / f"prepared/source/{split}/metadata.json").read_text())
    for scope in keys:
        directory = output / scope / "source" / split
        write_codebook(books[9, scope], directory / "codebook.json")
        metadata = {**original, **stats, "dataset_version": f"mimiciv-2.2-edin-{scope}-4000words-laya",
                    "max_words": 4000, "label_counts": dict(counts[scope]),
                    "candidate_codes": len(keys[scope]), "label_scope": scope,
                    "label_scope_origin": json.loads((output / "origin.json").read_text()),
                    "source_prepared": str(root / f"prepared/source/{split}")}
        (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))
        (directory / "subjects.json").write_text(json.dumps(sorted(subjects)))
    summary = {**stats, "patients": len(subjects), "published_ids_match": True,
               "observed_codes": {scope: len(counts[scope]) for scope in keys}}
    print(json.dumps({"split": split, **summary}), flush=True)
    return summary, subjects


def prepare_benchmark(args):
    root, references, output = map(Path, (args.data_root, args.references, args.output))
    output.mkdir(parents=True, exist_ok=False)
    (output / "codebooks").mkdir()
    origin = json.loads((references / "origin.json").read_text())
    (output / "origin.json").write_text(json.dumps(origin, indent=2))
    books = fixed_codebooks(root, references, output / "codebooks")
    expected = source_ids(root, references)
    tokenizer = load_tokenizer(args.model_dir)
    model_origin = json.loads((Path(args.model_dir) / "origin.json").read_text())
    summaries, subjects = {}, {}
    for split in ("train", "validation"):
        original = json.loads((root / f"prepared/source/{split}/metadata.json").read_text())
        if (original["chunk_size"], original["max_chunks"], original["tokenizer"]) != (256, 32, model_origin):
            raise ValueError("Reused source chunks must use the same tokenizer and chunk settings")
        summaries[split], subjects[split] = prepare_split(root, output, split, expected[split], books, tokenizer)
    summaries["patient_overlap"] = len(subjects["train"] & subjects["validation"])
    summaries["candidate_codes"] = {f"icd{v}_{s}": len(book) for (v, s), book in books.items()}
    (output / "summary.json").write_text(json.dumps(summaries, indent=2))
