import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.feather as feather
import torch
from transformers import PreTrainedTokenizerFast

from .codebook import load_codebook, normalize_code, write_codebook


SCHEMA = pa.schema([
    ("id", pa.string()), ("subject_id", pa.string()),
    ("chunks", pa.list_(pa.list_(pa.int32()))),
    ("labels", pa.list_(pa.string())),
])


def load_tokenizer(model_dir):
    root = Path(model_dir) / "tokenizer"
    config = json.loads((root / "tokenizer_config.json").read_text())
    names = ("cls_token", "sep_token", "pad_token", "mask_token", "unk_token")
    special = {name: config[name] for name in names if name in config}
    return PreTrainedTokenizerFast(tokenizer_file=str(root / "tokenizer.json"), **special)


def chunk_tokens(ids, tokenizer, chunk_size, max_chunks):
    width = chunk_size - 2
    if width < 1 or max_chunks < 1:
        raise ValueError("chunk_size must exceed 2; max_chunks must be positive")
    pieces = [ids[start:start + width] for start in range(0, len(ids), width)] or [[]]
    chunks = [[tokenizer.cls_token_id, *piece, tokenizer.sep_token_id]
              for piece in pieces[:max_chunks]]
    return chunks, max(0, len(ids) - width * max_chunks)


def iter_input(path, columns):
    path = Path(path)
    formats = {".parquet": "parquet", ".feather": "ipc", ".arrow": "ipc", ".jsonl": "json"}
    dataset = ds.dataset(path, format=formats[path.suffix])
    scanner = dataset.scanner(columns=columns, batch_size=128)
    for batch in scanner.to_batches():
        yield from batch.to_pylist()


def convert_split(path, output, codes, tokenizer, args):
    columns = [args.id_column, args.subject_column, args.text_column,
               f"icd{args.version}_diag", f"icd{args.version}_proc"]
    lookup = {(code.system, normalize_code(code.code)): code.key for code in codes}
    counts, subjects, ids = Counter(), set(), set()
    buffer = []
    stats = {"rows": 0, "tokens": 0, "chunks": 0, "truncated_rows": 0, "discarded_tokens": 0}
    with pa.OSFile(str(output), "wb") as sink, pa.ipc.new_file(sink, SCHEMA) as writer:
        for row in iter_input(path, columns):
            record, dropped = convert_row(row, lookup, tokenizer, args)
            if record["id"] in ids:
                raise ValueError(f"Duplicate admission ID: {record['id']}")
            ids.add(record["id"])
            subjects.add(record["subject_id"])
            counts.update(record["labels"])
            stats["rows"] += 1
            stats["tokens"] += sum(len(chunk) - 2 for chunk in record["chunks"]) + dropped
            stats["chunks"] += len(record["chunks"])
            stats["truncated_rows"] += int(dropped > 0)
            stats["discarded_tokens"] += dropped
            buffer.append(record)
            if len(buffer) == 128:
                writer.write_table(pa.Table.from_pylist(buffer, schema=SCHEMA))
                buffer.clear()
            if stats["rows"] % 5000 == 0:
                print(json.dumps({"split": args.split, **stats}), flush=True)
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=SCHEMA))
    if stats["rows"] == 0:
        raise ValueError(f"Empty split: {path}")
    return {**stats, "label_counts": dict(counts)}, subjects


def convert_row(row, lookup, tokenizer, args):
    labels = []
    for kind in ("diag", "proc"):
        system = f"icd{args.version}_{kind}"
        labels.extend(lookup[(system, normalize_code(code))] for code in row[system])
    if not isinstance(row[args.text_column], str) or not labels:
        raise ValueError("Each admission needs text and at least one code")
    tokens = tokenizer(row[args.text_column], add_special_tokens=False)["input_ids"]
    chunks, discarded = chunk_tokens(tokens, tokenizer, args.chunk_size, args.max_chunks)
    return {"id": str(row[args.id_column]), "subject_id": str(row[args.subject_column]),
            "chunks": chunks, "labels": sorted(set(labels))}, discarded


def prepare(args):
    if args.split == "test" and not args.test_approved:
        raise PermissionError("Test preparation requires explicit user approval and --test-approved")
    if args.split != "test" and args.version != 9:
        raise ValueError("Development preparation is restricted to source ICD-9")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    codes = load_codebook(args.codebook)
    if {code.system for code in codes} - {f"icd{args.version}_diag", f"icd{args.version}_proc"}:
        raise ValueError("Codebook version does not match the requested dataset")
    tokenizer = load_tokenizer(args.model_dir)
    stats, subjects = convert_split(args.input, output / f"{args.split}.arrow", codes, tokenizer, args)
    write_codebook(codes, output / "codebook.json")
    origin = json.loads((Path(args.model_dir) / "origin.json").read_text())
    metadata = {"split": args.split, "icd_version": args.version, "dataset_version": args.dataset_version,
                "chunk_size": args.chunk_size, "max_chunks": args.max_chunks,
                "tokenizer": origin, "pad_id": tokenizer.pad_token_id, **stats}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (output / "subjects.json").write_text(json.dumps(sorted(subjects)))
    print(json.dumps({key: value for key, value in metadata.items() if key != "label_counts"}))


class Notes(torch.utils.data.Dataset):
    def __init__(self, directory, split, test_approved=False):
        if split == "test" and not test_approved:
            raise PermissionError("Test access requires explicit user approval")
        self.root = Path(directory)
        self.meta = json.loads((self.root / "metadata.json").read_text())
        if self.meta["split"] != split:
            raise ValueError("Requested split does not match prepared metadata")
        self.table = feather.read_table(self.root / f"{split}.arrow", memory_map=True)

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        return self.table.slice(index, 1).to_pylist()[0]


def pad_sequences(sequences, pad_id, device):
    ids = torch.full((len(sequences), max(map(len, sequences))), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, sequence in enumerate(sequences):
        ids[i, :len(sequence)] = torch.tensor(sequence, device=device)
        mask[i, :len(sequence)] = True
    return ids, mask


def targets_for(rows, codes, device):
    positions = {code.key: i for i, code in enumerate(codes)}
    target = torch.zeros((len(rows), len(codes)), device=device)
    for i, row in enumerate(rows):
        selected = [positions[key] for key in row["labels"] if key in positions]
        target[i, selected] = 1
    return target

