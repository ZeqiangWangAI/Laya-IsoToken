import json
import shutil
from collections import Counter
from pathlib import Path

import pyarrow as pa

from .data import SCHEMA, chunk_tokens, load_tokenizer
from .transfer_data import save_json


TRANSFER_SCHEMA = SCHEMA.append(pa.field("label_groups", pa.list_(pa.list_(pa.string()))))


def write_token_batch(rows, writer, tokenizer, chunk_size, max_chunks, stats):
    sequences = tokenizer([row["text"] for row in rows], add_special_tokens=False)["input_ids"]
    records = []
    for row, ids in zip(rows, sequences, strict=True):
        if not ids:
            raise ValueError("Empty clinical text after tokenization")
        chunks, discarded = chunk_tokens(ids, tokenizer, chunk_size, max_chunks)
        records.append({"id": row["id"], "subject_id": row["subject_id"], "chunks": chunks,
                        "labels": row.get("labels", []), "label_groups": row.get("label_groups", [])})
        stats.update(rows=1, tokens=len(ids), chunks=len(chunks),
                     truncated_rows=int(discarded > 0), discarded_tokens=discarded,
                     unknown_tokens=ids.count(tokenizer.unk_token_id))
    writer.write_table(pa.Table.from_pylist(records, schema=TRANSFER_SCHEMA))


def tokenize_split(path, output, tokenizer, chunk_size, max_chunks):
    stats, batch = Counter(), []
    with path.open() as source, pa.OSFile(str(output), "wb") as sink:
        writer = pa.ipc.new_file(sink, TRANSFER_SCHEMA)
        for line in source:
            batch.append(json.loads(line))
            if len(batch) == 128:
                write_token_batch(batch, writer, tokenizer, chunk_size, max_chunks, stats)
                batch.clear()
                if stats["rows"] % 12800 == 0:
                    print(json.dumps({"split": path.stem, "rows": stats["rows"]}), flush=True)
        if batch:
            write_token_batch(batch, writer, tokenizer, chunk_size, max_chunks, stats)
        writer.close()
    return dict(stats)


def tokenize_transfer(args):
    root, output = Path(args.input), Path(args.output)
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata["role"] != "target_evaluation_only" or metadata["model_fit_allowed"]:
        raise ValueError("This command prepares frozen transfer targets only")
    if set(metadata["splits"]) - {"train", "dev"}:
        raise PermissionError("Transfer development preparation accepts official train/dev only")
    tokenizer = load_tokenizer(args.model_dir)
    output.mkdir(parents=True, exist_ok=False)
    stats = {split: tokenize_split(root / f"{split}.jsonl", output / f"{split}.arrow", tokenizer,
                                  args.chunk_size, args.max_chunks) for split in metadata["splits"]}
    shutil.copyfile(root / "codebook.json", output / "codebook.json")
    if (root / "groups.json").exists():
        shutil.copyfile(root / "groups.json", output / "groups.json")
    save_json(output / "metadata.json", {
        **metadata, "input": str(root), "tokenization": stats, "chunk_size": args.chunk_size,
        "max_chunks": args.max_chunks, "pad_id": tokenizer.pad_token_id,
        "tokenizer": json.loads((Path(args.model_dir) / "origin.json").read_text()),
    })
