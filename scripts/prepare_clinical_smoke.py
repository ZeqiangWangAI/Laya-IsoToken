import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import pyarrow.feather as feather


def prepare(source, output):
    output.mkdir(parents=True, exist_ok=False)
    summaries = {}
    for split, limit in (("train", 32), ("validation", 8)):
        root = source / split
        meta = json.loads((root / "metadata.json").read_text())
        if meta["icd_version"] != 9 or meta["split"] != split:
            raise ValueError("Smoke preparation accepts source ICD-9 only")
        table = feather.read_table(root / f"{split}.arrow", memory_map=True).slice(0, limit)
        rows = table.to_pylist()
        counts = Counter(key for row in rows for key in row["labels"])
        subjects = sorted({row["subject_id"] for row in rows})
        directory = output / split
        directory.mkdir()
        feather.write_feather(table, directory / f"{split}.arrow")
        shutil.copyfile(root / "codebook.json", directory / "codebook.json")
        meta.update(rows=len(rows), label_counts=dict(counts), source_prepared=str(root),
                    dataset_version=meta["dataset_version"] + "-clinical-smoke",
                    chunks=sum(len(row["chunks"]) for row in rows),
                    tokens=sum(len(chunk) - 2 for row in rows for chunk in row["chunks"]),
                    role="source_smoke_only")
        for key in ("truncated_rows", "discarded_tokens", "word_truncated_rows", "discarded_words"):
            meta.pop(key, None)
        (directory / "metadata.json").write_text(json.dumps(meta, indent=2))
        (directory / "subjects.json").write_text(json.dumps(subjects))
        summaries[split] = {"rows": len(rows), "patients": len(subjects), "chunks": meta["chunks"]}
    (output / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
