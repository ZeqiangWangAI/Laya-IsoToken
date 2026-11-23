import csv
import gzip
import json
import re
import sys
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path


def clean_text(text):
    text = re.sub("[^A-Za-z0-9]+", " ", text.lower())
    text = re.sub(r"(\s\d+)+\s", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def source_codes(path, allowed_ids, version=9):
    codes = defaultdict(set)
    with gzip.open(path, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["hadm_id"] in allowed_ids and row["icd_version"] == str(version):
                codes[row["hadm_id"]].add(row["icd_code"])
    return codes


def extract_source(args):
    csv.field_size_limit(sys.maxsize)
    splits = json.loads(Path(args.split_ids).read_text())
    if set(splits) != {"train", "validation"}:
        raise ValueError("Source extraction accepts train and validation IDs only")
    memberships = {str(identifier): split for split, ids in splits.items() for identifier in ids}
    if len(memberships) != sum(map(len, splits.values())):
        raise ValueError("Source split IDs overlap")
    root = Path(args.raw)
    hospital = root / "mimiciv/2.2/hosp"
    diagnoses = source_codes(hospital / "diagnoses_icd.csv.gz", memberships)
    procedures = source_codes(hospital / "procedures_icd.csv.gz", memberships)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    seen = set()
    with ExitStack() as stack:
        streams = {split: stack.enter_context((output / f"{split}.jsonl").open("x")) for split in splits}
        notes = stack.enter_context(gzip.open(root / "mimic-iv-note/2.2/note/discharge.csv.gz", "rt", newline=""))
        for row in csv.DictReader(notes):
            identifier = row["hadm_id"]
            if identifier not in memberships:
                continue
            if identifier in seen:
                raise ValueError("Multiple discharge documents for one admission require explicit consolidation")
            seen.add(identifier)
            record = {"_id": identifier, "subject_id": row["subject_id"], "text": clean_text(row["text"]),
                      "icd9_diag": sorted(diagnoses[identifier]), "icd9_proc": sorted(procedures[identifier])}
            streams[memberships[identifier]].write(json.dumps(record) + "\n")
    if seen != set(memberships):
        raise ValueError(f"Raw data is missing {len(set(memberships) - seen)} published source admissions")
    (output / "metadata.json").write_text(json.dumps({
        "raw_version": "MIMIC-IV 2.2 and MIMIC-IV-Note 2.2", "split_source": str(args.split_ids),
        "text_cleaning": "Edin SIGIR 2023", "label_filter": "none; no test-derived frequency filtering",
        "examples": len(seen),
    }, indent=2))

