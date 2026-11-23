import csv
import gzip
import json
import shutil
import sys
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

from .acquire import EDIN_REVISION
from .codebook import load_codebook, normalize_code, write_codebook
from .drg_alignment import normalized_definition
from .raw_source import source_codes
from .transfer_data import clinical_text, save_json


def target_memberships():
    url = ("https://raw.githubusercontent.com/JoakimEdin/medical-coding-reproducibility/"
           f"{EDIN_REVISION}/files/data/mimiciv_icd10/mimiciv_icd10_split.feather")
    response = httpx.get(url, follow_redirects=True, timeout=60)
    response.raise_for_status()
    table = feather.read_table(pa.BufferReader(response.content))
    selected = table.filter(pc.is_in(table["split"], value_set=pa.array(["train", "val"])))
    mapping = {str(row["_id"]): {"train": "train", "val": "dev"}[row["split"]]
               for row in selected.select(["_id", "split"]).to_pylist()}
    if len(mapping) != len(selected) or set(mapping.values()) != {"train", "dev"}:
        raise ValueError("Published target train/validation admission IDs are incomplete or duplicated")
    return mapping, url


def target_drg(path, allowed, aliases):
    lookup = {(row["code"], row["normalized_definition"]): row["key"] for row in aliases}
    records = {}
    with gzip.open(path, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["hadm_id"] not in allowed or row["drg_type"] != "HCFA":
                continue
            if row["hadm_id"] in records:
                raise ValueError("Target admission has multiple HCFA assignments")
            identity = row["drg_code"], normalized_definition(row["description"])
            label = lookup[identity] if identity in lookup else "unmapped_drg:" + ":".join(identity)
            records[row["hadm_id"]] = {"subject_id": row["subject_id"], "label": label}
    return records


def write_target_note(raw, split, codes, diagnoses, procedures, drg, excluded, streams, stats):
    identifier = raw["hadm_id"]
    labels = {f"icd10_{kind}:{normalize_code(code)}" for kind, records in (("diag", diagnoses), ("proc", procedures))
              for code in records[identifier]}
    retained = sorted(labels & codes)
    if not retained:
        raise ValueError("Published target admission has no benchmark ICD-10 labels")
    words = clinical_text(raw["text"]).split()
    row = {"id": identifier, "subject_id": raw["subject_id"], "text": " ".join(words[:4000]), "labels": retained}
    streams["icd10", split].write(json.dumps(row, ensure_ascii=False) + "\n")
    overlap = raw["subject_id"] in excluded
    stats["icd10", split].update(rows=1, source_patient_overlap_rows=int(overlap),
                                  labels_outside_benchmark=len(labels - codes), word_truncated_rows=int(len(words) > 4000))
    if identifier not in drg or overlap:
        return
    assigned = drg[identifier]
    if assigned["subject_id"] != raw["subject_id"]:
        raise ValueError("DRG and note patient identifiers differ")
    row["labels"] = [assigned["label"]]
    streams["drg", split].write(json.dumps(row, ensure_ascii=False) + "\n")
    stats["drg", split].update(rows=1, unmapped_labels=int(assigned["label"].startswith("unmapped_drg:")))


def extract_targets(args, members, codes, diagnoses, procedures, drg, excluded, streams):
    seen, stats = set(), {(task, split): Counter() for task in ("icd10", "drg") for split in ("train", "dev")}
    csv.field_size_limit(sys.maxsize)
    path = Path(args.raw) / "mimic-iv-note/2.2/note/discharge.csv.gz"
    with gzip.open(path, "rt", newline="") as source:
        for row in csv.DictReader(source):
            identifier = row["hadm_id"]
            if identifier not in members:
                continue
            if identifier in seen:
                raise ValueError("Target admission has multiple discharge summaries")
            seen.add(identifier)
            write_target_note(row, members[identifier], codes, diagnoses, procedures, drg, excluded, streams, stats)
    if seen != members.keys():
        raise ValueError("Published target admissions are missing from the raw notes")
    return stats


def prepare_mimic_transfer(args):
    if not args.target_preparation_approved:
        raise PermissionError("The agreed protocol requires approval before preparing target MIMIC cases")
    members, split_url = target_memberships()
    unavailable = set(json.loads(Path(args.unavailable_notes).read_text()))
    if not unavailable <= members.keys():
        raise ValueError("Unavailable-note exclusions must belong to the published non-test cohort")
    published_counts = Counter(members.values())
    members = {key: split for key, split in members.items() if key not in unavailable}
    codes = load_codebook(args.codebook)
    if {code.system for code in codes} != {"icd10_diag", "icd10_proc"}:
        raise ValueError("MIMIC target preparation requires the ICD-10 benchmark codebook")
    source = Path(args.source)
    excluded = set().union(*(json.loads((source / split / "subjects.json").read_text()) for split in ("train", "validation")))
    hospital = Path(args.raw) / "mimiciv/2.2/hosp"
    diagnoses = source_codes(hospital / "diagnoses_icd.csv.gz", members, 10)
    procedures = source_codes(hospital / "procedures_icd.csv.gz", members, 10)
    alignment = Path(args.drg_alignment)
    aliases = json.loads((alignment / "edition_aliases.json").read_text())
    drg = target_drg(hospital / "drgcodes.csv.gz", members, aliases)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    with ExitStack() as stack:
        streams = {}
        for task in ("icd10", "drg"):
            (output / task).mkdir()
            for split in ("train", "dev"):
                streams[task, split] = stack.enter_context((output / task / f"{split}.jsonl").open("x"))
        stats = extract_targets(args, members, {code.key for code in codes}, diagnoses, procedures, drg, excluded, streams)
    books = {"icd10": codes, "drg": load_codebook(alignment / "codebook.json")}
    for task, book in books.items():
        write_codebook(book, output / task / "codebook.json")
        save_json(output / task / "metadata.json", {
            "dataset": f"MIMIC-IV 2.2 {task}", "language": "en", "role": "target_evaluation_only",
            "model_fit_allowed": False, "threshold_selection_allowed": False, "translation": False,
            "splits": {split: dict(stats[task, split]) for split in ("train", "dev")},
            "candidate_codes": len(book), "patient_id_namespace": "mimiciv", "max_words": 4000,
            "source_patients_excluded": task == "drg", "source_subjects": len(excluded),
            "public_split_url": split_url, "output_mode": "single_label" if task == "drg" else "multilabel",
            "text_processing": "NFC; whitespace; Unicode punctuation and numbers retained",
            "test_cases_processed": False,
            "published_non_test_counts": dict(published_counts),
            "unavailable_note_exclusions": len(unavailable), "exclusion_manifest": str(args.unavailable_notes),
        })
    save_json(output / "origin.json", {"split_url": split_url, "source_exclusions": str(source),
                                       "raw": str(args.raw), "drg_alignment": str(alignment)})
    save_json(output / "memberships.json", members)
    print(json.dumps({f"{task}-{split}": dict(values) for (task, split), values in stats.items()}), flush=True)


def prepare_disjoint_target(args):
    root, source, output = map(Path, (args.input, args.source, args.output))
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata["role"] != "target_evaluation_only" or set(metadata["splits"]) - {"train", "dev"}:
        raise PermissionError("Patient filtering accepts non-test transfer targets only")
    excluded = set().union(*(json.loads((source / split / "subjects.json").read_text()) for split in ("train", "validation")))
    output.mkdir(parents=True, exist_ok=False)
    stats, patients = {}, {}
    for split in metadata["splits"]:
        table = feather.read_table(root / f"{split}.arrow", memory_map=True)
        keep = pc.invert(pc.is_in(table["subject_id"], value_set=pa.array(sorted(excluded))))
        selected = table.filter(keep)
        feather.write_feather(selected, output / f"{split}.arrow")
        chunks = pc.list_flatten(selected["chunks"])
        stats[split] = {"rows": len(selected), "excluded_rows": len(table) - len(selected),
                        "chunks": len(chunks), "retained_content_tokens": pc.sum(pc.list_value_length(chunks)).as_py() - 2 * len(chunks)}
        patients[split] = sorted(set(selected["subject_id"].to_pylist()))
    shutil.copyfile(root / "codebook.json", output / "codebook.json")
    save_json(output / "subjects.json", patients)
    save_json(output / "metadata.json", {
        **metadata, "input": str(root), "source_exclusions": str(source), "source_patients_excluded": True,
        "splits": stats, "tokenization": stats, "parent_tokenization": metadata["tokenization"],
    })
    print(json.dumps(stats), flush=True)
