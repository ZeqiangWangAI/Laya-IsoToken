"""Prepare the four official gold-labeled locked test splits after explicit user approval.

Outputs under data/test-locked-20261121/ (non-destructive; never modifies train/val/dev):
  1. source-icd9/test/              Official Edin MIMIC-IV ICD-9 test split (0 patient overlap with train/val)
  2. icd10-patient-disjoint/        Official Edin MIMIC-IV ICD-10 test split excluding all source ICD-9 patients
     + cascade_300.jsonl            Deterministic first 300 sorted patient-disjoint ICD-10 test notes (raw text)
  3. drg-patient-disjoint/          Official MIMIC-IV HCFA MS-DRG test split (same patient-disjoint admissions)
  4. codiesp/                       Official CLEF 2020 CodiEsp gold test split (Zenodo 3837305, 250 Spanish notes)
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import httpx
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather

from icd_transfer.acquire import EDIN_REVISION
from icd_transfer.codebook import load_codebook, normalize_code, write_codebook
from icd_transfer.data import SCHEMA, chunk_tokens, load_tokenizer
from icd_transfer.drg_aliases import match_drg_title, public_drg_aliases
from icd_transfer.drg_alignment import normalized_definition
from icd_transfer.raw_source import source_codes
from icd_transfer.transfer_data import clinical_text
from icd_transfer.transfer_tokens import TRANSFER_SCHEMA

OUT_ROOT = ROOT / "data/test-locked-20261121"
MODEL_DIR = ROOT / "artifacts/laya-multilingual"
RAW_ROOT = ROOT / "data/raw"
CHUNK_SIZE = 256
MAX_CHUNKS = 32
MAX_WORDS = 4000


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _source_excluded_subjects() -> set[str]:
    src = ROOT / "data/benchmarks/mimiciv-2.2/clinical-multilingual/source"
    out: set[str] = set()
    for sp in ("train", "validation"):
        out.update(json.loads((src / sp / "subjects.json").read_text()))
    return out


def _write_arrow_batched(records: list[dict], out_arrow: Path, schema: pa.Schema, tokenizer) -> Counter:
    stats = Counter()
    out_arrow.parent.mkdir(parents=True, exist_ok=True)
    with pa.OSFile(str(out_arrow), "wb") as sink:
        writer = pa.ipc.new_file(sink, schema)
        for start in range(0, len(records), 128):
            batch = records[start:start + 128]
            seqs = tokenizer([r["text"] for r in batch], add_special_tokens=False)["input_ids"]
            out_rows = []
            for row, ids in zip(batch, seqs, strict=True):
                if not ids:
                    raise ValueError(f"Empty clinical text after tokenization: {row['id']}")
                chunks, discarded = chunk_tokens(ids, tokenizer, CHUNK_SIZE, MAX_CHUNKS)
                rec = {
                    "id": row["id"],
                    "subject_id": row["subject_id"],
                    "chunks": chunks,
                    "labels": row["labels"],
                }
                if "label_groups" in schema.names:
                    rec["label_groups"] = row.get("label_groups", [])
                out_rows.append(rec)
                stats.update(
                    rows=1,
                    tokens=len(ids),
                    chunks=len(chunks),
                    truncated_rows=int(discarded > 0),
                    discarded_tokens=discarded,
                    unknown_tokens=ids.count(tokenizer.unk_token_id),
                )
            writer.write_table(pa.Table.from_pylist(out_rows, schema=schema))
        writer.close()
    return stats


def prepare_mimic_tests(tokenizer, origin: dict, excluded_subjects: set[str]) -> dict:
    # 1. ICD-9 test IDs from local Edin split feather
    icd9_table = feather.read_table(ROOT / "data/references/duallaat/data/splits/mimiciv_icd9_split.feather")
    icd9_test_ids = {
        str(r["_id"])
        for r in icd9_table.filter(pc.equal(icd9_table["split"], "test")).select(["_id"]).to_pylist()
    }
    print(f"[ICD-9] Published Edin test admissions: {len(icd9_test_ids)}", flush=True)

    # 2. ICD-10 test IDs from pinned Edin revision
    icd10_url = (
        "https://raw.githubusercontent.com/JoakimEdin/medical-coding-reproducibility/"
        f"{EDIN_REVISION}/files/data/mimiciv_icd10/mimiciv_icd10_split.feather"
    )
    resp = httpx.get(icd10_url, follow_redirects=True, timeout=60)
    resp.raise_for_status()
    icd10_table = feather.read_table(pa.BufferReader(resp.content))
    icd10_test_ids = {
        str(r["_id"])
        for r in icd10_table.filter(pc.equal(icd10_table["split"], "test")).select(["_id"]).to_pylist()
    }
    print(f"[ICD-10] Published Edin test admissions: {len(icd10_test_ids)}", flush=True)

    # Codebooks (exact order from train/dev)
    icd9_src_dir = ROOT / "data/benchmarks/mimiciv-2.2/clinical-multilingual/source/train"
    icd9_codes = load_codebook(icd9_src_dir / "codebook.json")
    icd9_catalog = {c.key for c in icd9_codes}

    icd10_dev_dir = ROOT / "data/transfer/mimiciv-icd10-drg/tokens-laya-multilingual/icd10-patient-disjoint"
    icd10_codes = load_codebook(icd10_dev_dir / "codebook.json")
    icd10_catalog = {c.key for c in icd10_codes}

    drg_semantic = ROOT / "data/transfer/drg/semantic-alignment"
    drg_table5 = ROOT / "data/transfer/drg/cms-table5-late"
    drg_codes = load_codebook(drg_semantic / "codebook.json")
    drg_code_map = {c.key: c for c in drg_codes}
    drg_aliases = public_drg_aliases(drg_semantic, drg_table5)

    # Load hospital gold tables once
    hosp = RAW_ROOT / "mimiciv/2.2/hosp"
    print("[MIMIC] Loading diagnoses/procedures/DRG gold tables...", flush=True)
    diag9 = source_codes(hosp / "diagnoses_icd.csv.gz", icd9_test_ids, 9)
    proc9 = source_codes(hosp / "procedures_icd.csv.gz", icd9_test_ids, 9)
    diag10 = source_codes(hosp / "diagnoses_icd.csv.gz", icd10_test_ids, 10)
    proc10 = source_codes(hosp / "procedures_icd.csv.gz", icd10_test_ids, 10)

    drg_records: dict[str, dict[str, str]] = {}
    with gzip.open(hosp / "drgcodes.csv.gz", "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            hid = row["hadm_id"]
            if hid not in icd10_test_ids or row["drg_type"] != "HCFA":
                continue
            if hid in drg_records:
                raise ValueError(f"Multiple HCFA DRGs for {hid}")
            drg_records[hid] = {
                "subject_id": row["subject_id"],
                "drg_code": row["drg_code"],
                "title": normalized_definition(row["description"]),
            }

    # Stream discharge.csv.gz once
    wanted = icd9_test_ids | icd10_test_ids
    csv.field_size_limit(sys.maxsize)
    icd9_rows, icd10_pd_rows, drg_pd_rows = [], [], []
    icd9_subjects, icd10_pd_subjects, drg_pd_subjects = set(), set(), set()
    icd9_counts = Counter()
    icd10_overlap_excluded = 0
    drg_match_stats = Counter()
    seen_ids = set()

    print("[MIMIC] Streaming discharge.csv.gz once for locked test notes...", flush=True)
    with gzip.open(RAW_ROOT / "mimic-iv-note/2.2/note/discharge.csv.gz", "rt", newline="") as stream:
        for raw in csv.DictReader(stream):
            hid = raw["hadm_id"]
            if hid not in wanted:
                continue
            if hid in seen_ids:
                raise ValueError(f"Duplicate discharge note for {hid}")
            seen_ids.add(hid)
            subj = raw["subject_id"]
            words = clinical_text(raw["text"]).split()
            text = " ".join(words[:MAX_WORDS])

            if hid in icd9_test_ids:
                if subj in excluded_subjects:
                    raise ValueError(f"ICD-9 test patient {subj} overlaps train/validation!")
                lbls = sorted(
                    ({f"icd9_diag:{normalize_code(c)}" for c in diag9[hid]}
                     | {f"icd9_proc:{normalize_code(c)}" for c in proc9[hid]})
                    & icd9_catalog
                )
                if not lbls:
                    raise ValueError(f"ICD-9 test admission {hid} has no catalog labels")
                icd9_rows.append({"id": hid, "subject_id": subj, "text": text, "labels": lbls})
                icd9_subjects.add(subj)
                icd9_counts.update(lbls)

            if hid in icd10_test_ids:
                if subj in excluded_subjects:
                    icd10_overlap_excluded += 1
                    continue
                lbls10 = sorted(
                    ({f"icd10_diag:{normalize_code(c)}" for c in diag10[hid]}
                     | {f"icd10_proc:{normalize_code(c)}" for c in proc10[hid]})
                    & icd10_catalog
                )
                if not lbls10:
                    raise ValueError(f"ICD-10 test admission {hid} has no catalog labels")
                icd10_pd_rows.append({"id": hid, "subject_id": subj, "text": text, "labels": lbls10})
                icd10_pd_subjects.add(subj)

                if hid in drg_records:
                    rec = drg_records[hid]
                    if rec["subject_id"] != subj:
                        raise ValueError(f"DRG subject mismatch on {hid}")
                    matches, method = match_drg_title(rec["drg_code"], rec["title"], drg_aliases)
                    drg_lbls = matches or [f"unmapped_drg:{rec['drg_code']}:{rec['title']}"]
                    drg_pd_rows.append({
                        "id": hid,
                        "subject_id": subj,
                        "text": text,
                        "labels": drg_lbls,
                        "label_groups": [drg_lbls],
                    })
                    drg_pd_subjects.add(subj)
                    drg_match_stats[method] += 1
                    drg_match_stats["multiple_compatible_labels"] += int(len(matches) > 1)

    missing_icd9 = icd9_test_ids - seen_ids
    missing_icd10 = icd10_test_ids - seen_ids
    print(f"[MIMIC] Found ICD-9 test={len(icd9_rows)} (missing={len(missing_icd9)}), "
          f"ICD-10 patient-disjoint={len(icd10_pd_rows)} (excluded_overlap={icd10_overlap_excluded}, missing={len(missing_icd10)}), "
          f"DRG patient-disjoint={len(drg_pd_rows)}", flush=True)

    # Sort deterministically by admission ID
    icd9_rows.sort(key=lambda r: int(r["id"]))
    icd10_pd_rows.sort(key=lambda r: int(r["id"]))
    drg_pd_rows.sort(key=lambda r: int(r["id"]))

    # Write Source ICD-9 test
    src_out = OUT_ROOT / "source-icd9/test"
    src_stats = _write_arrow_batched(icd9_rows, src_out / "test.arrow", SCHEMA, tokenizer)
    write_codebook(icd9_codes, src_out / "codebook.json")
    _write_json(src_out / "subjects.json", sorted(icd9_subjects))
    _write_json(src_out / "metadata.json", {
        "split": "test",
        "icd_version": 9,
        "dataset_version": "mimiciv-2.2-edin-full-clinical-multilingual",
        "role": "locked_test_evaluation_only",
        "rows": len(icd9_rows),
        "patients": len(icd9_subjects),
        "patient_overlap_with_train_val": 0,
        "missing_raw_notes": sorted(missing_icd9),
        "label_counts": dict(icd9_counts),
        "tokenization": dict(src_stats),
        "chunk_size": CHUNK_SIZE,
        "max_chunks": MAX_CHUNKS,
        "max_words": MAX_WORDS,
        "pad_id": tokenizer.pad_token_id,
        "tokenizer": origin,
    })

    # Write ICD-10 patient-disjoint test
    icd10_out = OUT_ROOT / "icd10-patient-disjoint"
    icd10_stats = _write_arrow_batched(icd10_pd_rows, icd10_out / "test.arrow", TRANSFER_SCHEMA, tokenizer)
    write_codebook(icd10_codes, icd10_out / "codebook.json")
    _write_json(icd10_out / "metadata.json", {
        "dataset": "MIMIC-IV 2.2 icd10",
        "language": "en",
        "role": "target_evaluation_only",
        "model_fit_allowed": False,
        "threshold_selection_allowed": False,
        "translation": False,
        "output_mode": "multilabel",
        "candidate_codes": len(icd10_codes),
        "patient_id_namespace": "mimiciv",
        "source_patients_excluded": True,
        "excluded_overlap_rows": icd10_overlap_excluded,
        "missing_raw_notes": sorted(missing_icd10),
        "splits": {"test": {"rows": len(icd10_pd_rows), "patients": len(icd10_pd_subjects), **dict(icd10_stats)}},
        "chunk_size": CHUNK_SIZE,
        "max_chunks": MAX_CHUNKS,
        "max_words": MAX_WORDS,
        "pad_id": tokenizer.pad_token_id,
        "tokenizer": origin,
        "public_split_url": icd10_url,
        "test_cases_processed": True,
    })
    # Also save deterministic first 300 sorted patient-disjoint ICD-10 test notes for System-1 -> System-2 LLM Cascade
    with (icd10_out / "cascade_300.jsonl").open("w", encoding="utf-8") as f:
        for row in icd10_pd_rows[:300]:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Write MS-DRG patient-disjoint test
    drg_out = OUT_ROOT / "drg-patient-disjoint"
    drg_stats = _write_arrow_batched(drg_pd_rows, drg_out / "test.arrow", TRANSFER_SCHEMA, tokenizer)
    shutil.copyfile(drg_semantic / "codebook.json", drg_out / "codebook.json")
    _write_json(drg_out / "metadata.json", {
        "dataset": "MIMIC-IV 2.2 drg",
        "language": "en",
        "role": "target_evaluation_only",
        "model_fit_allowed": False,
        "threshold_selection_allowed": False,
        "translation": False,
        "output_mode": "single_label_alternative_groups",
        "candidate_codes": len(drg_codes),
        "patient_id_namespace": "mimiciv",
        "source_patients_excluded": True,
        "splits": {"test": {"rows": len(drg_pd_rows), "patients": len(drg_pd_subjects), **dict(drg_match_stats)}},
        "tokenization": {"test": dict(drg_stats)},
        "chunk_size": CHUNK_SIZE,
        "max_chunks": MAX_CHUNKS,
        "max_words": MAX_WORDS,
        "pad_id": tokenizer.pad_token_id,
        "tokenizer": origin,
        "test_cases_processed": True,
    })

    return {
        "source_icd9_test": {"rows": len(icd9_rows), "patients": len(icd9_subjects), "missing": len(missing_icd9)},
        "icd10_patient_disjoint_test": {"rows": len(icd10_pd_rows), "patients": len(icd10_pd_subjects), "excluded_overlap": icd10_overlap_excluded, "missing": len(missing_icd10)},
        "drg_patient_disjoint_test": {"rows": len(drg_pd_rows), "patients": len(drg_pd_subjects), "match_stats": dict(drg_match_stats)},
    }


def prepare_codiesp_test(tokenizer, origin: dict) -> dict:
    # Download official Zenodo 3837305 gold standard CodiEsp corpus
    url = "https://zenodo.org/api/records/3837305/files/codiesp.zip/content"
    zip_path = OUT_ROOT / "raw_codiesp_zenodo_3837305.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        print(f"[CodiEsp] Downloading official gold test archive from {url}...", flush=True)
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            with zip_path.open("wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)

    dev_dir = ROOT / "data/transfer/codiesp/tokens-hierarchical-laya-multilingual"
    codes = load_codebook(dev_dir / "codebook.json")
    catalog = {c.key for c in codes}

    labels: dict[str, set[str]] = defaultdict(set)
    texts: dict[str, str] = {}
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        d_file = [n for n in names if n.endswith("/testD.tsv") or n == "testD.tsv"]
        p_file = [n for n in names if n.endswith("/testP.tsv") or n == "testP.tsv"]
        if not d_file or not p_file:
            raise RuntimeError(f"Could not find testD.tsv/testP.tsv in {zip_path}: sample={names[:20]}")
        for kind, member in (("diag", d_file[0]), ("proc", p_file[0])):
            raw = zf.read(member).decode("utf-8")
            for doc_id, code in csv.reader(io.StringIO(raw), delimiter="\t"):
                labels[doc_id].add(f"cie10es2018_{kind}:{normalize_code(code)}")
        # Only take text_files under final_dataset_v4_to_publish/test/text_files/ (Spanish, NOT text_files_en)
        for n in names:
            if "/test/text_files/" in n and n.endswith(".txt") and "text_files_en" not in n:
                doc_id = Path(n).stem
                if doc_id in labels:
                    texts[doc_id] = clinical_text(zf.read(n).decode("utf-8"))

    missing = set(labels) - set(texts)
    if missing:
        raise RuntimeError(f"Missing CodiEsp test text files for {len(missing)} annotated docs: {sorted(missing)[:5]}")
    rows = []
    unmatched = Counter()
    for doc_id in sorted(labels):
        t_lbls = sorted(labels[doc_id])
        unmatched.update(set(t_lbls) - catalog)
        rows.append({
            "id": doc_id,
            "subject_id": None,
            "text": texts[doc_id],
            "labels": t_lbls,
            "label_groups": [],
        })
    print(f"[CodiEsp] Prepared official gold test docs={len(rows)}, out_of_catalog={dict(unmatched)}", flush=True)

    cod_out = OUT_ROOT / "codiesp"
    cod_stats = _write_arrow_batched(rows, cod_out / "test.arrow", TRANSFER_SCHEMA, tokenizer)
    shutil.copyfile(dev_dir / "codebook.json", cod_out / "codebook.json")
    shutil.copyfile(dev_dir / "groups.json", cod_out / "groups.json")
    _write_json(cod_out / "metadata.json", {
        "dataset": "CodiEsp",
        "language": "es",
        "edition": "CIE-10-ES 2018",
        "role": "target_evaluation_only",
        "model_fit_allowed": False,
        "threshold_selection_allowed": False,
        "translation": False,
        "candidate_codes": len(codes),
        "procedure_families": len(json.loads((dev_dir / "groups.json").read_text())),
        "zenodo_record": 3837305,
        "splits": {"test": {"rows": len(rows), "out_of_catalog": dict(unmatched), **dict(cod_stats)}},
        "chunk_size": CHUNK_SIZE,
        "max_chunks": MAX_CHUNKS,
        "pad_id": tokenizer.pad_token_id,
        "tokenizer": origin,
        "test_cases_processed": True,
    })
    return {"codiesp_test": {"rows": len(rows), "out_of_catalog": dict(unmatched)}}


def main() -> None:
    if OUT_ROOT.exists():
        raise FileExistsError(f"{OUT_ROOT} already exists; refusing to overwrite locked test splits.")
    tokenizer = load_tokenizer(MODEL_DIR)
    origin = json.loads((MODEL_DIR / "origin.json").read_text())
    excluded = _source_excluded_subjects()
    print(f"Loaded {len(excluded)} excluded source ICD-9 subjects.", flush=True)
    summary = {}
    summary.update(prepare_mimic_tests(tokenizer, origin, excluded))
    summary.update(prepare_codiesp_test(tokenizer, origin))
    _write_json(OUT_ROOT / "manifest.json", summary)
    print("\n=== LOCKED TEST SPLITS READY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
