import csv
import gzip
import io
import json
import unicodedata
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

import pyarrow.feather as feather

from .codebook import Code, load_codebook, normalize_code, write_codebook


def clinical_text(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def ccs_rows(archive, member):
    text = archive.read(member).decode("utf-8-sig")
    rows = csv.reader(io.StringIO(text))
    next(rows)
    return [[cell.strip().strip("'").strip() for cell in row] for row in rows]


def prepare_ccs(args):
    codes = load_codebook(args.codebook)
    selected = {code.key for code in codes}
    paths, groups = {}, {}
    with zipfile.ZipFile(args.input) as archive:
        for kind, name in (("diag", "dx"), ("proc", "pr")):
            rows = ccs_rows(archive, f"ccs_multi_{name}_tool_2015.csv")
            add_ccs_groups(rows, kind, selected, paths, groups)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    write_codebook([Code(c.key, c.system, c.code, c.description,
                        tuple(group["description"] for group in paths.get(c.key, [])))
                    for c in codes], output / "icd9-codebook.json")
    save_json(output / "groups.json", list(groups.values()))
    save_json(output / "paths.json", paths)
    save_json(output / "metadata.json", {
        "source": "AHRQ Multi-Level CCS 2015", "role": "source_public_structure",
        "source_codebook": str(args.codebook), "source_codes": len(codes),
        "covered_codes": len(paths), "groups": len(groups),
        "unmapped_codes": sorted(selected - paths.keys()),
    })


def add_ccs_groups(rows, kind, selected, paths, groups):
    for row in rows:
        key = f"icd9_{kind}:{normalize_code(row[0])}"
        if key not in selected:
            continue
        path = []
        for category, description in zip(row[1::2], row[2::2]):
            if not category:
                continue
            group_key = f"ccs2015_{kind}:{category}"
            path.append({"key": group_key, "description": description})
            group = groups.setdefault(group_key, {"key": group_key, "description": description,
                                                  "operator": "OR", "members": []})
            group["members"].append(key)
        paths[key] = path


def source_admissions(source):
    result = {}
    for split in ("train", "validation"):
        path = Path(source) / split / f"{split}.arrow"
        table = feather.read_table(path, columns=["id", "subject_id"], memory_map=True)
        for row in table.to_pylist():
            if row["id"] in result:
                raise ValueError("Source admissions overlap")
            result[row["id"]] = (split, row["subject_id"])
    return result


def drg_links(path, admissions):
    seen = set()
    with gzip.open(path, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["hadm_id"] not in admissions:
                continue
            split, subject = admissions[row["hadm_id"]]
            if row["subject_id"] != subject:
                raise ValueError("DRG and clinical admission patient IDs disagree")
            identity = (row["hadm_id"], row["drg_type"])
            if identity in seen:
                raise ValueError("Multiple DRG assignments for one admission and DRG type")
            seen.add(identity)
            yield {"id": row["hadm_id"], "subject_id": subject, "source_split": split,
                   "drg_type": row["drg_type"], "drg_code": row["drg_code"],
                   "description": row["description"], "edition": None,
                   "drg_severity": row["drg_severity"], "drg_mortality": row["drg_mortality"]}


def prepare_drg(args):
    admissions = source_admissions(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    counts, subjects, definitions = Counter(), defaultdict(set), defaultdict(set)
    with ExitStack() as stack:
        streams = {(split, kind): stack.enter_context((output / f"{split}-{kind}.jsonl").open("x"))
                   for split in ("train", "validation") for kind in ("HCFA", "APR")}
        for row in drg_links(args.input, admissions):
            key = (row["source_split"], row["drg_type"])
            streams[key].write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[key] += 1
            subjects[key].add(row["subject_id"])
            definitions[row["drg_type"], row["drg_code"]].add(row["description"])
    save_json(output / "metadata.json", {
        "raw_version": "MIMIC-IV 2.2", "source": str(args.input),
        "role": "source_linkage_audit_only", "model_fit_allowed": False,
        "target_evaluation_cohort": False,
        "edition": "not recorded in MIMIC DRG table",
        "splits": {f"{s}-{k}": {"rows": counts[s, k], "patients": len(subjects[s, k])}
                   for s, k in streams},
        "codes_with_multiple_descriptions": [
            {"drg_type": kind, "drg_code": code, "descriptions": sorted(values)}
            for (kind, code), values in sorted(definitions.items()) if len(values) > 1],
    })


def codiesp_codebook(directory):
    codes, groups = [], []
    for kind, name in (("diag", "D"), ("proc", "P")):
        path = Path(directory) / f"codiesp-{name}_codes.tsv"
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream, delimiter="\t"))
        terms, families = codiesp_terms(rows, kind)
        codes.extend(terms)
        groups.extend(families)
    return codes, groups


def codiesp_terms(rows, kind):
    system = f"cie10es2018_{kind}"
    descriptions = {normalize_code(row[0]): row[1].strip() for row in rows}
    children, codes, groups = defaultdict(list), [], []
    for code in descriptions:
        if kind == "proc" and len(code) == 7:
            children[code[:4]].append(code)
    for code, description in descriptions.items():
        if not description:
            members = sorted(children[code])
            if not members or any(not descriptions[child] for child in members):
                raise ValueError(f"Missing official leaf definitions for CodiEsp {code}")
            description = "Any of: " + "; ".join(descriptions[child] for child in members)
            groups.append({"key": f"{system}:{code}", "operator": "OR",
                           "members": [f"{system}:{child}" for child in members]})
        codes.append(Code(f"{system}:{code}", system, code, description))
    return codes, groups


def codiesp_annotations(directory, split):
    labels = defaultdict(set)
    for kind, name in (("diag", "D"), ("proc", "P")):
        with (directory / split / f"{split}{name}.tsv").open(encoding="utf-8", newline="") as stream:
            for identifier, code in csv.reader(stream, delimiter="\t"):
                labels[identifier].add(f"cie10es2018_{kind}:{normalize_code(code)}")
    return labels


def convert_codiesp_split(root, split, output, catalog):
    labels = codiesp_annotations(root, split)
    paths = sorted((root / split / "text_files").glob("*.txt"))
    if set(labels) - {path.stem for path in paths}:
        raise ValueError("CodiEsp annotations reference missing documents")
    counts, unmatched, empty = Counter(), Counter(), 0
    with (output / f"{split}.jsonl").open("x", encoding="utf-8") as stream:
        for path in paths:
            targets = sorted(labels[path.stem])
            row = {"id": path.stem, "subject_id": None, "language": "es",
                   "task": "clinical_coding", "role": "target_evaluation_only",
                   "official_split": split, "text": clinical_text(path.read_text(encoding="utf-8")),
                   "labels": targets}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts.update(targets)
            unmatched.update(set(targets) - catalog)
            empty += not targets
    return {"rows": len(paths), "label_assignments": sum(counts.values()),
            "observed_labels": len(counts), "unlabeled_documents": empty,
            "out_of_catalog": dict(unmatched), "patient_ids": "not provided"}


def prepare_codiesp(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    codes, groups = codiesp_codebook(args.code_directory)
    write_codebook(codes, output / "codebook.json")
    save_json(output / "groups.json", groups)
    catalog = {code.key for code in codes}
    if len(catalog) != len(codes):
        raise ValueError("CodiEsp official code dictionary has duplicate normalized keys")
    stats = {split: convert_codiesp_split(Path(args.input), split, output, catalog)
             for split in ("train", "dev")}
    save_json(output / "metadata.json", {
        "dataset": "CodiEsp", "language": "es", "edition": "CIE-10-ES 2018",
        "role": "target_evaluation_only", "model_fit_allowed": False,
        "threshold_selection_allowed": False, "translation": False,
        "text_processing": "NFC and whitespace normalization",
        "codebook": "official CodiEsp D/P candidate dictionaries; Spanish definitions",
        "procedure_families": len(groups),
        "family_definition": "OR of all official seven-character descendants sharing the four-character prefix",
        "family_inference": "aggregate descendant label representations; do not truncate expanded OR definitions",
        "candidate_codes": len(codes), "splits": stats,
    })


def who_codebook(archive_path, label_path):
    with zipfile.ZipFile(archive_path) as archive:
        root = ET.fromstring(archive.read("icd102019en.xml"))
    classes = {row.attrib["code"]: row for row in root.findall("Class")}
    titles = {key: clinical_text("".join(row.find("Rubric[@kind='preferred']/Label").itertext()))
              for key, row in classes.items()}
    composite = ET.Element("Class", code="D50-D64")
    ET.SubElement(composite, "SuperClass", code="III")
    classes["D50-D64"] = composite
    titles["D50-D64"] = "Any of: " + "; ".join(titles[k] for k in ("D50-D53", "D55-D59", "D60-D64"))
    codes = []
    for key in Path(label_path).read_text().split():
        parent, ancestors = classes[key].find("SuperClass"), []
        while parent is not None:
            parent_key = parent.attrib["code"]
            ancestors.append(titles[parent_key])
            parent = classes[parent_key].find("SuperClass")
        system = "who_icd10_2019_diag"
        codes.append(Code(f"{system}:{key}", system, key, titles[key], tuple(reversed(ancestors))))
    return codes


def prepare_greek(args):
    root, output = Path(args.input), Path(args.output)
    codes = who_codebook(args.who_dictionary, root / "labelset.txt")
    catalog = {code.code: code.key for code in codes}
    output.mkdir(parents=True, exist_ok=False)
    write_codebook(codes, output / "codebook.json")
    stats, identifiers = Counter(), set()
    with (root / "trainELCardioCC2026.jsonl").open() as source, (output / "train.jsonl").open("x") as sink:
        for line in source:
            row = json.loads(line)
            identifier = str(row["patient_id"])
            if identifier in identifiers:
                raise ValueError("Duplicate ELCardioCC patient_id")
            identifiers.add(identifier)
            groups = row["document_level_annotations"]
            mapped = [[catalog[code] for code in group] for group in groups if group]
            record = {"id": identifier, "subject_id": f"elcardiocc2026:{identifier}",
                      "language": "el", "task": "clinical_coding", "role": "target_evaluation_only",
                      "official_split": "train", "text": clinical_text(row["text"]),
                      "label_groups": mapped}
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats.update(rows=1, groups=len(mapped), alternative_groups=sum(len(g) > 1 for g in groups),
                         empty_groups_skipped=sum(not g for g in groups),
                         mention_annotated=bool(row["mention_level_annotations"]))
    save_json(output / "metadata.json", {
        "dataset": "ELCardioCC 2026", "language": "el", "role": "target_evaluation_only",
        "model_fit_allowed": False, "threshold_selection_allowed": False, "translation": False,
        "edition": "official task labelset; dataset does not specify ICD annual edition",
        "definition_reference": "WHO ICD-10 2019 English official definitions",
        "definition_language": "en", "candidate_codes": len(codes),
        "composite_labels": {"D50-D64": ["D50-D53", "D55-D59", "D60-D64"]},
        "text_processing": "NFC and whitespace normalization; original spans retained in raw archive",
        "output_mode": "alternative_label_groups", "patient_id_namespace": "elcardiocc2026",
        "splits": {"train": dict(stats)},
    })


def cdn_answer(text):
    text = "".join(chr(ord(c) - 65248) if 65281 <= ord(c) <= 65374 else c for c in text)
    return text.replace("\u3000", " ").strip('"').split("##")


def cdn_codebook(path):
    from openpyxl import load_workbook
    workbook = load_workbook(path, read_only=True, data_only=True)
    terms = defaultdict(set)
    for row in workbook.active.values:
        terms[row[1]].add(str(row[0]))
    workbook.close()
    codes = [Code(f"chip_cdn_term:{term}", "chip_cdn_term", term, term) for term in sorted(terms)]
    return codes, {term: sorted(values) for term, values in terms.items()}


def convert_cdn_split(root, split, output, catalog):
    rows = json.loads((root / f"CHIP-CDN_{split}.json").read_text())
    counts, unmatched = Counter(), Counter()
    with (output / f"{split}.jsonl").open("x") as sink:
        for i, row in enumerate(rows):
            labels = sorted({f"chip_cdn_term:{term}" for term in cdn_answer(row["normalized_result"])})
            record = {"id": f"{split}:{i}", "subject_id": None, "language": "zh",
                      "task": "diagnosis_normalization", "role": "target_evaluation_only",
                      "official_split": split, "text": clinical_text(row["text"]), "labels": labels}
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts.update(labels)
            unmatched.update(set(labels) - catalog)
    return {"rows": len(rows), "label_assignments": sum(counts.values()),
            "observed_labels": len(counts), "out_of_catalog_labels": len(unmatched),
            "out_of_catalog_assignments": sum(unmatched.values()),
            "out_of_catalog": dict(unmatched), "patient_ids": "not provided"}


def prepare_cdn(args):
    root, output = Path(args.input), Path(args.output)
    codes, terms = cdn_codebook(root / "国际疾病分类 ICD-10北京临床版v601.xlsx")
    output.mkdir(parents=True, exist_ok=False)
    write_codebook(codes, output / "codebook.json")
    save_json(output / "term_to_icd.json", terms)
    catalog = {code.key for code in codes}
    stats = {split: convert_cdn_split(root, split, output, catalog) for split in ("train", "dev")}
    save_json(output / "metadata.json", {
        "dataset": "CHIP-CDN", "language": "zh", "edition": "ICD-10 Beijing clinical v601",
        "role": "target_evaluation_only", "model_fit_allowed": False,
        "threshold_selection_allowed": False, "translation": False,
        "candidate_codes": len(codes), "candidate_unit": "unique standard diagnosis term",
        "text_processing": "NFC and whitespace normalization",
        "label_processing": "official CBLUE fullwidth ASCII conversion, quote trim, ## splitting",
        "out_of_catalog_policy": "retain truth as false negatives; do not expand catalog from target labels",
        "splits": stats,
    })
