import json
import random
from collections import Counter
from pathlib import Path

import pyarrow.feather as feather
import torch

from .codebook import Code, load_codebook, write_codebook
from .transfer_data import save_json


def family_partitions(families, seed):
    families = sorted(set(families))
    random.Random(seed).shuffle(families)
    first, second = int(0.6 * len(families)), int(0.8 * len(families))
    return {family: ("train" if i < first else "selection" if i < second else "audit")
            for i, family in enumerate(families)}


def ccs_schemas(groups, max_members):
    eligible = [group for group in groups if 2 <= len(group["members"]) <= max_members]
    partitions = family_partitions((group["key"].split(".", 1)[0] for group in eligible), 42)
    return [{**group, "family": group["key"].split(".", 1)[0],
             "partition": partitions[group["key"].split(".", 1)[0]]} for group in eligible]


def source_pair_counts(path):
    meta = json.loads((path / "metadata.json").read_text())
    if meta["split"] != "train" or meta["icd_version"] != 9:
        raise ValueError("AND schema discovery requires source ICD-9 training annotations")
    table = feather.read_table(path / "train.arrow", columns=["labels"], memory_map=True)
    counts = Counter()
    for labels in table.column("labels").to_pylist():
        diagnoses = sorted(key for key in labels if key.startswith("icd9_diag:"))
        procedures = sorted(key for key in labels if key.startswith("icd9_proc:"))
        counts.update((diagnosis, procedure) for diagnosis in diagnoses for procedure in procedures)
    return counts


def and_schemas(counts, codes, paths, min_support, limit):
    descriptions = {code.key: code.description for code in codes}
    pairs = sorted((pair for pair, count in counts.items() if count >= min_support),
                   key=lambda pair: (-counts[pair], pair))[:limit]
    families = {pair: "AND:" + "&".join(paths[key][0]["key"] for key in pair) for pair in pairs}
    partitions = family_partitions(families.values(), 42)
    return [{"key": "source_and:" + "&".join(pair), "operator": "AND", "members": list(pair),
             "description": "Both diagnoses and procedures are documented: " + " AND ".join(descriptions[k] for k in pair),
             "family": families[pair], "partition": partitions[families[pair]], "source_support": counts[pair]}
            for pair in pairs]


def schema_codes(groups):
    return [Code(group["key"], group["key"].split(":")[0] if group["operator"] == "OR" else "source_schema_term",
                 group["key"], group["description"]) for group in groups]


def schema_targets(rows, groups, device):
    observed = [set(row["labels"]) for row in rows]
    values = [[any(key in labels for key in group["members"]) if group["operator"] == "OR"
               else all(key in labels for key in group["members"]) for group in groups] for labels in observed]
    return torch.tensor(values, dtype=torch.float32, device=device)


def sample_schemas(rows, groups, count, rng):
    targets = schema_targets(rows, groups, "cpu")
    positive = (targets.sum(0) > 0).tolist()
    included = [i for i, value in enumerate(positive) if value]
    absent = [i for i, value in enumerate(positive) if not value]
    selected = rng.sample(included, min(len(included), count // 2))
    selected += rng.sample(absent, min(len(absent), count - len(selected)))
    return [groups[i] for i in selected]


def prepare_source_schemas(args):
    root = Path(args.ccs)
    codes = load_codebook(root / "icd9-codebook.json")
    paths = json.loads((root / "paths.json").read_text())
    groups = ccs_schemas(json.loads((root / "groups.json").read_text()), args.max_members)
    counts = source_pair_counts(Path(args.train_data))
    groups += and_schemas(counts, codes, paths, args.min_support, args.and_limit)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    write_codebook(codes + schema_codes(groups), output / "codebook.json")
    save_json(output / "groups.json", groups)
    save_json(output / "metadata.json", {
        "role": "source_structural_training", "source": str(args.train_data), "split_seed": 42,
        "family_split": [0.6, 0.2, 0.2], "max_or_members": args.max_members,
        "and_min_support": args.min_support, "and_limit": args.and_limit,
        "schema_counts": {f"{operator}-{partition}": sum(g["operator"] == operator and g["partition"] == partition for g in groups)
                          for operator in ("OR", "AND") for partition in ("train", "selection", "audit")},
        "source_annotations_used": "ICD-9 train only", "candidate_codes": len(codes),
        "and_semantics": "both codes annotated; absent annotations do not establish clinical absence",
    })
