import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from .cms_data import table5_rows
from .codebook import load_codebook
from .drg_alignment import normalized_definition
from .transfer_data import save_json


def public_drg_aliases(semantic, table5):
    records = json.loads((semantic / "edition_aliases.json").read_text())
    by_edition = {key: row["key"] for row in records for key in row["edition_keys"]}
    aliases = defaultdict(set)
    for row in records:
        aliases[row["code"], row["normalized_definition"]].add(row["key"])
    for item in json.loads((table5 / "origin.json").read_text()):
        version = item["year"] - 1983
        for row in table5_rows(table5 / item["file"]):
            key = by_edition[f"ms_drg_v{version}:{row['code']}"]
            aliases[row["code"], normalized_definition(row["description"])].add(key)
    return aliases


def match_drg_title(code, title, aliases):
    exact = aliases.get((code, title))
    if exact:
        return sorted(exact), "exact_public_title"
    matched = set().union(*(keys for (number, text), keys in aliases.items()
                            if number == code and text.startswith(title)))
    return sorted(matched), "public_title_prefix" if matched else "unmapped"


def relabel_split(path, output, codes, aliases):
    stats = Counter()
    with path.open() as source, output.open("x") as destination:
        for line in source:
            row = json.loads(line)
            label = row["labels"][0]
            if label in codes:
                code, title = codes[label].code, normalized_definition(codes[label].description)
            else:
                _, code, title = label.split(":", 2)
            matches, method = match_drg_title(code, title, aliases)
            row["labels"] = matches or [label]
            row["label_groups"] = [row["labels"]]
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats.update(rows=1, multiple_compatible_labels=int(len(matches) > 1))
            stats[method] += 1
    return dict(stats)


def prepare_drg_aliases(args):
    root, semantic, table5, output = map(Path, (args.input, args.semantic, args.table5, args.output))
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata["role"] != "target_evaluation_only" or set(metadata["splits"]) - {"train", "dev"}:
        raise PermissionError("DRG title alignment accepts non-test frozen targets only")
    codes = {code.key: code for code in load_codebook(semantic / "codebook.json")}
    aliases = public_drg_aliases(semantic, table5)
    output.mkdir(parents=True, exist_ok=False)
    stats = {split: relabel_split(root / f"{split}.jsonl", output / f"{split}.jsonl", codes, aliases)
             for split in metadata["splits"]}
    shutil.copyfile(semantic / "codebook.json", output / "codebook.json")
    save_json(output / "public_aliases.json", [
        {"code": code, "title": title, "semantic_labels": sorted(keys)} for (code, title), keys in sorted(aliases.items())])
    save_json(output / "metadata.json", {
        **metadata, "input": str(root), "splits": stats, "output_mode": "single_label_alternative_groups",
        "matching": "same code and exact public long/short title; otherwise exact title prefix",
        "ambiguous_edition_policy": "retain all compatible public semantic labels in one acceptable-label group",
        "unknown_label_policy": "retain as an out-of-catalog false negative",
        "public_title_sources": [str(semantic), str(table5)],
    })
    print(json.dumps(stats), flush=True)
