import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .codebook import Code, load_codebook, write_codebook
from .transfer_data import save_json


def normalized_definition(text):
    return re.sub(r"\s+", " ", text.upper().strip())


def cms_semantic_catalog(root):
    identities = defaultdict(list)
    for version in range(25, 38):
        for code in load_codebook(root / f"ms-drg-v{version}.json"):
            identities[code.code, normalized_definition(code.description)].append((version, code))
    codes, aliases = [], []
    for (number, description), members in sorted(identities.items()):
        key = f"ms_drg_semantic:{number}:v{members[0][0]}"
        codes.append(Code(key, "ms_drg_semantic", number, description))
        aliases.append({"key": key, "code": number, "normalized_definition": description,
                        "editions": [version for version, _ in members],
                        "edition_keys": [code.key for _, code in members]})
    return codes, aliases


def source_alignment(root, aliases):
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata["role"] != "source_linkage_audit_only":
        raise ValueError("DRG alignment development uses source-linked annotations only")
    lookup = {(item["code"], item["normalized_definition"]): item for item in aliases}
    observed, unmatched, summaries = Counter(), Counter(), {}
    for split in ("train", "validation"):
        counts = Counter()
        with (root / f"{split}-HCFA.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                identity = row["drg_code"], normalized_definition(row["description"])
                observed[identity] += 1
                counts.update(rows=1, matched=int(identity in lookup))
                if identity not in lookup:
                    unmatched[identity] += 1
        summaries[split] = dict(counts)
    matches = [{"code": code, "definition": description, "source_rows": count,
                "semantic_key": lookup[code, description]["key"],
                "compatible_editions": lookup[code, description]["editions"]}
               for (code, description), count in sorted(observed.items()) if (code, description) in lookup]
    missing = [{"code": code, "definition": description, "source_rows": count}
               for (code, description), count in sorted(unmatched.items())]
    return matches, missing, summaries


def prepare_drg_alignment(args):
    codes, aliases = cms_semantic_catalog(Path(args.cms))
    matches, missing, summaries = source_alignment(Path(args.source), aliases)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    write_codebook(codes, output / "codebook.json")
    save_json(output / "edition_aliases.json", aliases)
    save_json(output / "source_matches.json", matches)
    save_json(output / "source_unmatched.json", missing)
    result = {"role": "target_public_codebook_with_source_alignment_audit", "drg_type": "HCFA/MS-DRG",
              "candidate_labels": len(codes), "public_editions": list(range(25, 38)),
              "source_observed_definitions": len(matches) + len(missing), "unmatched_definitions": len(missing),
              "source_linkage": summaries, "matching": "code plus case/whitespace-normalized exact definition",
              "candidate_source": str(args.cms), "source_audit": str(args.source),
              "model_fit_allowed": False, "target_cohort_prepared": False,
              "edition_policy": "retain all compatible editions; no annual edition inferred from shifted dates",
              "semantics": "hospital-recorded DRG label prediction; not replication of a billing grouper"}
    save_json(output / "metadata.json", result)
    print(json.dumps(result), flush=True)
