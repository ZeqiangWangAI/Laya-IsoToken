import getpass
import json
from pathlib import Path

import httpx
import pyarrow.compute as pc
import pyarrow.feather as feather
from huggingface_hub import snapshot_download


LAYA_REPO = "convaiinnovations/laya"
LAYA_REVISION = "e333cb7c823cd1787276228260f56d02fbf5180b"
EDIN_REVISION = "70bda64dce532f2334f8b97dcd50c02f66dd232f"
SOURCE_FILES = [
    ("mimiciv/2.2", "hosp/d_icd_diagnoses.csv.gz"),
    ("mimiciv/2.2", "hosp/d_icd_procedures.csv.gz"),
    ("mimiciv/2.2", "hosp/diagnoses_icd.csv.gz"),
    ("mimiciv/2.2", "hosp/procedures_icd.csv.gz"),
    ("mimic-iv-note/2.2", "note/discharge.csv.gz"),
]


def fetch_laya(args):
    output = Path(args.output)
    origin = {"repo": LAYA_REPO, "revision": LAYA_REVISION, "synthetic": False}
    if (output / "origin.json").exists() and json.loads((output / "origin.json").read_text()) != origin:
        raise ValueError("Existing model directory contains a different revision")
    patterns = ["encoder/config.json", "rl_agent_config.json", "tokenizer/*"]
    if args.weights:
        patterns.append("model.safetensors")
    snapshot_download(LAYA_REPO, revision=LAYA_REVISION, local_dir=output, allow_patterns=patterns)
    (output / "origin.json").write_text(json.dumps(origin, indent=2))
    print(output)


def fetch_source_splits(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    url = ("https://raw.githubusercontent.com/JoakimEdin/medical-coding-reproducibility/"
           f"{EDIN_REVISION}/files/data/mimiciv_icd9/mimiciv_icd9_split.feather")
    response = httpx.get(url, follow_redirects=True, timeout=60)
    response.raise_for_status()
    import pyarrow as pa
    table = feather.read_table(pa.BufferReader(response.content))
    selected = table.filter(pc.is_in(table["split"], value_set=pa.array(["train", "val"])))
    splits = {split: selected.filter(pc.equal(selected["split"], upstream))["_id"].to_pylist()
              for split, upstream in [("train", "train"), ("validation", "val")]}
    if not all(splits.values()):
        raise ValueError("Published source splits do not contain train and val IDs")
    (output / "source_ids.json").write_text(json.dumps(splits))
    (output / "origin.json").write_text(json.dumps({"repository": "JoakimEdin/medical-coding-reproducibility",
                                                    "revision": EDIN_REVISION, "kind": "public split identifiers"}, indent=2))
    print(json.dumps({name: len(ids) for name, ids in splits.items()}))


def download_source(args):
    username = input("PhysioNet username: ")
    password = getpass.getpass("PhysioNet password: ")
    with httpx.Client(auth=(username, password), timeout=120, follow_redirects=False) as client:
        for dataset, filename in SOURCE_FILES:
            url = f"https://physionet.org/files/{dataset}/{filename}"
            destination = Path(args.output) / dataset / filename
            download_file(client, url, destination)


def download_file(client, url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = client.head(url)
    response.raise_for_status()
    size = int(response.headers["content-length"])
    if destination.exists():
        if destination.stat().st_size != size:
            raise ValueError(f"Existing file has unexpected size: {destination}")
        print(f"Already present: {destination}", flush=True)
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with client.stream("GET", url, headers=headers) as result:
        result.raise_for_status()
        if offset and result.status_code != 206:
            raise ValueError("Server does not support resuming this partial download")
        with partial.open("ab" if offset else "xb") as stream:
            for block in result.iter_bytes(1024 * 1024):
                stream.write(block)
    if partial.stat().st_size != size:
        raise IOError(f"Incomplete download: {partial}")
    partial.rename(destination)
    print(f"Downloaded {destination}: {size / 1e9:.3f} GB", flush=True)

