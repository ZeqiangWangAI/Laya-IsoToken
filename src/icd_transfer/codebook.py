import csv
import gzip
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Code:
    key: str
    system: str
    code: str
    description: str
    ancestors: tuple[str, ...] = ()

    @property
    def text(self):
        return ": ".join((self.system.rsplit("_", 1)[-1], *self.ancestors, self.description))


def normalize_code(value):
    return str(value).replace(".", "").strip().upper()


def load_codebook(path):
    rows = json.loads(Path(path).read_text())
    codes = [Code(**{**row, "ancestors": tuple(row.get("ancestors", []))}) for row in rows]
    if not codes or len({code.key for code in codes}) != len(codes):
        raise ValueError("Codebook must contain distinct keys")
    if any(not code.description.strip() for code in codes):
        raise ValueError("Every code needs a description")
    return codes


def write_codebook(codes, path):
    with Path(path).open("x") as stream:
        json.dump([asdict(code) for code in codes], stream, ensure_ascii=False, indent=2)


def read_dictionary(path, version, kind):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as stream:
        rows = list(csv.DictReader(stream))
    system = f"icd{version}_{kind}"
    selected = [row for row in rows if int(row["icd_version"]) == version]
    return [Code(f"{system}:{normalize_code(row['icd_code'])}", system,
                 row["icd_code"], row["long_title"]) for row in selected]


def make_codebook(args):
    codes = read_dictionary(args.diagnoses, args.version, "diag")
    codes += read_dictionary(args.procedures, args.version, "proc")
    write_codebook(sorted(codes, key=lambda code: code.key), args.output)

