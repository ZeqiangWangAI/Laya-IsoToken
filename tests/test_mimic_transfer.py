import contextlib
import csv
import gzip
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from icd_transfer.codebook import Code, write_codebook
from icd_transfer.drg_aliases import match_drg_title
from icd_transfer.mimic_transfer import prepare_mimic_transfer


def gzip_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class MimicTransferTests(unittest.TestCase):
    def test_drg_aliases_keep_version_ambiguity_and_code_identity(self):
        aliases = {("001", "TITLE WITH MCC"): {"a"}, ("001", "TITLE WITHOUT MCC"): {"b"},
                   ("002", "TITLE WITH MCC"): {"different-code"}}
        self.assertEqual(match_drg_title("001", "TITLE W", aliases), (["a", "b"], "public_title_prefix"))
        self.assertEqual(match_drg_title("001", "TITLE WITH MCC", aliases), (["a"], "exact_public_title"))
        self.assertEqual(match_drg_title("003", "TITLE W", aliases), ([], "unmapped"))

    def test_unapproved_target_fails_before_network_or_data_access(self):
        with patch("icd_transfer.mimic_transfer.target_memberships") as fetch:
            with self.assertRaises(PermissionError):
                prepare_mimic_transfer(SimpleNamespace(target_preparation_approved=False))
            fetch.assert_not_called()

    def test_source_patient_exclusion_and_unmapped_drg_truth(self):
        root = Path(tempfile.mkdtemp(prefix="mimic-transfer-"))
        codes = [Code("icd10_diag:J45", "icd10_diag", "J45", "asthma"),
                 Code("icd10_proc:0BP", "icd10_proc", "0BP", "procedure")]
        write_codebook(codes, root / "codes.json")
        source = root / "source"
        for split in ("train", "validation"):
            (source / split).mkdir(parents=True)
            (source / split / "subjects.json").write_text(json.dumps(["known"] if split == "train" else ["selection"]))
        alignment = root / "alignment"
        alignment.mkdir()
        write_codebook([Code("ms_drg_semantic:001:v25", "ms_drg_semantic", "001", "known definition")], alignment / "codebook.json")
        (alignment / "edition_aliases.json").write_text(json.dumps([
            {"key": "ms_drg_semantic:001:v25", "code": "001", "normalized_definition": "KNOWN DEFINITION"}]))
        admissions = [("a", "known"), ("b", "new"), ("c", "another"), ("unselected", "unselected")]
        hospital = root / "raw/mimiciv/2.2/hosp"
        gzip_rows(hospital / "diagnoses_icd.csv.gz", [{"hadm_id": i, "icd_version": "10", "icd_code": "J45"} for i, _ in admissions])
        gzip_rows(hospital / "procedures_icd.csv.gz", [{"hadm_id": i, "icd_version": "10", "icd_code": "0BP"} for i, _ in admissions])
        gzip_rows(hospital / "drgcodes.csv.gz", [{"hadm_id": i, "subject_id": patient, "drg_type": "HCFA", "drg_code": "001",
                                                 "description": "unknown definition" if i == "c" else "known definition"}
                                                for i, patient in admissions])
        gzip_rows(root / "raw/mimic-iv-note/2.2/note/discharge.csv.gz",
                  [{"hadm_id": i, "subject_id": patient, "text": "Dose: 2.5 mg; O₂ 98%."} for i, patient in admissions])
        args = SimpleNamespace(target_preparation_approved=True, codebook=root / "codes.json", source=source,
                               raw=root / "raw", drg_alignment=alignment, output=root / "prepared", unavailable_notes=root / "unavailable.json")
        (root / "unavailable.json").write_text('["missing"]')
        with patch("icd_transfer.mimic_transfer.target_memberships", return_value=({"a": "train", "b": "train", "c": "dev", "missing": "train"}, "synthetic")):
            with contextlib.redirect_stdout(io.StringIO()):
                prepare_mimic_transfer(args)
        train = [json.loads(line) for line in (root / "prepared/drg/train.jsonl").read_text().splitlines()]
        dev = [json.loads(line) for line in (root / "prepared/drg/dev.jsonl").read_text().splitlines()]
        self.assertEqual([row["id"] for row in train], ["b"])
        self.assertTrue(dev[0]["labels"][0].startswith("unmapped_drg:"))
        self.assertEqual(train[0]["text"], "Dose: 2.5 mg; O₂ 98%.")
        meta = json.loads((root / "prepared/icd10/metadata.json").read_text())
        self.assertEqual(meta["splits"]["train"]["rows"], 2)
        self.assertEqual(meta["splits"]["train"]["source_patient_overlap_rows"], 1)
        self.assertFalse(meta["model_fit_allowed"])
        self.assertEqual(meta["unavailable_note_exclusions"], 1)


if __name__ == "__main__":
    unittest.main()
