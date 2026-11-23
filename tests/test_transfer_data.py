import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from icd_transfer.transfer_data import cdn_answer, clinical_text, drg_links, prepare_codiesp
from icd_transfer.transfer_tokens import tokenize_transfer


class TransferDataTests(unittest.TestCase):
    def test_chinese_answers_match_official_width_and_quote_rules(self):
        self.assertEqual(cdn_answer('"甲状腺炎（慢性）##高血压１级"'), ["甲状腺炎(慢性)", "高血压1级"])

    def test_target_tokenization_rejects_test_before_loading_files(self):
        root = Path(tempfile.mkdtemp(prefix="transfer-guard-"))
        (root / "metadata.json").write_text(json.dumps({"role": "target_evaluation_only",
                                                       "model_fit_allowed": False, "splits": {"test": {}}}))
        with self.assertRaises(PermissionError):
            tokenize_transfer(SimpleNamespace(input=root, output=root / "output"))

    def test_clinical_text_preserves_native_letters_and_measurements(self):
        text = "  Ασθενής\n温度 38.5°C; presión 120/80; cafe\u0301  "
        self.assertEqual(clinical_text(text), "Ασθενής 温度 38.5°C; presión 120/80; café")

    def test_drg_joins_only_source_admissions_and_checks_patient(self):
        root = Path(tempfile.mkdtemp(prefix="drg-test-"))
        path = root / "drg.csv.gz"
        fields = ["subject_id", "hadm_id", "drg_type", "drg_code", "description",
                  "drg_severity", "drg_mortality"]
        with gzip.open(path, "wt", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerows([["p1", "a1", "HCFA", "1", "Definition A", "", ""],
                              ["p1", "a1", "APR", "1", "Definition B", "1", "1"],
                              ["p2", "heldout", "HCFA", "2", "Held out", "", ""]])
        rows = list(drg_links(path, {"a1": ("train", "p1")}))
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["drg_type"] for row in rows}, {"HCFA", "APR"})
        with self.assertRaisesRegex(ValueError, "patient IDs"):
            list(drg_links(path, {"a1": ("train", "wrong-patient")}))

    def test_spanish_preserves_oov_and_never_reads_test(self):
        root = Path(tempfile.mkdtemp(prefix="codiesp-test-"))
        codes = root / "codes"
        codes.mkdir()
        (codes / "codiesp-D_codes.tsv").write_text("A00\tCólera\tCholera\n")
        (codes / "codiesp-P_codes.tsv").write_text("0016\t\t\n0016070\tDerivación A\tBypass A\n0016071\tDerivación B\tBypass B\n")
        for split in ("train", "dev"):
            notes = root / split / "text_files"
            notes.mkdir(parents=True)
            (notes / "doc.txt").write_text("Paciente con fiebre: 38.5°C.")
            (root / split / f"{split}D.tsv").write_text("doc\ta00\ndoc\tz99\n")
            (root / split / f"{split}P.tsv").write_text("")
        (root / "test").write_bytes(b"not a readable test directory")
        output = root / "prepared"
        prepare_codiesp(SimpleNamespace(input=root, code_directory=codes, output=output))
        metadata = json.loads((output / "metadata.json").read_text())
        row = json.loads((output / "train.jsonl").read_text())
        self.assertEqual(metadata["splits"]["train"]["out_of_catalog"], {"cie10es2018_diag:Z99": 1})
        self.assertIn("cie10es2018_diag:Z99", row["labels"])
        self.assertIsNone(row["subject_id"])
        self.assertFalse(metadata["model_fit_allowed"])
        self.assertFalse((output / "test.jsonl").exists())
        groups = json.loads((output / "groups.json").read_text())
        self.assertEqual(groups, [{"key": "cie10es2018_proc:0016", "operator": "OR",
                                  "members": ["cie10es2018_proc:0016070", "cie10es2018_proc:0016071"]}])


if __name__ == "__main__":
    unittest.main()
