import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow as pa
import pyarrow.feather as feather
import torch

from icd_transfer.baseline_transfer import baseline_logits, compile_baseline
from icd_transfer.cli import parser
from icd_transfer.codebook import Code, load_codebook, write_codebook
from icd_transfer.data import Notes, load_tokenizer
from icd_transfer.evaluate_transfer import evaluate_transfer, source_checkpoint
from icd_transfer.inference import compile_codebook, embed_codebook, predict
from icd_transfer.models import build_model, cast_projection, cast_weights
from icd_transfer.smoke import fixture
from icd_transfer.train import train


class BaselineTransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.root = Path(tempfile.mkdtemp(prefix="baseline-transfer-"))
        with contextlib.redirect_stdout(io.StringIO()):
            cls.model_dir = fixture(cls.root / "fixture")
        cls.codes = load_codebook(cls.root / "fixture/codes.json")
        cls.codes += [Code("icd10_proc:SYN_GROUP", "icd10_proc", "SYN_GROUP", "asthma " * 300)]
        cls.groups = [{"key": cls.codes[-1].key, "operator": "OR",
                       "members": [cls.codes[0].key, cls.codes[2].key]}]
        cls.tokenizer = load_tokenizer(cls.model_dir)
        cls.args = SimpleNamespace(device="cpu", label_max_len=128, head_max_len=192,
                                   label_batch_size=2, pair_batch_size=2, ridge=1.0)

    def test_public_composition_and_frozen_codebook_swap(self):
        for kind in ("native", "duallaat", "linear"):
            with self.subTest(kind=kind), torch.inference_mode():
                self.check_composition(kind)

    def test_cast_mapping_retains_float32_under_autocast(self):
        codes = self.codes[:-1]
        model = build_model("linear", self.model_dir, codes, 16).eval()
        embeddings = torch.randn(len(codes), model.classifier.in_features).bfloat16()
        expected = cast_weights(embeddings, cast_projection(
            embeddings, model.classifier.weight, model.classifier.bias, self.args.ridge))
        with patch("icd_transfer.baseline_transfer.embed_codebook", return_value=embeddings):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                compiled = compile_baseline(model, "linear", codes, codes, [], self.tokenizer, self.args)
        self.assertEqual(compiled.dtype, torch.float32)
        torch.testing.assert_close(compiled, expected, rtol=0, atol=0)

    def check_composition(self, kind):
        source = self.codes[:-1]
        model = build_model(kind, self.model_dir, source, 16).eval()
        before = {key: value.clone() for key, value in model.state_dict().items()}
        rows = [Notes(self.root / "fixture/train", "train")[0]]
        book = compile_baseline(model, kind, self.codes, source, self.groups, self.tokenizer, self.args)
        scores = baseline_logits(model, kind, rows, book, self.tokenizer, self.args)
        old, _ = compile_codebook(model, kind, source, source, self.tokenizer, self.args)
        expected = predict(model, kind, rows, old, self.tokenizer, self.args)
        torch.testing.assert_close(scores[:, :4].sigmoid(), expected)
        if kind == "native":
            torch.testing.assert_close(scores[:, 4], scores[:, [0, 2]].amax(-1))
        else:
            embeddings = embed_codebook(model, source, self.tokenizer, self.args)
            merged = embeddings[[0, 2]].mean(0, keepdim=True)
            if kind == "linear":
                projection = cast_projection(embeddings, model.classifier.weight, model.classifier.bias, 1.0)
                merged = cast_weights(merged, projection)
            torch.testing.assert_close(book[4:5], merged)
        order = [4, 2, 0, 3, 1]
        moved = compile_baseline(model, kind, [self.codes[i] for i in order], source,
                                 self.groups, self.tokenizer, self.args)
        reordered = baseline_logits(model, kind, rows, moved, self.tokenizer, self.args)
        torch.testing.assert_close(reordered, scores[:, order])
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def train_baseline(self, kind):
        argv = ["train", "--kind", kind, "--train-data", str(self.root / "fixture/train"),
                "--validation-data", str(self.root / "fixture/validation"),
                "--model-dir", str(self.model_dir), "--device", "cpu", "--epochs", "1",
                "--batch-size", "2", "--candidates", "4", "--projection-dim", "16",
                "--warmup-steps", "0", "--max-steps", "1", "--checkpoint-every", "1",
                "--output", str(self.root / kind)]
        with contextlib.redirect_stdout(io.StringIO()):
            return train(parser().parse_args(argv))

    def target_data(self, mode):
        path = self.root / f"target-{mode}"
        path.mkdir()
        rows = feather.read_table(self.root / "fixture/validation/validation.arrow").to_pylist()
        for i, row in enumerate(rows):
            row["label_groups"] = [row["labels"]]
            if i > 0:
                row["subject_id"] = f"target-{i}"
        rows[-1]["subject_id"] = None
        feather.write_feather(pa.Table.from_pylist(rows), path / "dev.arrow")
        write_codebook(self.codes, path / "codebook.json")
        (path / "groups.json").write_text(json.dumps(self.groups))
        origin = json.loads((self.model_dir / "origin.json").read_text())
        metadata = {"dataset": "synthetic", "role": "target_evaluation_only", "model_fit_allowed": False,
                    "threshold_selection_allowed": False, "splits": {"dev": {"rows": 4}},
                    "output_mode": mode, "patient_id_namespace": "mimiciv", "tokenizer": origin}
        (path / "metadata.json").write_text(json.dumps(metadata))
        return path

    def test_source_train_to_all_frozen_task_modes(self):
        modes = ("multilabel", "alternative_label_groups", "single_label_alternative_groups")
        targets = {mode: self.target_data(mode) for mode in modes}
        for kind in ("native", "duallaat", "linear"):
            summary = self.train_baseline(kind)
            checkpoint = summary["best"]["checkpoint"]
            with self.assertRaises(ValueError):
                source_checkpoint(self.root / kind / "train-step-00000001.pt")
            for mode, target in targets.items():
                self.check_evaluation(kind, mode, target, checkpoint, summary)

    def check_evaluation(self, kind, mode, target, checkpoint, summary):
        output = self.root / f"result-{kind}-{mode}"
        argv = ["evaluate-transfer", "--checkpoint", checkpoint, "--data", str(target), "--split", "dev",
                "--model-dir", str(self.model_dir), "--device", "cpu", "--output", str(output)]
        with contextlib.redirect_stdout(io.StringIO()):
            result = evaluate_transfer(parser().parse_args(argv))
        self.assertEqual(result["examples"], 3)
        self.assertEqual(result["source_patient_rows_excluded"], 1)
        self.assertEqual(result["rows_without_patient_id"], 1)
        self.assertEqual(result["threshold"], summary["best"]["threshold"])
        self.assertEqual(result["kind"], kind)
        self.assertFalse(result["model_fit_on_target"])
        if mode.startswith("single_label"):
            predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
            self.assertTrue(all(len(row["predicted"]) == 1 for row in predictions))


if __name__ == "__main__":
    unittest.main()
