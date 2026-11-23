import contextlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from icd_transfer.cli import parser
from icd_transfer.benchmark import align_record, published_code
from icd_transfer.codebook import Code, load_codebook
from icd_transfer.data import Notes, chunk_tokens, load_tokenizer, prepare
from icd_transfer.inference import compile_codebook, predict
from icd_transfer.metrics import Counts
from icd_transfer.models import build_model, cast_projection, cast_weights
from icd_transfer.sampling import merged_labels, sample_candidates
from icd_transfer.smoke import fixture
from icd_transfer.train import train


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        Path(".cache").mkdir(exist_ok=True)
        cls.root = Path(tempfile.mkdtemp(prefix="unit-", dir=".cache"))
        with contextlib.redirect_stdout(io.StringIO()):
            cls.model_dir = fixture(cls.root / "fixture")
        cls.tokenizer = load_tokenizer(cls.model_dir)
        cls.codes = load_codebook(cls.root / "fixture/codes.json")
        cls.notes = Notes(cls.root / "fixture/train", "train")
        cls.args = SimpleNamespace(device="cpu", label_max_len=256, label_batch_size=2,
                                   head_max_len=192, pair_batch_size=2, ridge=1.0)

    def test_test_data_requires_approval_before_file_access(self):
        with self.assertRaises(PermissionError):
            Notes("does-not-exist", "test")
        with self.assertRaises(PermissionError):
            prepare(SimpleNamespace(split="test", test_approved=False))

    def test_source_only_preparation(self):
        with self.assertRaises(ValueError):
            prepare(SimpleNamespace(split="train", version=10))

    def test_code_text_does_not_encode_version(self):
        left = Code("icd9_diag:X", "icd9_diag", "X", "asthma")
        right = Code("icd10_diag:Y", "icd10_diag", "Y", "asthma")
        self.assertEqual(left.text, right.text)

    def test_merge_targets_are_or_not_sum(self):
        truth = torch.tensor([[1., 1.], [0., 1.], [0., 0.]])
        texts, target = merged_labels(self.codes[:2], truth, 3, random.Random(1))
        self.assertEqual(len(texts), 3)
        torch.testing.assert_close(target, torch.tensor([[1., 1., 1.], [1., 1., 1.], [0., 0., 0.]]))

    def test_sampled_loss_estimates_full_label_loss(self):
        rows = [{"labels": [self.codes[0].key]}]
        logits, target = torch.tensor([[0.2, -1., 0.8, 1.3]]), torch.tensor([[1., 0., 0., 0.]])
        losses = {}
        for seed in range(30):
            indices, weights = sample_candidates(rows, self.codes, 2, random.Random(seed), "cpu")
            negative = next(index for index in indices if index != 0)
            loss = F.binary_cross_entropy_with_logits(logits[:, indices], target[:, indices], reduction="none")
            losses[negative] = (loss * weights).sum() / 4
        self.assertEqual(len(losses), 3)
        torch.testing.assert_close(torch.stack(list(losses.values())).mean(), F.binary_cross_entropy_with_logits(logits, target))

    def test_cast_matches_ridge_solution_with_bias(self):
        embedding = torch.eye(3)
        weight = torch.tensor([[1., 0.], [0., 2.], [-1., 1.]])
        bias = torch.tensor([0.1, 0.2, 0.3])
        projection = cast_projection(embedding, weight, bias, ridge=1.0)
        expected = torch.cat((weight, bias[:, None]), dim=1) / 2
        torch.testing.assert_close(projection, expected)
        torch.testing.assert_close(cast_weights(embedding[[2, 0]], projection), expected[[2, 0]])

    def test_metrics_against_hand_calculation(self):
        counts = Counts([0.5], 3)
        counts.update(np.array([[0.9, 0.9, 0.1], [0.1, 0.9, 0.1]]), np.array([[1, 0, 1], [0, 1, 0]]))
        self.assertAlmostEqual(counts.result(0)["micro_f1"], 2 / 3)
        self.assertAlmostEqual(counts.result(0)["macro_f1"], 5 / 9)
        unsupported = Counts([0.5], 4)
        unsupported.update(np.array([[0.9, 0.9, 0.1, 0.8], [0.1, 0.9, 0.1, 0.1]]),
                           np.array([[1, 0, 1, 0], [0, 1, 0, 0]]))
        self.assertAlmostEqual(unsupported.result(0)["macro_f1"], 5 / 12)
        self.assertAlmostEqual(unsupported.result(0)["macro_f1_supported"], 5 / 9)
        self.assertAlmostEqual(unsupported.result(0)["micro_f1"], 4 / 7)
        self.assertAlmostEqual(unsupported.result(0)["micro_f1_supported"], 2 / 3)

    def test_published_code_notation_preserves_diagnosis_and_procedure(self):
        cases = [("icd9_diag", "0010", "001.0"), ("icd9_proc", "0010", "00.10"),
                 ("icd9_diag", "E8120", "E812.0"), ("icd9_diag", "V100", "V10.0"),
                 ("icd10_diag", "I10", "I10."), ("icd10_proc", "0016070", "0016070")]
        for system, value, expected in cases:
            self.assertEqual(published_code(Code("key", system, value, "description")), expected)

    def test_benchmark_truncates_words_before_tokenization(self):
        raw = {"_id": "source", "subject_id": "patient", "text": "asthma " * 4000 + "infection",
               "icd9_diag": ["SYN_A"], "icd9_proc": []}
        tokens = self.tokenizer(raw["text"], add_special_tokens=False)["input_ids"]
        chunks, _ = chunk_tokens(tokens, self.tokenizer, 256, 32)
        old = {"id": "source", "subject_id": "patient", "labels": ["icd9_diag:SYN_A"], "chunks": chunks}
        new, words, tokens = align_record(raw, old, {"icd9_diag:SYN_A"}, self.tokenizer)
        kept = [token for chunk in new["chunks"] for token in chunk[1:-1]]
        self.assertEqual(kept, [self.tokenizer.convert_tokens_to_ids("asthma")] * 4000)
        self.assertEqual((words, tokens), (1, 0))
        with self.assertRaises(ValueError):
            align_record(raw, old, set(), self.tokenizer)

    def test_swap_codebook_is_frozen_and_permutation_equivariant(self):
        for kind in ("evidence", "duallaat", "linear", "native"):
            with self.subTest(kind=kind):
                self.check_swap(kind)

    def check_swap(self, kind):
        model = build_model(kind, self.model_dir, self.codes, 16)
        before = {key: tensor.clone() for key, tensor in model.state_dict().items()}
        book, _ = compile_codebook(model, kind, self.codes, self.codes, self.tokenizer, self.args)
        rows = [self.notes[0], self.notes[1]]
        original = predict(model, kind, rows, book, self.tokenizer, self.args)
        reordered = [self.codes[2], self.codes[0], self.codes[3], self.codes[1]]
        other, _ = compile_codebook(model, kind, reordered, self.codes, self.tokenizer, self.args)
        moved = predict(model, kind, rows, other, self.tokenizer, self.args)
        torch.testing.assert_close(moved, original[:, [2, 0, 3, 1]], atol=1e-6, rtol=1e-5)
        new_codes = self.codes + [Code("icd10_diag:SYN_NEW", "icd10_diag", "SYN_NEW", "asthma infection")]
        extended, _ = compile_codebook(model, kind, new_codes, self.codes, self.tokenizer, self.args)
        expanded = predict(model, kind, rows, extended, self.tokenizer, self.args)
        torch.testing.assert_close(expanded[:, :4], original, atol=1e-6, rtol=1e-5)
        single = predict(model, kind, rows[:1], book, self.tokenizer, self.args)
        torch.testing.assert_close(single, original[:1], atol=1e-6, rtol=1e-5)
        for key, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, before[key], rtol=0, atol=0)

    def training_args(self, output, steps, resume=None):
        argv = ["train", "--kind", "evidence", "--train-data", str(self.root / "fixture/train"),
                "--validation-data", str(self.root / "fixture/validation"), "--model-dir", str(self.model_dir),
                "--device", "cpu", "--epochs", "1", "--batch-size", "1", "--candidates", "3",
                "--merges", "2", "--projection-dim", "16", "--warmup-steps", "0", "--max-steps", str(steps),
                "--output", str(output)]
        if resume:
            argv += ["--resume", str(resume)]
        return parser().parse_args(argv)

    def test_resume_matches_uninterrupted_training(self):
        direct, first, resumed = [self.root / name for name in ("direct", "first", "resumed")]
        with contextlib.redirect_stdout(io.StringIO()):
            train(self.training_args(direct, 2))
            train(self.training_args(first, 1))
            train(self.training_args(resumed, 2, first / "step-00000001.pt"))
        a = torch.load(direct / "step-00000002.pt", weights_only=False)
        b = torch.load(resumed / "step-00000002.pt", weights_only=False)
        for key in a["model"]:
            torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)
        self.assertEqual(a["scheduler"], b["scheduler"])
        self.assertEqual(a["threshold"], b["threshold"])


if __name__ == "__main__":
    unittest.main()

