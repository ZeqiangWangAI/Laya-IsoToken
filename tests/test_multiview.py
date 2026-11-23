import contextlib
import io
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F

from icd_transfer.codebook import Code
from icd_transfer.data import Notes, load_tokenizer
from icd_transfer.label_views import encode_views
from icd_transfer.models import build_model
from icd_transfer.smoke import fixture
from icd_transfer.source_schemas import family_partitions, schema_codes, schema_targets


class MultiViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.root = Path(tempfile.mkdtemp(prefix="multiview-"))
        with contextlib.redirect_stdout(io.StringIO()):
            cls.model_dir = fixture(cls.root / "fixture")
        cls.tokenizer = load_tokenizer(cls.model_dir)
        cls.rows = [Notes(cls.root / "fixture/train", "train")[i] for i in range(4)]
        cls.codes = [Code("icd9_diag:SYN_A", "icd9_diag", "SYN_A", "asthma", ("clinical",)),
                     Code("icd9_proc:SYN_O", "icd9_proc", "SYN_O", "oxygen")]
        cls.groups = [{"key": "ccs2015_diag:union", "operator": "OR", "description": "asthma OR oxygen",
                       "members": [c.key for c in cls.codes]},
                      {"key": "synthetic:both", "operator": "AND", "description": "asthma oxygen",
                       "members": [c.key for c in cls.codes]}]
        cls.args = SimpleNamespace(label_max_len=128, label_batch_size=8, device="cpu")

    def model(self):
        torch.manual_seed(13)
        return build_model("multiview", self.model_dir, self.codes, projection_dim=16)

    def test_or_and_targets_and_family_partition(self):
        truth = schema_targets([{"labels": [self.codes[0].key]},
                                {"labels": [c.key for c in self.codes]}, {"labels": []}], self.groups, "cpu")
        torch.testing.assert_close(truth, torch.tensor([[1., 0.], [1., 1.], [0., 0.]]))
        families = [f"family-{i}" for i in range(20)]
        first = family_partitions(families, 42)
        random.Random(9).shuffle(families)
        self.assertEqual(first, family_partitions(families, 42))
        self.assertEqual([list(first.values()).count(s) for s in ("train", "selection", "audit")], [12, 4, 4])

    def test_composition_uses_all_members_without_encoding_expanded_text(self):
        model = self.model().eval()
        group = {k: v for k, v in self.groups[0].items() if k != "description"}
        codes = self.codes + [Code(group["key"], "source_schema_term", "union", "asthma " * 1000)]
        labels = encode_views(model, codes, [group], self.tokenizer, self.args)
        torch.testing.assert_close(labels.vectors[-1, 2], labels.vectors[:2, 0].mean(0))
        self.assertEqual(labels.available[-1].tolist(), [False, False, True])
        self.assertEqual(labels.operators.tolist(), [0, 0, 1])

    def test_sparse_composition_runs_in_float32_under_autocast(self):
        model = self.model().eval()
        original_mm = torch.sparse.mm
        def sparse_mm(matrix, vectors):
            self.assertFalse(torch.is_autocast_enabled("cpu"))
            self.assertEqual(vectors.dtype, torch.float32)
            return original_mm(matrix, vectors)
        with patch("torch.sparse.mm", side_effect=sparse_mm):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                labels = encode_views(model, self.codes + schema_codes(self.groups),
                                      self.groups, self.tokenizer, self.args)
        expected = labels.vectors[:2, 0].float().mean(0).to(labels.vectors.dtype)
        torch.testing.assert_close(labels.vectors[-1, 2], expected, rtol=0, atol=0)

    def test_frozen_encoder_head_learning_and_codebook_swap(self):
        model = self.model().train()
        before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
        codes = self.codes + schema_codes(self.groups)
        labels = encode_views(model, codes, self.groups, self.tokenizer, self.args)
        hidden = model.encode_notes(self.rows, self.tokenizer.pad_token_id, "cpu")
        self.assertEqual([len(h) for h, _ in hidden], [len(row["chunks"]) for row in self.rows])
        target = schema_targets(self.rows, self.groups, "cpu")
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        losses = []
        for _ in range(12):
            optimizer.zero_grad()
            scores = torch.stack([model.score(h, labels[2:]) for h, _ in hidden])
            loss = F.binary_cross_entropy_with_logits(scores, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0])
        self.assertFalse(model.encoder.training)
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
        for name, value in model.encoder.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        state = {k: v.clone() for k, v in model.state_dict().items()}
        model.eval()
        with torch.no_grad():
            original = model.score(hidden[0][0], labels)
            permuted = encode_views(model, list(reversed(codes)), self.groups, self.tokenizer, self.args)
            torch.testing.assert_close(model.score(hidden[0][0], permuted), original.flip(0))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)

    def test_partial_encoder_gradients_through_composition(self):
        model = self.model()
        model.set_trainable_encoder_layers(1)
        model.train()
        labels = encode_views(model, self.codes + schema_codes(self.groups), self.groups, self.tokenizer, self.args)
        hidden = model.encode_notes(self.rows[:1], self.tokenizer.pad_token_id, "cpu")[0][0]
        model.score(hidden, labels).square().mean().backward()
        self.assertTrue(any(p.grad is not None for p in model.encoder.layers[-1].parameters()))
        self.assertTrue(all(p.grad is None for p in model.encoder.layers[0].parameters()))
        self.assertTrue(all(p.grad is None for p in model.encoder.embeddings.parameters()))


if __name__ == "__main__":
    unittest.main()
