import copy
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModel

from .data import pad_sequences


def encoder_from_config(model_dir):
    config = AutoConfig.from_pretrained(Path(model_dir) / "encoder", local_files_only=True)
    config.reference_compile = False
    return AutoModel.from_config(config, attn_implementation="sdpa")


def load_encoder(model_dir):
    encoder = encoder_from_config(model_dir)
    with safe_open(Path(model_dir) / "model.safetensors", framework="pt") as weights:
        state = {key.removeprefix("encoder."): weights.get_tensor(key)
                 for key in weights.keys() if key.startswith("encoder.")}
    encoder.load_state_dict(state, strict=True)
    return encoder


def encode_notes(encoder, rows, pad_id, device):
    chunks = [chunk for row in rows for chunk in row["chunks"]]
    ids, mask = pad_sequences(chunks, pad_id, device)
    hidden = encoder(input_ids=ids, attention_mask=mask).last_hidden_state
    lengths = [len(row["chunks"]) for row in rows]
    pieces = hidden.split(lengths)
    masks = mask.split(lengths)
    return [(piece[valid], torch.ones(int(valid.sum()), device=device, dtype=torch.bool))
            for piece, valid in zip(pieces, masks)]


def encode_labels(encoder, label_ids, pad_id, device):
    ids, mask = pad_sequences(label_ids, pad_id, device)
    hidden = encoder(input_ids=ids, attention_mask=mask).last_hidden_state
    return (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)


class EvidenceHead(nn.Module):
    def __init__(self, width, projection_dim):
        super().__init__()
        self.note_projection = nn.Linear(width, projection_dim, bias=False)
        self.label_projection = nn.Linear(width, projection_dim, bias=False)
        self.readout = nn.Sequential(nn.Linear(3 * width, projection_dim), nn.GELU(), nn.Linear(projection_dim, 1))

    def score(self, hidden, labels):
        query = self.label_projection(labels)
        key = self.note_projection(hidden)
        attention = (query @ key.T / math.sqrt(key.shape[-1])).float().softmax(-1).to(hidden.dtype)
        evidence = attention @ hidden
        features = torch.cat((evidence, labels, evidence * labels), dim=-1)
        return self.readout(features).squeeze(-1).float()


class EvidenceModel(EvidenceHead):
    def __init__(self, encoder, projection_dim):
        super().__init__(encoder.config.hidden_size, projection_dim)
        self.encoder = encoder

    def encode_labels(self, label_ids, pad_id, device):
        return encode_labels(self.encoder, label_ids, pad_id, device)

    def encode_notes(self, rows, pad_id, device):
        return encode_notes(self.encoder, rows, pad_id, device)


class MultiViewHead(EvidenceHead):
    def __init__(self, width, projection_dim):
        super().__init__(width, projection_dim)
        self.view_weights = nn.Parameter(torch.zeros(3))
        self.operator_embeddings = nn.Embedding(3, width, padding_idx=0)

    def score(self, hidden, labels):
        vectors = labels.vectors.clone()
        vectors[:, 2] = vectors[:, 2] + self.operator_embeddings(labels.operators)
        scores = super().score(hidden, vectors.flatten(0, 1)).reshape(-1, 3)
        weights = self.view_weights.expand_as(scores).masked_fill(~labels.available, -torch.inf).softmax(-1)
        return (scores * weights).sum(-1)


class AlignedMultiViewHead(nn.Module):
    """Pure evidence-label alignment head without a note-independent label frequency shortcut."""

    def __init__(self, width, projection_dim):
        super().__init__()
        self.projection_dim = projection_dim
        self.note_key = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, projection_dim, bias=False))
        self.label_query = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, projection_dim, bias=False))
        self.note_value = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, projection_dim), nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.label_value = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, projection_dim), nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.bias = nn.Parameter(torch.tensor(-6.0))
        self.view_weights = nn.Parameter(torch.zeros(3))
        self.operator_embeddings = nn.Embedding(3, width, padding_idx=0)

    def _active_label_views(self, labels):
        active_views = torch.nonzero(labels.available.any(0), as_tuple=False).flatten()
        vectors = labels.vectors[:, active_views].clone()
        for pos, view in enumerate(active_views.tolist()):
            if view == 2:
                vectors[:, pos] = vectors[:, pos] + self.operator_embeddings(labels.operators)
        flat = vectors.reshape(-1, vectors.shape[-1])
        query = self.label_query(flat).float()
        raw_val = self.label_value(flat).float()
        if labels.delta_query is not None or labels.delta_value is not None:
            num_views = len(active_views)
            if labels.delta_query is not None:
                dq = labels.delta_query.float()[:, None, :].expand(-1, num_views, -1).reshape(-1, self.projection_dim)
                query = query + dq
            if labels.delta_value is not None:
                dv = labels.delta_value.float()[:, None, :].expand(-1, num_views, -1).reshape(-1, self.projection_dim)
                raw_val = raw_val + dv
        label_val = F.normalize(raw_val, dim=-1, eps=1e-6)
        return active_views, query, label_val

    def _combine_views(self, view_scores, available, active_views):
        avail = available[..., active_views]
        view_w = self.view_weights[active_views].expand_as(view_scores).masked_fill(~avail, -torch.inf).softmax(-1)
        return (view_scores * view_w).sum(-1)

    def score(self, hidden, labels):
        active_views, query, label_val = self._active_label_views(labels)
        key = self.note_key(hidden).float()
        note_val = self.note_value(hidden).float()
        attn = (query @ key.T / math.sqrt(self.projection_dim)).softmax(-1)
        evidence = F.normalize(attn @ note_val, dim=-1, eps=1e-6)
        scale = self.log_scale.exp().clamp(max=100.0)
        view_scores = (scale * (evidence * label_val).sum(-1) + self.bias).reshape(len(labels), len(active_views))
        return self._combine_views(view_scores, labels.available, active_views)

    def score_batch(self, notes, labels):
        active_views, query, label_val = self._active_label_views(labels)
        batch = len(notes)
        max_chunks = max(len(note) for note in notes)
        width = notes[0].shape[-1]
        padded = notes[0].new_zeros((batch, max_chunks, width))
        mask = torch.zeros((batch, max_chunks), dtype=torch.bool, device=padded.device)
        for i, note in enumerate(notes):
            padded[i, :len(note)] = note
            mask[i, :len(note)] = True
        key = self.note_key(padded).float()
        note_val = self.note_value(padded).float()
        attn_logits = torch.einsum("kr,bmr->bkm", query, key) / math.sqrt(self.projection_dim)
        attn = attn_logits.masked_fill(~mask[:, None, :], -torch.inf).softmax(-1)
        evidence = F.normalize(torch.einsum("bkm,bmr->bkr", attn, note_val), dim=-1, eps=1e-6)
        scale = self.log_scale.exp().clamp(max=100.0)
        scores = (scale * (evidence * label_val[None]).sum(-1) + self.bias).reshape(
            batch, len(labels), len(active_views),
        )
        return self._combine_views(scores, labels.available[None], active_views)


class AttentionCASTHead(AlignedMultiViewHead):
    """AlignedMultiViewHead with per-source-code residuals compiled to new codebooks via Attention-CAST."""

    def __init__(self, width, projection_dim, source_size, residual_dropout=0.25):
        super().__init__(width, projection_dim)
        if source_size <= 0:
            raise ValueError("AttentionCASTHead requires positive source_size")
        self.source_size = int(source_size)
        self.residual_dropout = float(residual_dropout)
        self.delta_query = nn.Parameter(torch.zeros(self.source_size, projection_dim))
        self.delta_value = nn.Parameter(torch.zeros(self.source_size, projection_dim))

    def attach_source_residuals(self, labels, source_indices):
        index = torch.as_tensor(source_indices, device=self.delta_query.device, dtype=torch.long)
        dq = self.delta_query[index]
        dv = self.delta_value[index]
        if self.training and self.residual_dropout > 0.0:
            keep = (torch.rand((len(index), 1), device=dq.device) >= self.residual_dropout).to(dq.dtype)
            dq = dq * keep
            dv = dv * keep
        return labels.with_residuals(dq, dv)


class NoteResidualProjector(nn.Module):
    """Residual FFN block that maps frozen clinical note chunks into the locked multilingual label space."""

    def __init__(self, width, hidden_dim=None, dropout=0.1):
        super().__init__()
        inner = hidden_dim or (2 * width)
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, inner)
        self.fc2 = nn.Linear(inner, width)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return x + self.fc2(self.dropout(F.gelu(self.fc1(self.norm(x)))))


class IsometricMultiViewHead(nn.Module):
    """Locked-label isometric clinical reader: never warps the pretrained multilingual label manifold."""

    def __init__(self, width, view_dropout=0.5, note_dropout=0.1):
        super().__init__()
        self.width = int(width)
        self.view_dropout = float(view_dropout)
        self.register_buffer("label_mean", torch.zeros(self.width))
        self.register_buffer("has_label_mean", torch.tensor(False))
        self.note_key = NoteResidualProjector(self.width, 2 * self.width, dropout=note_dropout)
        self.note_value = nn.Sequential(
            NoteResidualProjector(self.width, 2 * self.width, dropout=note_dropout),
            NoteResidualProjector(self.width, 2 * self.width, dropout=note_dropout),
        )
        self.attn_log_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.log_scale = nn.Parameter(torch.tensor(math.log(12.0)))
        self.bias = nn.Parameter(torch.tensor(-6.0))
        self.view_weights = nn.Parameter(torch.zeros(3))
        self.operator_embeddings = nn.Embedding(3, self.width, padding_idx=0)
        nn.init.zeros_(self.operator_embeddings.weight)

    def set_label_mean(self, mean):
        self.label_mean.copy_(mean.detach().to(device=self.label_mean.device, dtype=self.label_mean.dtype))
        self.has_label_mean.fill_(True)

    def _center(self, vectors):
        if bool(self.has_label_mean.item()):
            return vectors.float() - self.label_mean.float()
        return vectors.float()

    def _active_label_views(self, labels):
        active_views = torch.nonzero(labels.available.any(0), as_tuple=False).flatten()
        vectors = labels.vectors[:, active_views].float()
        if any(v == 2 for v in active_views.tolist()):
            vectors = vectors.clone()
            for pos, view in enumerate(active_views.tolist()):
                if view == 2:
                    vectors[:, pos] = vectors[:, pos] + self.operator_embeddings(labels.operators).float()
        base = F.normalize(self._center(vectors.reshape(-1, self.width)), dim=-1, eps=1e-6)
        query = base
        proto = base
        if labels.delta_query is not None or labels.delta_value is not None:
            num_views = len(active_views)
            if labels.delta_query is not None:
                dq = labels.delta_query.float()[:, None, :].expand(-1, num_views, -1).reshape(-1, self.width)
                query = F.normalize(base + dq, dim=-1, eps=1e-6)
            if labels.delta_value is not None:
                dv = labels.delta_value.float()[:, None, :].expand(-1, num_views, -1).reshape(-1, self.width)
                proto = F.normalize(base + dv, dim=-1, eps=1e-6)
        return active_views, query, proto

    def _combine_views(self, view_scores, available, active_views):
        avail = available[..., active_views]
        if self.training and self.view_dropout > 0.0 and len(active_views) > 1:
            views = active_views.tolist()
            if 0 in views and 1 in views:
                p0, p1 = views.index(0), views.index(1)
                both = avail[..., p0] & avail[..., p1]
                drop = both & (torch.rand(both.shape, device=both.device) < self.view_dropout)
                avail = avail.clone()
                avail[..., p1] = avail[..., p1] & ~drop
        view_w = self.view_weights[active_views].expand_as(view_scores).masked_fill(~avail, -torch.inf).softmax(-1)
        return (view_scores * view_w).sum(-1)

    def _cosine_readout(self, key, val, mask, query, proto):
        # Exact cosine similarity cos(sum_c alpha_c v_c, proto) without materializing [B, K, D]
        attn_scale = self.attn_log_scale.exp().clamp(max=100.0)
        attn_logits = attn_scale * torch.matmul(key, query.T).transpose(1, 2)
        attn = attn_logits.masked_fill(~mask[:, None, :], -torch.inf).softmax(-1)
        chunk_dot = torch.matmul(val, proto.T).transpose(1, 2)
        dot = (attn * chunk_dot).sum(-1)
        gram = torch.matmul(val, val.transpose(1, 2))
        norm_sq = (torch.matmul(attn, gram) * attn).sum(-1).clamp_min(1e-12)
        return dot * norm_sq.rsqrt()

    def score(self, hidden, labels):
        active_views, query, proto = self._active_label_views(labels)
        key = F.normalize(self._center(self.note_key(hidden)), dim=-1, eps=1e-6).unsqueeze(0)
        val = self._center(self.note_value(hidden)).unsqueeze(0)
        mask = torch.ones((1, len(hidden)), dtype=torch.bool, device=hidden.device)
        cos = self._cosine_readout(key, val, mask, query, proto).squeeze(0)
        scale = self.log_scale.exp().clamp(max=100.0)
        view_scores = (scale * cos + self.bias).reshape(len(labels), len(active_views))
        return self._combine_views(view_scores, labels.available, active_views)

    def score_batch(self, notes, labels):
        active_views, query, proto = self._active_label_views(labels)
        batch = len(notes)
        max_chunks = max(len(note) for note in notes)
        padded = notes[0].new_zeros((batch, max_chunks, self.width))
        mask = torch.zeros((batch, max_chunks), dtype=torch.bool, device=padded.device)
        for i, note in enumerate(notes):
            padded[i, :len(note)] = note
            mask[i, :len(note)] = True
        key = F.normalize(self._center(self.note_key(padded)), dim=-1, eps=1e-6)
        val = self._center(self.note_value(padded)) * mask.unsqueeze(-1).float()
        cos = self._cosine_readout(key, val, mask, query, proto)
        scale = self.log_scale.exp().clamp(max=100.0)
        scores = (scale * cos + self.bias).reshape(batch, len(labels), len(active_views))
        return self._combine_views(scores, labels.available[None], active_views)


class IsometricAttentionCASTHead(IsometricMultiViewHead):
    """Locked-label isometric clinical reader + per-code residuals compiled via Attention-CAST."""

    def __init__(self, width, source_size, residual_dropout=0.5, view_dropout=0.5, note_dropout=0.1):
        super().__init__(width, view_dropout=view_dropout, note_dropout=note_dropout)
        if source_size <= 0:
            raise ValueError("IsometricAttentionCASTHead requires positive source_size")
        self.source_size = int(source_size)
        self.residual_dropout = float(residual_dropout)
        self.delta_query = nn.Parameter(torch.zeros(self.source_size, self.width))
        self.delta_value = nn.Parameter(torch.zeros(self.source_size, self.width))

    def attach_source_residuals(self, labels, source_indices):
        index = torch.as_tensor(source_indices, device=self.delta_query.device, dtype=torch.long)
        dq = self.delta_query[index]
        dv = self.delta_value[index]
        if self.training and self.residual_dropout > 0.0:
            keep = (torch.rand((len(index), 1), device=dq.device) >= self.residual_dropout).to(dq.dtype)
            dq = dq * keep
            dv = dv * keep
        return labels.with_residuals(dq, dv)


def build_multiview_head(width, projection_dim, head_mode="mlp", source_size=0, residual_dropout=0.25, view_dropout=0.5):
    if head_mode == "mlp":
        return MultiViewHead(width, projection_dim)
    if head_mode == "aligned":
        return AlignedMultiViewHead(width, projection_dim)
    if head_mode == "attention_cast":
        return AttentionCASTHead(width, projection_dim, source_size, residual_dropout=residual_dropout)
    if head_mode == "isometric":
        return IsometricMultiViewHead(width, view_dropout=view_dropout)
    if head_mode == "isometric_cast":
        return IsometricAttentionCASTHead(
            width, source_size, residual_dropout=residual_dropout, view_dropout=view_dropout,
        )
    raise ValueError(f"Unknown multiview head mode: {head_mode}")


class MultiViewEvidence(nn.Module):
    def __init__(self, encoder, projection_dim, head_mode="mlp", source_size=0, residual_dropout=0.25, view_dropout=0.5):
        super().__init__()
        self.encoder = encoder
        self.head = build_multiview_head(
            encoder.config.hidden_size,
            projection_dim,
            head_mode,
            source_size=source_size,
            residual_dropout=residual_dropout,
            view_dropout=view_dropout,
        )
        self.set_trainable_encoder_layers(0)

    def set_trainable_encoder_layers(self, count):
        if count < 0 or count > len(self.encoder.layers):
            raise ValueError("Invalid number of trainable encoder layers")
        self.encoder.requires_grad_(False)
        for layer in list(self.encoder.layers)[len(self.encoder.layers) - count:]:
            layer.requires_grad_(True)
        self.trainable_encoder_layers = count
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        if mode:
            for layer in list(self.encoder.layers)[len(self.encoder.layers) - self.trainable_encoder_layers:]:
                layer.train()
        return self

    def encode_notes(self, rows, pad_id, device):
        chunks = [chunk for row in rows for chunk in row["chunks"]]
        ids, mask = pad_sequences(chunks, pad_id, device)
        hidden = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        pieces = pooled.split([len(row["chunks"]) for row in rows])
        return [(piece, torch.ones(len(piece), dtype=torch.bool, device=device)) for piece in pieces]

    def score(self, hidden, labels):
        return self.head.score(hidden, labels)

    def encode_labels(self, label_ids, pad_id, device):
        return encode_labels(self.encoder, label_ids, pad_id, device)


class DualLAAT(nn.Module):
    def __init__(self, encoder, projection_dim):
        super().__init__()
        self.note_encoder = encoder
        self.label_encoder = copy.deepcopy(encoder)
        width = encoder.config.hidden_size
        self.note_projection = nn.Linear(width, projection_dim, bias=False)
        self.label_projection = nn.Linear(width, projection_dim, bias=False)
        self.classifier = nn.Linear(width, 1)

    def score(self, hidden, labels):
        query = torch.tanh(self.label_projection(labels))
        key = torch.tanh(self.note_projection(hidden))
        attention = (query @ key.T).float().softmax(-1).to(hidden.dtype)
        return self.classifier(attention @ hidden).squeeze(-1).float()

    def encode_labels(self, label_ids, pad_id, device):
        return encode_labels(self.label_encoder, label_ids, pad_id, device)

    def encode_notes(self, rows, pad_id, device):
        return encode_notes(self.note_encoder, rows, pad_id, device)


class LinearModel(nn.Module):
    def __init__(self, encoder, num_codes):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.config.hidden_size, num_codes)

    def encode_notes(self, rows, pad_id, device):
        return encode_notes(self.encoder, rows, pad_id, device)

    def encode_labels(self, label_ids, pad_id, device):
        return encode_labels(self.encoder, label_ids, pad_id, device)

    def score_source(self, hidden, indices):
        return F.linear(hidden.mean(0), self.classifier.weight[indices], self.classifier.bias[indices]).float()


class NativeDecision(nn.Module):
    """Laya's published option-marker architecture, with checkpoint-compatible names."""

    def __init__(self, encoder, head_layers=2):
        super().__init__()
        self.encoder = encoder
        width = encoder.config.hidden_size
        layer = nn.TransformerEncoderLayer(width, max(1, width // 64), 4 * width, 0.1,
                                           batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
        self.type_emb = nn.Embedding(3, width)
        self.scorer = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        self.act_head = nn.Sequential(nn.Linear(width + 4, 256), nn.GELU(), nn.Linear(256, 2))
        self.register_buffer("temperature", torch.ones(3))

    def forward(self, input_ids, attention_mask, marker_pos):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = hidden + self.type_emb.weight[0]
        hidden = self.head(hidden, src_key_padding_mask=~attention_mask.bool())
        indices = marker_pos[:, :, None].expand(-1, -1, hidden.shape[-1])
        logits = self.scorer(hidden.gather(1, indices)).squeeze(-1).float()
        return logits[:, 1] - logits[:, 0]


def build_model(kind, model_dir, source_codes, projection_dim=256, initialize=True, head_mode="mlp", residual_dropout=0.25, view_dropout=0.5):
    encoder = load_encoder(model_dir) if initialize else encoder_from_config(model_dir)
    if kind == "native":
        config = json.loads((Path(model_dir) / "rl_agent_config.json").read_text())
        model = NativeDecision(encoder, config["head_layers"])
        if initialize:
            model.load_state_dict(load_file(Path(model_dir) / "model.safetensors"), strict=True)
        model.act_head.requires_grad_(False)
        return model
    if kind == "duallaat":
        return DualLAAT(encoder, projection_dim)
    if kind == "linear":
        return LinearModel(encoder, len(source_codes))
    if kind in ("evidence", "evidence_plain"):
        return EvidenceModel(encoder, projection_dim)
    if kind == "multiview":
        return MultiViewEvidence(
            encoder,
            projection_dim,
            head_mode=head_mode,
            source_size=len(source_codes),
            residual_dropout=residual_dropout,
            view_dropout=view_dropout,
        )
    raise ValueError(f"Unknown model: {kind}")


def native_prompt(tokenizer, text, head_max_len=192):
    question = tokenizer("choice question: Does this clinical note support assigning " + text + "?",
                         add_special_tokens=False)["input_ids"]
    options = [tokenizer(" A: no, the code is not supported", add_special_tokens=False)["input_ids"],
               tokenizer(" B: yes, the code is supported", add_special_tokens=False)["input_ids"]]
    if len(question) + sum(map(len, options)) + 2 > head_max_len:
        raise ValueError("Code description exceeds native head budget; increase --head-max-len")
    ids = [tokenizer.cls_token_id, *question, tokenizer.sep_token_id]
    markers = []
    for option in options:
        markers.append(len(ids))
        ids += [tokenizer.mask_token_id, *option]
    return ids + [tokenizer.sep_token_id], markers


def native_scores(model, rows, prompts, pad_id, device, pair_batch_size):
    answers = []
    for row in rows:
        sequences, positions = native_sequences(row["chunks"], prompts)
        scores = []
        for start in range(0, len(sequences), pair_batch_size):
            ids, mask = pad_sequences(sequences[start:start + pair_batch_size], pad_id, device)
            markers = torch.tensor(positions[start:start + pair_batch_size], device=device)
            if model.training and torch.is_grad_enabled():
                scores.append(checkpoint(model, ids, mask, markers, use_reentrant=False))
            else:
                scores.append(model(ids, mask, markers))
        by_code = torch.cat(scores).reshape(len(prompts), len(row["chunks"]))
        answers.append(by_code.max(-1).values)
    return torch.stack(answers)


def native_sequences(chunks, prompts):
    sequences, positions = [], []
    for prefix, markers in prompts:
        sequences.extend(prefix + chunk[1:] for chunk in chunks)
        positions.extend([markers] * len(chunks))
    return sequences, positions


def cast_projection(embeddings, weight, bias, ridge):
    if ridge <= 0:
        raise ValueError("CAST ridge must be positive")
    source = F.normalize(embeddings.double(), dim=-1)
    outputs = torch.cat((weight.double(), bias.double().unsqueeze(-1)), dim=-1)
    gram = source.T @ source + ridge * torch.eye(source.shape[1], device=source.device, dtype=source.dtype)
    return torch.linalg.solve(gram, source.T @ outputs).float()


def cast_weights(embeddings, projection):
    return F.normalize(embeddings.float(), dim=-1) @ projection

