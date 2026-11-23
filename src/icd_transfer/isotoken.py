import math

import torch
from torch import nn
from torch.nn import functional as F

from .data import pad_sequences
from .models import encoder_from_config, load_encoder


def set_encoder_trainable_layers(encoder, count):
    """Freeze embeddings and bottom layers; unfreeze top `count` transformer layers + final norm."""
    layers = list(encoder.layers)
    total = len(layers)
    if count < 0 or count > total:
        raise ValueError(f"trainable_encoder_layers={count} outside [0, {total}]")
    encoder.requires_grad_(False)
    for layer in layers[total - count:]:
        layer.requires_grad_(True)
    if count > 0:
        for name in ("final_norm", "norm", "layer_norm"):
            mod = getattr(encoder, name, None)
            if mod is not None:
                mod.requires_grad_(True)
    return count


def pool_note_spans(hidden, mask, chunk_counts, span_size=4, input_ids=None, special_ids=()):
    """Pool chunked token states [N_chunks, L, D] into padded clinical spans [B, S_max, D].

    Drops [CLS]/[SEP]/[PAD] tokens when `special_ids` are provided (falling back to any non-pad
    token if a chunk contains only specials), then computes exact non-overlapping masked mean
    pooling of width `span_size` inside every chunk.
    """
    if span_size <= 0:
        raise ValueError("span_size must be positive")
    valid = mask.bool()
    if input_ids is not None and special_ids:
        content = valid.clone()
        for sid in special_ids:
            if sid is not None:
                content = content & (input_ids != int(sid))
        empty_chunk = ~content.any(dim=1, keepdim=True)
        valid = torch.where(empty_chunk, valid, content)
    n_chunks, seq_len, dim = hidden.shape
    if span_size == 1:
        span_vecs = hidden
        span_valid = valid
    else:
        rem = seq_len % span_size
        if rem:
            pad_len = span_size - rem
            hidden = F.pad(hidden, (0, 0, 0, pad_len))
            valid = F.pad(valid, (0, pad_len), value=False)
        n_spans = hidden.shape[1] // span_size
        grouped_h = hidden.view(n_chunks, n_spans, span_size, dim)
        grouped_m = valid.view(n_chunks, n_spans, span_size)
        counts = grouped_m.sum(dim=2)
        span_valid = counts > 0
        denom = counts.clamp_min(1).unsqueeze(-1).to(hidden.dtype)
        span_vecs = (grouped_h * grouped_m.unsqueeze(-1).to(hidden.dtype)).sum(dim=2) / denom

    pieces_h = span_vecs.split(chunk_counts, dim=0)
    pieces_m = span_valid.split(chunk_counts, dim=0)
    notes_h = []
    lengths = []
    for piece_h, piece_m in zip(pieces_h, pieces_m):
        flat_h = piece_h.reshape(-1, dim)
        flat_m = piece_m.reshape(-1)
        kept = flat_h[flat_m]
        if len(kept) == 0:
            kept = flat_h[:1]
        notes_h.append(kept)
        lengths.append(len(kept))
    max_spans = max(lengths)
    padded = hidden.new_zeros((len(notes_h), max_spans, dim))
    out_mask = torch.zeros((len(notes_h), max_spans), dtype=torch.bool, device=hidden.device)
    for idx, (note_h, length) in enumerate(zip(notes_h, lengths)):
        padded[idx, :length] = note_h
        out_mask[idx, :length] = True
    return padded, out_mask


class SpanToConceptProjector(nn.Module):
    """Residual MLP projecting clinical span hidden states (R^{note_dim}) into locked concept space (R^{label_dim})."""

    def __init__(self, note_dim, label_dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        inner = hidden_dim or max(note_dim, label_dim)
        self.norm = nn.LayerNorm(note_dim)
        self.skip = nn.Linear(note_dim, label_dim, bias=False)
        self.fc1 = nn.Linear(note_dim, inner)
        self.fc2 = nn.Linear(inner, label_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        normed = self.norm(x)
        return self.skip(normed) + self.fc2(self.dropout(F.gelu(self.fc1(normed))))


class IsoTokenHead(nn.Module):
    """Fine-grained span-level isometric clinical operator on a strictly locked multilingual concept manifold.

    Zero trainable parameters touch label representations u_c in S^{d_z - 1}.
    Span keys and values are unit-normalized; attention-weighted evidence r_c(x) lies in [-1, 1],
    naturally contracting toward 0 when no span supports code c without O(S^2) Gram overhead.
    """

    def __init__(
        self,
        note_dim,
        label_dim,
        source_size=0,
        hidden_dim=None,
        dropout=0.1,
        label_chunk_size=4096,
        prior_tau=0.0,
        unlock_label_mlp=False,
    ):
        super().__init__()
        self.note_dim = int(note_dim)
        self.label_dim = int(label_dim)
        self.source_size = int(source_size)
        self.label_chunk_size = int(label_chunk_size)
        self.prior_tau = float(prior_tau)
        self.unlock_label_mlp = bool(unlock_label_mlp)
        self.register_buffer("label_mean", torch.zeros(self.label_dim))
        self.register_buffer("has_label_mean", torch.tensor(False))
        self.note_key = SpanToConceptProjector(self.note_dim, self.label_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.note_value = SpanToConceptProjector(self.note_dim, self.label_dim, hidden_dim=hidden_dim, dropout=dropout)
        if self.unlock_label_mlp:
            # Trainable label-side residual MLP control (starts on the identity manifold, free to warp during training).
            self.label_proj = SpanToConceptProjector(self.label_dim, self.label_dim, hidden_dim=hidden_dim, dropout=dropout)
            nn.init.eye_(self.label_proj.skip.weight)
        else:
            self.label_proj = None
        self.attn_log_scale = nn.Parameter(torch.tensor(math.log(15.0)))
        self.log_scale = nn.Parameter(torch.tensor(math.log(15.0)))
        self.bias = nn.Parameter(torch.tensor(-6.9))
        if self.prior_tau > 0.0 and self.source_size > 0:
            # Fixed empirical Bayesian log-odds prior (Menon et al. logit adjustment) used only during source training/in-domain eval.
            self.register_buffer("source_prior_delta", torch.zeros(self.source_size))
            self.register_parameter("source_bias", None)
        elif self.source_size > 0:
            # Detached in-domain source bias calibration only; never receives encoder gradients and never used on transfer.
            self.register_buffer("source_prior_delta", None)
            self.source_bias = nn.Parameter(torch.zeros(self.source_size))
        else:
            self.register_buffer("source_prior_delta", None)
            self.register_parameter("source_bias", None)

    def set_source_prior(self, counts, num_examples, smoothing=0.5):
        if self.source_prior_delta is None:
            return
        c = torch.as_tensor(counts, dtype=torch.float64)
        n = float(max(num_examples, 1))
        pi = ((c + float(smoothing)) / (n + 2.0 * float(smoothing))).clamp(min=1e-7, max=1.0 - 1e-7)
        ell = torch.log(pi / (1.0 - pi))
        bar_pi = pi.mean().clamp(min=1e-7, max=1.0 - 1e-7)
        bar_ell = torch.log(bar_pi / (1.0 - bar_pi))
        delta = (ell - bar_ell).to(device=self.source_prior_delta.device, dtype=self.source_prior_delta.dtype)
        self.source_prior_delta.copy_(delta)

    def set_label_mean(self, mean):
        self.label_mean.copy_(mean.detach().to(device=self.label_mean.device, dtype=self.label_mean.dtype))
        self.has_label_mean.fill_(True)

    def prepare_prototypes(self, raw_vectors):
        """Center by source mean mu_A (when set) and L2-normalize onto S^{d_z - 1}."""
        vecs = raw_vectors.float()
        if bool(self.has_label_mean.item()):
            vecs = vecs - self.label_mean.float()
        normed = F.normalize(vecs, dim=-1, eps=1e-6)
        if self.label_proj is not None:
            normed = F.normalize(self.label_proj(normed.to(self.label_proj.skip.weight.dtype)).float(), dim=-1, eps=1e-6)
        return normed

    def project_spans(self, spans, span_mask):
        mask_f = span_mask.unsqueeze(-1).to(spans.dtype)
        key = F.normalize(self.note_key(spans).float(), dim=-1, eps=1e-6) * mask_f.float()
        val = F.normalize(self.note_value(spans).float(), dim=-1, eps=1e-6) * mask_f.float()
        return key, val

    def score_spans(self, spans, span_mask, prototypes, label_chunk_size=None, return_evidence=False):
        """Compute exact O(B * S * C) span-attention logits [B, C].

        `prototypes` must already be unit-normalized [C, d_z] (via `prepare_prototypes`).
        """
        key, val = self.project_spans(spans, span_mask)
        proto = prototypes.to(device=key.device, dtype=key.dtype)
        attn_scale = self.attn_log_scale.exp().clamp(min=1.0, max=100.0)
        scale = self.log_scale.exp().clamp(min=1.0, max=100.0)
        chunk_size = label_chunk_size or self.label_chunk_size
        total_codes = len(proto)
        mask_3d = span_mask[:, :, None]
        # Compute matmuls in bf16 when on CUDA for Tensor Core speed, accumulate softmax/dot in fp32.
        use_bf16 = key.is_cuda and torch.cuda.is_bf16_supported()
        key_mm = key.bfloat16() if use_bf16 else key
        val_mm = val.bfloat16() if use_bf16 else val
        proto_mm = proto.bfloat16() if use_bf16 else proto

        out_chunks = []
        top_span_chunks = [] if return_evidence else None
        for start in range(0, total_codes, chunk_size):
            p_chunk = proto_mm[start:start + chunk_size]
            attn_logits = torch.matmul(key_mm, p_chunk.T).float() * attn_scale
            attn_logits = attn_logits.masked_fill(~mask_3d, -torch.inf)
            attn = attn_logits.softmax(dim=1)
            val_dot = torch.matmul(val_mm, p_chunk.T).float()
            evidence = (attn * val_dot).sum(dim=1)
            out_chunks.append(scale * evidence + self.bias)
            if return_evidence:
                top_span_chunks.append(attn.argmax(dim=1))
        logits = torch.cat(out_chunks, dim=1)
        if return_evidence:
            return logits, torch.cat(top_span_chunks, dim=1)
        return logits

    def detached_source_logits(self, pure_logits):
        if self.prior_tau > 0.0 and self.source_prior_delta is not None:
            return pure_logits + self.prior_tau * self.source_prior_delta
        if self.source_bias is None:
            return pure_logits
        return pure_logits.detach() + self.source_bias


class PlmIcdHead(nn.Module):
    """Standard closed-set LAAT / PLM-ICD label-wise attention head (Edin et al., SIGIR 2023)."""

    def __init__(self, note_dim, num_codes, attn_dim=512, dropout=0.1, label_chunk_size=2048):
        super().__init__()
        self.note_dim = int(note_dim)
        self.num_codes = int(num_codes)
        self.attn_dim = int(attn_dim)
        self.label_chunk_size = int(label_chunk_size)
        self.dropout = nn.Dropout(dropout)
        self.first_linear = nn.Linear(self.note_dim, self.attn_dim, bias=False)
        self.second_linear = nn.Linear(self.attn_dim, self.num_codes, bias=False)
        self.classifier_weight = nn.Parameter(torch.empty(self.num_codes, self.note_dim))
        self.classifier_bias = nn.Parameter(torch.full((self.num_codes,), -6.9))
        nn.init.xavier_uniform_(self.first_linear.weight)
        nn.init.xavier_uniform_(self.second_linear.weight)
        nn.init.xavier_uniform_(self.classifier_weight)

    def score_spans(
        self,
        spans,
        span_mask,
        custom_query=None,
        custom_weight=None,
        custom_bias=None,
        label_chunk_size=None,
        return_evidence=False,
    ):
        """Score [B, S, D] spans against either native source weights or CAST-compiled target weights."""
        h = self.dropout(spans)
        z = torch.tanh(self.first_linear(h))
        query = self.second_linear.weight if custom_query is None else custom_query
        weight = self.classifier_weight if custom_weight is None else custom_weight
        bias = self.classifier_bias if custom_bias is None else custom_bias
        query = query.to(device=h.device, dtype=h.dtype)
        weight = weight.to(device=h.device, dtype=h.dtype)
        bias = bias.to(device=h.device, dtype=torch.float32)

        chunk_size = label_chunk_size or self.label_chunk_size
        mask_3d = span_mask[:, :, None]
        out_chunks = []
        top_span_chunks = [] if return_evidence else None
        for start in range(0, len(query), chunk_size):
            q_chunk = query[start:start + chunk_size]
            w_chunk = weight[start:start + chunk_size]
            b_chunk = bias[start:start + chunk_size]
            attn_logits = torch.matmul(z, q_chunk.T).float().masked_fill(~mask_3d, -torch.inf)
            attn = attn_logits.softmax(dim=1).to(h.dtype)
            # Pooled per-code evidence [B, c, D] = attn^T @ h
            pooled = torch.matmul(attn.transpose(1, 2), h)
            logits = (pooled * w_chunk.unsqueeze(0)).sum(dim=-1).float() + b_chunk.unsqueeze(0)
            out_chunks.append(logits)
            if return_evidence:
                top_span_chunks.append(attn.argmax(dim=1))
        logits = torch.cat(out_chunks, dim=1)
        if return_evidence:
            return logits, torch.cat(top_span_chunks, dim=1)
        return logits


class IsoTokenModel(nn.Module):
    """End-to-end clinical note span reader (`mmBERT-base`) paired with IsoTokenHead or PlmIcdHead."""

    def __init__(
        self,
        encoder,
        arch="isotoken",
        label_dim=1024,
        source_size=0,
        span_size=4,
        trainable_encoder_layers=12,
        head_hidden_dim=1024,
        attn_dim=512,
        dropout=0.1,
        label_chunk_size=4096,
        prior_tau=0.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.arch = arch
        self.span_size = int(span_size)
        self.note_dim = int(encoder.config.hidden_size)
        if arch in ("isotoken", "isotoken_unlocked"):
            self.head = IsoTokenHead(
                note_dim=self.note_dim,
                label_dim=label_dim,
                source_size=source_size,
                hidden_dim=head_hidden_dim,
                dropout=dropout,
                label_chunk_size=label_chunk_size,
                prior_tau=prior_tau,
                unlock_label_mlp=(arch == "isotoken_unlocked"),
            )
        elif arch == "plmicd":
            if source_size <= 0:
                raise ValueError("PlmIcdModel requires positive source_size")
            self.head = PlmIcdHead(
                note_dim=self.note_dim,
                num_codes=source_size,
                attn_dim=attn_dim,
                dropout=dropout,
                label_chunk_size=min(label_chunk_size, 2048),
            )
        else:
            raise ValueError(f"Unknown IsoTokenModel arch: {arch}")
        self.trainable_encoder_layers = set_encoder_trainable_layers(self.encoder, trainable_encoder_layers)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        if mode and self.trainable_encoder_layers > 0:
            layers = list(self.encoder.layers)
            for layer in layers[len(layers) - self.trainable_encoder_layers:]:
                layer.train()
            for name in ("final_norm", "norm", "layer_norm"):
                mod = getattr(self.encoder, name, None)
                if mod is not None:
                    mod.train()
        return self

    def encode_spans(self, rows, pad_id, device, special_ids=()):
        chunks = [chunk for row in rows for chunk in row["chunks"]]
        chunk_counts = [len(row["chunks"]) for row in rows]
        ids, mask = pad_sequences(chunks, pad_id, device)
        hidden = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        return pool_note_spans(
            hidden,
            mask,
            chunk_counts,
            span_size=self.span_size,
            input_ids=ids,
            special_ids=special_ids,
        )

    def forward(self, rows, pad_id, device, prototypes=None, special_ids=(), return_calibrated=False, **kwargs):
        spans, span_mask = self.encode_spans(rows, pad_id, device, special_ids=special_ids)
        if self.arch in ("isotoken", "isotoken_unlocked"):
            if prototypes is None:
                raise ValueError(f"IsoTokenModel(arch={self.arch!r}) requires prototypes")
            proto = self.head.prepare_prototypes(prototypes) if (self.arch == "isotoken_unlocked" and self.training) else prototypes
            logits = self.head.score_spans(spans, span_mask, proto, **kwargs)
            if return_calibrated:
                has_cal = (self.head.prior_tau > 0.0 and self.head.source_prior_delta is not None) or (self.head.source_bias is not None)
                cal = self.head.detached_source_logits(logits.float()) if has_cal else None
                return logits, cal
            return logits
        logits = self.head.score_spans(spans, span_mask, **kwargs)
        if return_calibrated:
            return logits, None
        return logits


def build_isotoken_model(
    model_dir,
    arch="isotoken",
    label_dim=1024,
    source_size=0,
    span_size=4,
    trainable_encoder_layers=12,
    head_hidden_dim=1024,
    attn_dim=512,
    dropout=0.1,
    label_chunk_size=4096,
    prior_tau=0.0,
    initialize=True,
):
    encoder = load_encoder(model_dir) if initialize else encoder_from_config(model_dir)
    return IsoTokenModel(
        encoder=encoder,
        arch=arch,
        label_dim=label_dim,
        source_size=source_size,
        span_size=span_size,
        trainable_encoder_layers=trainable_encoder_layers,
        head_hidden_dim=head_hidden_dim,
        attn_dim=attn_dim,
        dropout=dropout,
        label_chunk_size=label_chunk_size,
        prior_tau=prior_tau,
    )
