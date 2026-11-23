import json
import os
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from .codebook import load_codebook
from .label_views import membership_matrix
from .models import encode_labels as encode_laya_labels, load_encoder


def resolve_local_semantic_dir(model_spec):
    """Resolve a local path or cached HF snapshot directory offline without network calls."""
    direct = Path(model_spec)
    if direct.exists():
        return direct
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    slug = "models--" + str(model_spec).replace("/", "--")
    snaps = sorted((hf_home / "hub" / slug / "snapshots").glob("*"))
    if snaps:
        return snaps[-1]
    raise FileNotFoundError(
        f"Semantic encoder '{model_spec}' not found locally or under {hf_home / 'hub' / slug}"
    )


def _infer_pooling(model_dir, pooling):
    if pooling != "auto":
        return pooling
    name = str(model_dir).lower()
    if "qwen" in name and "embed" in name:
        return "last_token"
    if "medcpt" in name:
        return "cls"
    return "mean"


def _pool_cls(hidden):
    return hidden[:, 0]


def _pool_last_token(hidden, mask):
    # Works for both left-padded and right-padded batches.
    left_padded = bool((mask[:, -1].sum() == len(mask)).item())
    if left_padded:
        return hidden[:, -1]
    lengths = mask.sum(dim=1).clamp_min(1) - 1
    batch = torch.arange(len(hidden), device=hidden.device)
    return hidden[batch, lengths]


def _pool_mean(hidden, mask):
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def pool_semantic_hidden(hidden, mask, pool_kind):
    if pool_kind in ("last_token", "last"):
        return _pool_last_token(hidden, mask)
    if pool_kind == "cls":
        return _pool_cls(hidden)
    return _pool_mean(hidden, mask)


def load_semantic_backbone(model_spec, device, pooling="auto", dtype=torch.bfloat16):
    """Load an offline frozen semantic encoder + tokenizer + resolved pooling mode."""
    model_dir = resolve_local_semantic_dir(model_spec)
    dev = torch.device(device)
    if (model_dir / "model.safetensors").exists() and (model_dir / "encoder").is_dir():
        from .data import load_tokenizer

        tokenizer = load_tokenizer(model_dir)
        model = load_encoder(model_dir).to(dev).eval()
        return tokenizer, model, "mean"
    pool_kind = _infer_pooling(model_dir, pooling)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if pool_kind == "last_token":
        tokenizer.padding_side = "left"
    load_dtype = dtype if dev.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
        dtype=load_dtype,
    ).to(dev).eval()
    return tokenizer, model, pool_kind


def code_semantic_text(code, text_mode="description"):
    if text_mode == "description":
        return code.description.strip()
    if text_mode == "full":
        return code.text.strip()
    raise ValueError(f"Unknown semantic text_mode: {text_mode}")


@torch.inference_mode()
def encode_texts_hf(
    texts,
    model_spec,
    device,
    batch_size=256,
    max_length=128,
    pooling="auto",
    normalize=True,
    dtype=torch.bfloat16,
):
    """Encode label strings with an offline frozen HuggingFace semantic encoder (e.g. Qwen3-Embedding-0.6B)."""
    model_dir = resolve_local_semantic_dir(model_spec)
    dev = torch.device(device)
    # Support existing laya-multilingual layout (model.safetensors + encoder/ + tokenizer/) for local tests/ablations.
    if (model_dir / "model.safetensors").exists() and (model_dir / "encoder").is_dir():
        from .data import load_tokenizer
        from .inference import autocast, label_tokens

        tokenizer = load_tokenizer(model_dir)
        encoder = load_encoder(model_dir).to(dev).eval()
        tokens = label_tokens(tokenizer, texts, max_length)
        chunks = []
        with autocast(dev):
            for start in range(0, len(tokens), batch_size):
                vec = encode_laya_labels(encoder, tokens[start:start + batch_size], tokenizer.pad_token_id, dev)
                chunks.append(vec.float())
        out = torch.cat(chunks, dim=0)
        return F.normalize(out, dim=-1, eps=1e-6) if normalize else out

    pool_kind = _infer_pooling(model_dir, pooling)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if pool_kind == "last_token":
        tokenizer.padding_side = "left"
    load_dtype = dtype if dev.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
        dtype=load_dtype,
    ).to(dev).eval()
    max_pos = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    eff_max_len = min(max_length, max(int(max_pos) - 2, 32)) if max_pos is not None else max_length
    chunks = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=eff_max_len,
            return_tensors="pt",
        ).to(dev)
        out = model(**enc)
        hidden = out.last_hidden_state
        mask = enc["attention_mask"]
        pooled = pool_semantic_hidden(hidden, mask, pool_kind)
        chunks.append(pooled.float())
    vectors = torch.cat(chunks, dim=0)
    return F.normalize(vectors, dim=-1, eps=1e-6) if normalize else vectors


@torch.inference_mode()
def encode_codebook_semantic(
    codes,
    compositions,
    model_spec,
    device,
    text_mode="description",
    batch_size=256,
    max_length=128,
    pooling="auto",
):
    """Compile semantic vectors [|C|, d_z] for atomic and composed (OR/AND) codes."""
    composed_keys = {group["key"] for group in compositions}
    positions = {code.key: idx for idx, code in enumerate(codes)}
    atomic_idx = [idx for idx, code in enumerate(codes) if code.key not in composed_keys]
    if not atomic_idx:
        raise ValueError("Codebook must contain at least one atomic code")
    atomic_texts = [code_semantic_text(codes[idx], text_mode=text_mode) for idx in atomic_idx]
    atomic_vecs = encode_texts_hf(
        atomic_texts,
        model_spec=model_spec,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        pooling=pooling,
        normalize=True,
    )
    dim = atomic_vecs.shape[-1]
    vectors = atomic_vecs.new_zeros((len(codes), dim))
    vectors[atomic_idx] = atomic_vecs
    if compositions:
        mat = membership_matrix(codes, compositions, atomic_vecs.device)
        with torch.autocast(device_type=mat.device.type, enabled=False):
            pooled = torch.sparse.mm(mat, vectors.float())
            pooled = F.normalize(pooled, dim=-1, eps=1e-6)
        group_idx = [positions[group["key"]] for group in compositions]
        vectors[group_idx] = pooled.to(vectors.dtype)
    return vectors


def save_semantic_cache(path, codes, vectors, model_spec, text_mode="description", pooling="auto"):
    payload = {
        "keys": [code.key for code in codes],
        "vectors": vectors.detach().cpu().float(),
        "model_spec": str(model_spec),
        "text_mode": text_mode,
        "pooling": pooling,
        "dim": int(vectors.shape[-1]),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def load_semantic_cache(path, codes, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = [code.key for code in codes]
    if payload["keys"] != expected:
        raise ValueError("Cached semantic label vectors do not match codebook key order")
    meta = {k: v for k, v in payload.items() if k not in ("vectors", "keys")}
    return payload["vectors"].to(device=device, dtype=torch.float32), meta


def encode_semantic_labels_cli(args):
    codebook_path = Path(args.codebook)
    codes = load_codebook(codebook_path)
    group_path = Path(args.groups) if args.groups else (codebook_path.parent / "groups.json")
    compositions = json.loads(group_path.read_text()) if group_path.exists() else []
    vectors = encode_codebook_semantic(
        codes,
        compositions,
        model_spec=args.semantic_model,
        device=args.device,
        text_mode=args.text_mode,
        batch_size=args.batch_size,
        max_length=args.max_length,
        pooling=args.pooling,
    )
    payload = save_semantic_cache(
        args.output,
        codes,
        vectors,
        model_spec=args.semantic_model,
        text_mode=args.text_mode,
        pooling=args.pooling,
    )
    print(json.dumps({
        "output": str(args.output),
        "codes": len(codes),
        "compositions": len(compositions),
        "dim": payload["dim"],
        "semantic_model": payload["model_spec"],
        "text_mode": payload["text_mode"],
    }))
