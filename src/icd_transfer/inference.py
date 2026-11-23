import time
from contextlib import nullcontext

import torch

from .models import cast_projection, cast_weights, native_prompt, native_scores


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if str(device).startswith("cuda") else nullcontext()


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def label_tokens(tokenizer, texts, max_length):
    tokens = tokenizer(texts, add_special_tokens=True, truncation=False)["input_ids"]
    if any(len(ids) > max_length for ids in tokens):
        raise ValueError("Label text exceeds --label-max-len; definitions must not be silently truncated")
    return tokens


def embed_codebook(model, codes, tokenizer, args):
    tokens = label_tokens(tokenizer, [code.text for code in codes], args.label_max_len)
    vectors = []
    for start in range(0, len(tokens), args.label_batch_size):
        vectors.append(model.encode_labels(tokens[start:start + args.label_batch_size], tokenizer.pad_token_id, args.device))
    return torch.cat(vectors)


@torch.inference_mode()
def compile_codebook(model, kind, codes, source_codes, tokenizer, args):
    model.eval()
    sync(args.device)
    start = time.perf_counter()
    with autocast(args.device):
        if kind == "native":
            compiled = [native_prompt(tokenizer, code.text, args.head_max_len) for code in codes]
        else:
            compiled = embed_codebook(model, codes, tokenizer, args)
    if kind == "linear":
        with autocast(args.device):
            source = embed_codebook(model, source_codes, tokenizer, args)
        projection = cast_projection(source, model.classifier.weight, model.classifier.bias, args.ridge)
        compiled = cast_weights(compiled, projection)
    sync(args.device)
    return compiled, time.perf_counter() - start


@torch.inference_mode()
def predict(model, kind, rows, compiled, tokenizer, args):
    model.eval()
    with autocast(args.device):
        if kind == "native":
            return native_scores(model, rows, compiled, tokenizer.pad_token_id,
                                 args.device, args.pair_batch_size).sigmoid().float()
        notes = model.encode_notes(rows, tokenizer.pad_token_id, args.device)
        if kind == "linear":
            pooled = torch.stack([hidden.mean(0) for hidden, _ in notes]).float()
            return (pooled @ compiled[:, :-1].T + compiled[:, -1]).sigmoid()
        results = [score_chunks(model, hidden, compiled, args.label_batch_size) for hidden, _ in notes]
        return torch.stack(results).sigmoid().float()


def score_chunks(model, hidden, labels, label_batch_size):
    return torch.cat([model.score(hidden, labels[start:start + label_batch_size])
                      for start in range(0, len(labels), label_batch_size)])

