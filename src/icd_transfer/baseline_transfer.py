import torch

from .inference import embed_codebook, score_chunks
from .label_views import membership_matrix
from .models import cast_projection, cast_weights, native_prompt, native_scores


def compile_baseline(model, kind, codes, source_codes, groups, tokenizer, args):
    positions = {code.key: i for i, code in enumerate(codes)}
    composed = {group["key"] for group in groups}
    atomic = [i for i, code in enumerate(codes) if code.key not in composed]
    if any(key in composed for group in groups for key in group["members"]):
        raise ValueError("Baseline compositions require atomic members")
    if kind == "native":
        prompts = [native_prompt(tokenizer, codes[i].text, args.head_max_len) for i in atomic]
        return {"prompts": prompts, "positions": atomic, "size": len(codes),
                "groups": [(positions[group["key"]], group["operator"],
                            [positions[key] for key in group["members"]]) for group in groups]}
    embeddings = embed_codebook(model, [codes[i] for i in atomic], tokenizer, args)
    vectors = embeddings.new_zeros((len(codes), embeddings.shape[-1]))
    vectors[atomic] = embeddings
    if groups:
        matrix = membership_matrix(codes, groups, args.device)
        with torch.autocast(device_type=matrix.device.type, enabled=False):
            pooled = torch.sparse.mm(matrix, vectors.float())
        vectors[[positions[group["key"]] for group in groups]] = pooled.to(vectors.dtype)
    if kind != "linear":
        return vectors
    source = embed_codebook(model, source_codes, tokenizer, args)
    with torch.autocast(device_type=source.device.type, enabled=False):
        projection = cast_projection(source, model.classifier.weight, model.classifier.bias, args.ridge)
        return cast_weights(vectors, projection)


def baseline_logits(model, kind, rows, compiled, tokenizer, args):
    if kind == "native":
        return native_transfer_logits(model, rows, compiled, tokenizer, args)
    notes = model.encode_notes(rows, tokenizer.pad_token_id, args.device)
    if kind == "linear":
        pooled = torch.stack([hidden.mean(0) for hidden, _ in notes]).float()
        return pooled @ compiled[:, :-1].T + compiled[:, -1]
    return torch.stack([score_chunks(model, hidden, compiled, args.label_batch_size) for hidden, _ in notes])


def native_transfer_logits(model, rows, compiled, tokenizer, args):
    prompts = compiled["prompts"]
    logits = torch.empty((len(rows), compiled["size"]), device=args.device)
    for start in range(0, len(prompts), args.label_batch_size):
        stop = start + args.label_batch_size
        scores = native_scores(model, rows, prompts[start:stop], tokenizer.pad_token_id,
                               args.device, args.pair_batch_size)
        logits[:, compiled["positions"][start:stop]] = scores
    for position, operator, members in compiled["groups"]:
        if operator not in ("OR", "AND"):
            raise ValueError("Unknown composition operator")
        values = logits[:, members]
        logits[:, position] = values.amax(-1) if operator == "OR" else values.amin(-1)
    return logits
