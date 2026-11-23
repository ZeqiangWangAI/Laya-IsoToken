import torch


def sample_candidates(rows, codes, limit, rng, device, min_negatives=1):
    positives = set().union(*(set(row["labels"]) for row in rows))
    known = {code.key for code in codes}
    if not positives <= known:
        raise ValueError("Training labels are missing from the source codebook")
    if limit <= 0 or limit >= len(codes):
        return list(range(len(codes))), torch.ones(len(codes), device=device)
    included = [i for i, code in enumerate(codes) if code.key in positives]
    remaining = [i for i, code in enumerate(codes) if code.key not in positives]
    count = min(len(remaining), max(limit - len(included), min_negatives))
    sampled = rng.sample(remaining, count)
    indices = included + sampled
    rng.shuffle(indices)
    scale = len(remaining) / count if count else 1.0
    weights = torch.tensor([1.0 if codes[i].key in positives else scale for i in indices], device=device)
    return indices, weights


def merged_labels(codes, targets, count, rng):
    if len(codes) < 2 or count == 0:
        return [], targets[:, :0]
    pairs = [rng.sample(range(len(codes)), 2) for _ in range(count)]
    texts = [f"Any of these conditions applies: ({codes[a].text}) OR ({codes[b].text})" for a, b in pairs]
    union = torch.stack([torch.maximum(targets[:, a], targets[:, b]) for a, b in pairs], dim=1)
    return texts, union

