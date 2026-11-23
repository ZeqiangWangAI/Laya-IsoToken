from dataclasses import dataclass

import torch

from .inference import label_tokens


@dataclass
class LabelViews:
    vectors: torch.Tensor
    available: torch.Tensor
    operators: torch.Tensor
    delta_query: torch.Tensor | None = None
    delta_value: torch.Tensor | None = None

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, indices):
        return LabelViews(
            self.vectors[indices],
            self.available[indices],
            self.operators[indices],
            None if self.delta_query is None else self.delta_query[indices],
            None if self.delta_value is None else self.delta_value[indices],
        )

    def with_residuals(self, delta_query, delta_value):
        return LabelViews(self.vectors, self.available, self.operators, delta_query, delta_value)


def semantic_kind(code):
    if code.system == "source_schema_term":
        return "clinical criterion"
    if code.system.startswith("ms_drg"):
        return "diagnosis-related group"
    kinds = {"diag": "diagnosis", "proc": "procedure", "term": "diagnosis term"}
    return kinds[code.system.rsplit("_", 1)[-1]]


def definition_views(codes, groups):
    compositions = {group["key"]: group for group in groups}
    texts, locations = [], []
    for i, code in enumerate(codes):
        description = compositions[code.key].get("description", "") if code.key in compositions else code.description
        if not description:
            continue
        kind = semantic_kind(code)
        texts.append(f"{kind}: {description}")
        locations.append((i, 0))
        if code.ancestors:
            texts.append(": ".join((kind, *code.ancestors, description)))
            locations.append((i, 1))
    return texts, locations


def membership_matrix(codes, groups, device):
    positions = {code.key: i for i, code in enumerate(codes)}
    edges, weights = [], []
    for row, group in enumerate(groups):
        members = group["members"]
        if not members or len(set(members)) != len(members):
            raise ValueError("Composed labels require distinct, nonempty members")
        edges.extend((row, positions[key]) for key in members)
        weights.extend([1 / len(members)] * len(members))
    indices = torch.tensor(edges, dtype=torch.long, device=device).T
    values = torch.tensor(weights, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, (len(groups), len(codes))).coalesce()


def add_compositions(views, codes, groups):
    if not groups:
        return views
    matrix = membership_matrix(codes, groups, views.vectors.device)
    member_indices = matrix.indices()[1]
    if not views.available[member_indices, 0].all():
        raise ValueError("Composition members must have public atomic definitions")
    positions = {code.key: i for i, code in enumerate(codes)}
    rows = [positions[group["key"]] for group in groups]
    with torch.autocast(device_type=matrix.device.type, enabled=False):
        pooled = torch.sparse.mm(matrix, views.vectors[:, 0].float())
    views.vectors[rows, 2] = pooled.to(views.vectors.dtype)
    views.available[rows, 2] = True
    operators = {"OR": 1, "AND": 2}
    views.operators[rows] = torch.tensor([operators[group["operator"]] for group in groups],
                                        device=views.operators.device)
    return views


def encode_views(model, codes, groups, tokenizer, args):
    texts, locations = definition_views(codes, groups)
    tokens = label_tokens(tokenizer, texts, args.label_max_len)
    embeddings = [model.encode_labels(tokens[start:start + args.label_batch_size], tokenizer.pad_token_id, args.device)
                  for start in range(0, len(tokens), args.label_batch_size)]
    encoded = torch.cat(embeddings)
    vectors = encoded.new_zeros((len(codes), 3, encoded.shape[-1]))
    available = torch.zeros((len(codes), 3), dtype=torch.bool, device=args.device)
    rows, columns = zip(*locations)
    vectors[list(rows), list(columns)] = encoded
    available[list(rows), list(columns)] = True
    operators = torch.zeros(len(codes), dtype=torch.long, device=args.device)
    views = add_compositions(LabelViews(vectors, available, operators), codes, groups)
    if not views.available.any(-1).all():
        raise ValueError("Every target label needs a definition or a public composition")
    return views
