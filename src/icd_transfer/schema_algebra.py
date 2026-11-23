import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import zipfile

import torch
import torch.nn.functional as F


ALGEBRA_MODES = ("none", "frechet_max", "noisy_or", "frechet_clamp")


def noisy_or_logit(member_logits):
    """Exact log-space independent Noisy-OR logit: logit(1 - prod_i (1 - sigmoid(s_i)))."""
    if member_logits.shape[-1] == 0:
        raise ValueError("Noisy-OR requires at least one member")
    x = member_logits.double()
    total = F.softplus(x).sum(dim=-1)
    log_small = torch.logsumexp(x, dim=-1)
    small_val = log_small + 0.5 * torch.exp(log_small.clamp_max(0.0))
    safe_total = total.clamp_min(1e-12)
    normal_val = safe_total + torch.log((-torch.expm1(-safe_total)).clamp_min(1e-300))
    out = torch.where(total < 1e-6, small_val, normal_val)
    return out.to(member_logits.dtype)


def noisy_and_logit(member_logits):
    """Exact log-space independent Noisy-AND logit via De Morgan's law."""
    return -noisy_or_logit(-member_logits)


def _compile_op_edges(entries):
    if not entries:
        return None
    group_pos = [pos for pos, _ in entries]
    edge_slot, edge_member = [], []
    for slot, (_, members) in enumerate(entries):
        edge_slot.extend([slot] * len(members))
        edge_member.extend(members)
    return (
        torch.tensor(group_pos, dtype=torch.long),
        torch.tensor(edge_slot, dtype=torch.long),
        torch.tensor(edge_member, dtype=torch.long),
    )


def compile_group_index(codes, groups):
    """Compile groups into topologically layered flat edge indices for vectorized scatter algebra."""
    positions = {code.key: i for i, code in enumerate(codes)}
    raw = []
    for group in groups:
        op = group["operator"]
        if op not in ("OR", "AND"):
            raise ValueError(f"Unknown composition operator: {op}")
        pos = positions[group["key"]]
        members = [positions[key] for key in group["members"]]
        if not members:
            raise ValueError(f"Group {group['key']} has no members")
        raw.append((pos, op, members))
    # Split into layers only if a later group reads an earlier group's position
    layers = []
    current_or, current_and, written = [], [], set()
    for pos, op, members in raw:
        if any(m in written for m in members):
            layers.append((_compile_op_edges(current_or), _compile_op_edges(current_and)))
            current_or, current_and, written = [], [], set()
        if op == "OR":
            current_or.append((pos, members))
        else:
            current_and.append((pos, members))
        written.add(pos)
    if current_or or current_and:
        layers.append((_compile_op_edges(current_or), _compile_op_edges(current_and)))
    return layers


def _vectorized_or_bounds(out, group_pos, edge_slot, edge_member, negate=False):
    device = out.device
    gpos = group_pos.to(device)
    slot = edge_slot.to(device)
    memb = edge_member.to(device)
    prefix = out.shape[:-1]
    n_groups = len(gpos)
    x = out[..., memb].double()
    if negate:
        x = -x
    slot_exp = slot.view(*([1] * len(prefix)), -1).expand(*prefix, -1)
    max_val = torch.full((*prefix, n_groups), -torch.inf, dtype=torch.float64, device=device)
    max_val.scatter_reduce_(-1, slot_exp, x, reduce="amax", include_self=False)
    total = torch.zeros((*prefix, n_groups), dtype=torch.float64, device=device)
    total.scatter_add_(-1, slot_exp, F.softplus(x))
    shifted_sum = torch.zeros((*prefix, n_groups), dtype=torch.float64, device=device)
    shifted_sum.scatter_add_(-1, slot_exp, torch.exp(x - max_val.gather(-1, slot_exp)))
    log_small = max_val + torch.log(shifted_sum.clamp_min(1e-300))
    small_val = log_small + 0.5 * torch.exp(log_small.clamp_max(0.0))
    safe_total = total.clamp_min(1e-12)
    normal_val = safe_total + torch.log((-torch.expm1(-safe_total)).clamp_min(1e-300))
    noisy = torch.where(total < 1e-6, small_val, normal_val)
    if negate:
        return gpos, (-noisy).to(out.dtype), (-max_val).to(out.dtype)
    return gpos, max_val.to(out.dtype), noisy.to(out.dtype)


def apply_group_algebra(logits, codes, groups, mode="none", compiled_groups=None):
    """Apply zero-training probabilistic set algebra to composed OR/AND group logits."""
    if mode not in ALGEBRA_MODES:
        raise ValueError(f"Unknown group algebra mode: {mode}")
    if mode == "none" or not groups:
        return logits
    layers = compiled_groups if compiled_groups is not None else compile_group_index(codes, groups)
    out = logits.clone()
    with torch.autocast(device_type=out.device.type, enabled=False):
        for or_edges, and_edges in layers:
            if or_edges is not None:
                gpos, lower, upper = _vectorized_or_bounds(out, *or_edges, negate=False)
                if mode == "frechet_max":
                    out[..., gpos] = lower
                elif mode == "noisy_or":
                    out[..., gpos] = upper
                else:
                    out[..., gpos] = out[..., gpos].clamp(min=lower, max=upper)
            if and_edges is not None:
                gpos, lower, upper = _vectorized_or_bounds(out, *and_edges, negate=True)
                if mode == "frechet_max":
                    out[..., gpos] = upper
                elif mode == "noisy_or":
                    out[..., gpos] = lower
                else:
                    out[..., gpos] = out[..., gpos].clamp(min=lower, max=upper)
    return out


def load_who_icd10_hierarchy(who_xml_zip):
    """Parse official WHO ICD-10 ClaML XML into {code: (kind, parent_code, preferred_label)}."""
    with zipfile.ZipFile(who_xml_zip) as archive:
        xml_names = [name for name in archive.namelist() if name.endswith(".xml")]
        if len(xml_names) != 1:
            raise ValueError(f"Expected 1 XML in {who_xml_zip}, found {xml_names}")
        with archive.open(xml_names[0]) as stream:
            root = ET.parse(stream).getroot()
    nodes = {}
    for cls in root.findall(".//Class"):
        code = cls.attrib.get("code", "").strip()
        if not code:
            continue
        kind = cls.attrib.get("kind", "")
        super_el = cls.find("SuperClass")
        parent = super_el.attrib.get("code", "").strip() if super_el is not None else None
        label = ""
        for rubric in cls.findall("Rubric"):
            if rubric.attrib.get("kind") == "preferred":
                lbl = rubric.find("Label")
                if lbl is not None:
                    label = " ".join("".join(lbl.itertext()).split())
                    break
        nodes[code] = (kind, parent, label)
    return nodes


def who_ancestors_for_icd10_diag(raw_code, own_description, nodes):
    """Resolve official WHO ICD-10 ancestor chain [chapter, block, category, ...] for an ICD-10-CM code."""
    clean = raw_code.strip().upper().replace(".", "")
    candidates = []
    if len(clean) > 3:
        candidates.append(f"{clean[:3]}.{clean[3:]}")
        candidates.append(f"{clean[:3]}.{clean[3]}")
    if len(clean) >= 3:
        candidates.append(clean[:3])
    matched = next((cand for cand in candidates if cand in nodes), None)
    if matched is None:
        return []
    chain = []
    seen = set()
    curr = matched
    while curr and curr in nodes and curr not in seen:
        seen.add(curr)
        _kind, parent, label = nodes[curr]
        if label:
            chain.append(label)
        curr = parent
    chain.reverse()
    own_norm = " ".join(own_description.lower().split())
    if chain and " ".join(chain[-1].lower().split()) == own_norm:
        chain.pop()
    return chain


def enrich_icd10_directory_with_who(source_dir, who_xml_zip, output_dir):
    """Create a new target directory with WHO ICD-10 public ancestors attached to icd10_diag codes."""
    src = Path(source_dir)
    dst = Path(output_dir)
    if dst.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {dst}")
    nodes = load_who_icd10_hierarchy(who_xml_zip)
    raw_codes = json.loads((src / "codebook.json").read_text())
    diag_total, diag_enriched, preserved = 0, 0, 0
    enriched = []
    for entry in raw_codes:
        item = dict(entry)
        if item.get("ancestors"):
            preserved += 1
        elif item.get("system") == "icd10_diag":
            diag_total += 1
            ancestors = who_ancestors_for_icd10_diag(item["code"], item.get("description", ""), nodes)
            if ancestors:
                item["ancestors"] = ancestors
                diag_enriched += 1
        enriched.append(item)
    dst.mkdir(parents=True, exist_ok=False)
    for name in ("metadata.json", "groups.json", "train.arrow", "dev.arrow"):
        src_file = src / name
        if src_file.exists():
            shutil.copy2(src_file, dst / name)
    (dst / "codebook.json").write_text(json.dumps(enriched, indent=2, ensure_ascii=False) + "\n")
    summary = {
        "source_dir": str(src),
        "who_xml_zip": str(who_xml_zip),
        "total_codes": len(enriched),
        "diag_codes": diag_total,
        "diag_enriched": diag_enriched,
        "diag_coverage": diag_enriched / max(diag_total, 1),
        "existing_ancestors_preserved": preserved,
        "non_test_only": True,
    }
    (dst / "who_enrichment.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
