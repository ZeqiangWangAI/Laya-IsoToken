"""Zero-training BM25 lexical matching baseline calibrated strictly on source ICD-9 validation.

Uses pure Python/NumPy inverted-index scoring (zero scipy dependency) across all 5 transfer targets
(locked ICD-10 test, locked MS-DRG test, locked Spanish CodiEsp gold test, Greek ElCardioCC train,
and Chinese CHIP-CDN dev) using zero target clinical notes and zero target labels.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.ipc as ipc

from icd_transfer.codebook import load_codebook
from icd_transfer.transfer_metrics import TransferCounts

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "reports" / "remediation" / "bm25_lexical_baseline.json"
TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _tokenize(text: str) -> list[str]:
    low = text.lower()
    if CJK_RE.search(low):
        chars = [ch for ch in low if "\u4e00" <= ch <= "\u9fff" or ch.isalnum()]
        bigrams = [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
        return chars + bigrams
    return [t for t in TOKEN_RE.findall(low) if len(t) >= 2]


def _build_bm25_inverted_index(codes, k1: float = 1.5, b: float = 0.75):
    """Build inverted postings token -> (code_indices_int32, norm_weights_float32) over public descriptions."""
    doc_toks: list[Counter] = []
    df = Counter()
    lengths = np.zeros(len(codes), dtype=np.float64)
    for i, c in enumerate(codes):
        toks = _tokenize(c.description)
        cnt = Counter(toks)
        doc_toks.append(cnt)
        lengths[i] = max(len(toks), 1)
        for t in cnt:
            df[t] += 1
    n_docs = max(len(codes), 1)
    avgdl = float(lengths.mean()) if len(lengths) else 1.0
    idf = {t: math.log((n_docs - d + 0.5) / (d + 0.5) + 1.0) for t, d in df.items()}

    raw_postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for i, cnt in enumerate(doc_toks):
        dl = lengths[i]
        norm = k1 * (1.0 - b + b * dl / avgdl)
        weights = {}
        tot = 0.0
        for t, tf in cnt.items():
            w = idf[t] * (tf * (k1 + 1.0)) / (tf + norm)
            weights[t] = w
            tot += w
        if tot <= 0.0:
            continue
        inv_tot = 1.0 / tot
        for t, w in weights.items():
            raw_postings[t].append((i, w * inv_tot))

    postings = {}
    for t, items in raw_postings.items():
        idxs = np.array([x[0] for x in items], dtype=np.int32)
        ws = np.array([x[1] for x in items], dtype=np.float32)
        postings[t] = (idxs, ws)
    return postings


def _score_one_note(text: str, postings: dict[str, tuple[np.ndarray, np.ndarray]], n_codes: int) -> np.ndarray:
    scores = np.zeros(n_codes, dtype=np.float32)
    for t in set(_tokenize(text)):
        entry = postings.get(t)
        if entry is not None:
            idxs, ws = entry
            scores[idxs] += ws
    return scores


def _decode_texts_from_arrow(arrow_path: Path, tokenizer, limit: int = 0) -> tuple[list[dict], list[str]]:
    with ipc.open_file(arrow_path) as reader:
        tbl = reader.read_all()
    if limit and tbl.num_rows > limit:
        tbl = tbl.slice(0, limit)
    py_rows = tbl.to_pylist()
    texts = []
    for r in py_rows:
        ids = [tok for ch in r["chunks"] for tok in ch]
        texts.append(tokenizer.decode(ids, skip_special_tokens=True))
    return py_rows, texts


def calibrate_threshold_on_source_val(tokenizer, limit: int = 2000) -> dict:
    src_codes = load_codebook(ROOT / "data/benchmarks/mimiciv-2.2/codebooks/icd9-full.json")
    postings = _build_bm25_inverted_index(src_codes)
    val_dir = ROOT / "data/benchmarks/mimiciv-2.2/clinical-multilingual/source/validation"
    val_arrow = sorted(val_dir.glob("*.arrow"))[0]
    rows, texts = _decode_texts_from_arrow(val_arrow, tokenizer, limit=limit)
    code_to_idx = {c.key: i for i, c in enumerate(src_codes)}
    thrs = np.linspace(0.30, 0.95, 14)
    tp = np.zeros(len(thrs), dtype=np.int64)
    fp = np.zeros(len(thrs), dtype=np.int64)
    fn = np.zeros(len(thrs), dtype=np.int64)
    for row, txt in zip(rows, texts, strict=True):
        truth = {code_to_idx[k] for k in row["labels"] if k in code_to_idx}
        s = _score_one_note(txt, postings, len(src_codes))
        for ti, thr in enumerate(thrs):
            pred = set(np.flatnonzero(s >= thr).tolist())
            tp[ti] += len(pred & truth)
            fp[ti] += len(pred - truth)
            fn[ti] += len(truth - pred)
    f1s = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    best_i = int(np.argmax(f1s))
    return {
        "source_val_notes": len(rows),
        "best_threshold": float(thrs[best_i]),
        "source_val_micro_f1": float(f1s[best_i]),
        "source_val_precision": float(tp[best_i] / max(tp[best_i] + fp[best_i], 1)),
        "source_val_recall": float(tp[best_i] / max(tp[best_i] + fn[best_i], 1)),
    }


def eval_target_bm25(name: str, data_dir: Path, split: str, tokenizer, threshold: float, bench_groups_path: Path | None = None) -> dict:
    meta = json.loads((data_dir / "metadata.json").read_text())
    codes = load_codebook(data_dir / "codebook.json")
    mode = meta.get("output_mode", "multilabel")
    t0 = time.perf_counter()
    postings = _build_bm25_inverted_index(codes)
    compile_s = time.perf_counter() - t0

    arrow_path = data_dir / f"{split}.arrow"
    rows, texts = _decode_texts_from_arrow(arrow_path, tokenizer, limit=0)
    t1 = time.perf_counter()
    counts = TransferCounts([c.key for c in codes], mode)
    top_k = min(50, len(codes))
    n_codes = len(codes)
    for row, txt in zip(rows, texts, strict=True):
        s = _score_one_note(txt, postings, n_codes)
        top_idx = np.argpartition(-s, top_k - 1)[:top_k]
        top_idx = top_idx[np.argsort(-s[top_idx])]
        if mode.startswith("single_label"):
            pred = [codes[int(top_idx[0])].key]
        else:
            pred = [codes[int(i)].key for i in np.flatnonzero(s >= threshold)]
        counts.update(pred, row)
        counts.update_ranking([codes[int(i)].key for i in top_idx], row)
    infer_s = time.perf_counter() - t1
    res = {
        **counts.result(),
        "dataset": name,
        "split": split,
        "threshold": threshold,
        "compile_seconds": compile_s,
        "milliseconds_per_note": 1000.0 * infer_s / max(len(rows), 1),
    }
    if bench_groups_path and bench_groups_path.exists() and mode == "multilabel":
        bg = json.loads(bench_groups_path.read_text())
        kset = {c.key for c in codes}
        res["benchmark_groups"] = {g: counts.label_result([k for k in ks if k in kset]) for g, ks in bg.items()}
    return res


def main() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(ROOT / "artifacts/laya-multilingual/tokenizer"), local_files_only=True)
    print("Calibrating BM25 lexical threshold on MIMIC-IV ICD-9 validation (2,000 notes)...", flush=True)
    src = calibrate_threshold_on_source_val(tok, limit=2000)
    thr = src["best_threshold"]
    print(f"  Source ICD-9 val BM25: thr={thr:.2f}, Micro F1={100*src['source_val_micro_f1']:.2f}%", flush=True)

    targets = [
        ("icd10_test", ROOT / "data/test-locked-20261121/icd10-patient-disjoint", "test", ROOT / "data/benchmarks/mimiciv-2.2/codebooks/icd10-groups.json"),
        ("drg_test", ROOT / "data/test-locked-20261121/drg-patient-disjoint", "test", None),
        ("codiesp_test", ROOT / "data/test-locked-20261121/codiesp", "test", None),
        ("elcardiocc_train", ROOT / "data/transfer/elcardiocc-2026/tokens-laya-multilingual", "train", None),
        ("chipcdn_dev", ROOT / "data/transfer/chip-cdn/tokens-laya-multilingual", "dev", None),
    ]
    out = {"source_calibration": src, "targets": {}}
    for name, path, split, bg in targets:
        print(f"Evaluating BM25 on {name} ({split})...", flush=True)
        r = eval_target_bm25(name, path, split, tok, thr, bg)
        out["targets"][name] = r
        prim = r.get("accuracy") if str(r.get("output_mode", "")).startswith("single_label") else r.get("micro_f1")
        r50 = (r.get("ranking") or {}).get("r_at_50", 0.0)
        bg_s = ""
        if "benchmark_groups" in r:
            bf = r["benchmark_groups"]["benchmark_frequent"]["micro_f1"]
            br = r["benchmark_groups"]["benchmark_rare"]["micro_f1"]
            bg_s = f" (freq={100*bf:.2f}%, rare={100*br:.2f}%)"
        print(f"  {name:18s}: primary={100*(prim or 0):.2f}%{bg_s}, P={100*(r.get('precision') or 0):.2f}%, R={100*(r.get('recall') or 0):.2f}%, R@50={100*r50:.2f}%, {r['milliseconds_per_note']:.1f}ms/note", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
