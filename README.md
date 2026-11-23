# Laya-IsoToken: Zero-Training Clinical Codebook Switching via Closed-Form Operator Compilation

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg)](https://www.python.org/)
[![PyTorch 2.9](https://img.shields.io/badge/PyTorch-2.9-EE4C2C.svg)](https://pytorch.org/)

Official implementation of **Laya-IsoToken**, a dual-process clinical coding framework that decouples clinical document understanding from per-code output parameters so hospitals can switch across diagnostic taxonomies (ICD-9 to ICD-10/11), reimbursement groupers (MS-DRG), and languages **in seconds with zero target-labeled clinical notes**.

**Author:** Zeqiang Wang (`zeqiang.wang@surrey.ac.uk`)  
*Department of Computer Science, University of Surrey, Guildford, UK*

---

## Overview

Supervised medical coding architectures bind clinical text encoders to fixed per-code classification matrices (`d × |C|`), forcing costly re-annotation and retraining whenever coding standards evolve. **Laya-IsoToken** resolves this bottleneck through three principles:

1. **Zero-Label-Parameter Span Reader (System 1):** Pairs a fine-tuned multilingual clinical reader (`convaiinnovations/laya-multilingual`, 4-token span pooling) with a **strictly locked offline codebook vectorizer** (`Qwen3-Embedding-0.6B`, `pplx-embed-v1-0.6B`, or `Qwen3-Embedding-4B`) and Bayesian prior deconfounding. Eliminating all 17.4M trainable per-code weights matches or exceeds published supervised baselines on MIMIC-IV ICD-9 while reducing active calibration error by up to 32%.
2. **Closed-Form Codebook Compilation (`0.2–2.6 s`):** Compiles any unseen target codebook (`115` to `185,458` codes) from public descriptions alone via Bures–Wasserstein optimal transport, MMSE Wiener spherical attenuation compensation, whitened CSLS hubness centering, centered semantic prior projection, bounded set algebra, and GPU-vectorized span-grounded non-maximum suppression.
3. **Pre-Deployment Extrapolation Risk & System-2 Verification:** Computes a compile-time subspace residual `ρ_b` before observing any target patient note, routing ambiguous admissions with grounded 4-token spans to offline open-weight LLMs (`Gemma-4-31B-it` / `Qwen3.5-9B`) to eliminate invalid code hallucinations (`16.0% → 0.0%`) while preserving hospital data privacy.

---

## Repository Structure

```text
Laya-IsoToken/
├── src/icd_transfer/          # Core library: IsoToken reader, closed-form compiler, evaluation, cascade
│   ├── isotoken.py            # 4-token span reader & locked/unlocked codebook heads
│   ├── calibration.py         # Bures-Wasserstein OT, Wiener gain, CSLS, prior projection, Span-NMS
│   ├── schema_algebra.py      # Bounded compositional set algebra & WHO ICD-10 enrichment
│   ├── extrapolation.py       # Pre-deployment subspace residual rho_b & risk stratification
│   ├── system2_cascade.py     # Offline System-1 -> System-2 LLM verification cascade
│   ├── zero_shot_retrieval.py # Dense bi-encoder baselines (Qwen3, MedCPT, BioLORD-2023-M)
│   ├── compare.py             # Patient-clustered paired bootstrap & standardized effect sizes (d_z, r_rb)
│   └── cli.py                 # Unified command-line entrypoint (`icd-transfer`)
├── scripts/                   # Benchmark split preparation, BM25/GEMs baselines, and effect-size evaluation
├── tests/                     # 79 unit and integration tests covering math, leakage guards, and CLI
├── licenses/                  # Third-party Apache-2.0 notices (Laya, medical-coding-reproducibility)
├── LICENSE                    # GNU Affero General Public License v3.0 (AGPL-3.0-only)
└── pyproject.toml             # Locked Python 3.12 / PyTorch 2.9 environment specification
```

---

## Installation & Verification

Install [uv](https://docs.astral.sh/uv/) and synchronize the locked environment:

```bash
uv sync --frozen
uv run --frozen python -m unittest discover -s tests -v
uv run --frozen python -m icd_transfer.cli smoke --output /tmp/isotoken-smoke --device cpu
```

---

## End-to-End Reproduction Workflow

> **Data Governance Note:** This repository contains **zero clinical notes, patient identifiers, or MIMIC-derived data**. Credentialed researchers can obtain MIMIC-IV v2.2 and MIMIC-IV-Note v2.2 directly from [PhysioNet](https://physionet.org/content/mimiciv/2.2/). Public multilingual benchmarks are available from Zenodo ([CodiEsp](https://zenodo.org/records/3837305)), BioASQ ([ElCardioCC](http://bioasq.org/)), and CBLUE ([CHIP-CDN](https://tianchi.aliyun.com/dataset/95414)).

### 1. Stage Backbone & Prepare Patient-Disjoint Benchmarks
```bash
uv run --frozen python -m icd_transfer.cli fetch-laya --output artifacts/laya-multilingual --weights
uv run --frozen python -m icd_transfer.cli fetch-source-splits --output data/references
uv run --frozen python scripts/prepare_locked_test_splits.py
```

### 2. Encode Public Codebooks Offline (Once per Taxonomy)
```bash
uv run --frozen python -m icd_transfer.cli encode-semantic-labels \
  --codebook data/stage1/source/train/codebook.json \
  --semantic-model Qwen/Qwen3-Embedding-0.6B \
  --output runs/semantic-qwen3/source-icd9.pt
```

### 3. Train Source Reader Once on MIMIC-IV ICD-9 (4× GPU DDP)
```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 -m icd_transfer.cli train-isotoken \
  --train-data data/stage1/source/train \
  --validation-data data/stage1/source/validation \
  --model-dir artifacts/laya-multilingual \
  --semantic-model Qwen/Qwen3-Embedding-0.6B \
  --semantic-cache runs/semantic-qwen3/source-icd9.pt \
  --arch isotoken --span-size 4 --trainable-encoder-layers 22 --prior-tau 0.5 \
  --seed 42 --output runs/isotoken-qwen3-s42/train
```

### 4. Zero-Training Compilation & Inference on Any Unseen Target Codebook
```bash
uv run --frozen python -m icd_transfer.cli eval-isotoken \
  --checkpoint runs/isotoken-qwen3-s42/train/best.pt \
  --data data/test-locked/icd10-patient-disjoint --split test --test-approved \
  --model-dir artifacts/laya-multilingual \
  --source-semantic-cache runs/semantic-qwen3/source-icd9.pt \
  --target-semantic-cache runs/semantic-qwen3/icd10.pt \
  --operator-policy auto_v1 --record-geometry --record-calibration \
  --output runs/isotoken-qwen3-s42/eval-icd10
```

### 5. Paired Patient-Clustered Bootstrap & Standardized Effect Sizes
```bash
uv run --frozen python -m icd_transfer.cli compare \
  --method runs/isotoken-qwen3-s42/eval-icd10/predictions.jsonl \
  --baseline runs/plmicd-s42/eval-icd10-xwalk/predictions.jsonl \
  --replicates 2000 --output runs/compare-icd10.json
```

---

## License

This project is licensed under the **GNU Affero General Public License v3.0 only (`AGPL-3.0-only`)** — see [`LICENSE`](LICENSE) for the full text and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for upstream Apache-2.0 attributions.
