# ABLATION STUDY — Instructions & Pipeline Guide

## Overview

Four ablation configurations measure the contribution of each pipeline component
across all 4 LLMs: CodeBERT, GraphCodeBERT, UniXcoder, CodeT5+.

```
Full Pipeline:  train_raw ─→ metrics extraction ─→ RobustScaler ─→ SMOTE ─→ LLM + classifier
                                                          ↑              ↑          ↑
                                                    Ablation I      Abl. I+II  Ablation III/IV
```

### Component Matrix

| Configuration | RobustScaler | SMOTE | Code Metrics | LLM Embeddings |
|---|:---:|:---:|:---:|:---:|
| **Full Pipeline (w/ SMOTE)** ← already done | ✓ | ✓ | ✓ | ✓ |
| **Full Pipeline (no SMOTE)** ← already done | ✓ | ✗ | ✓ | ✓ |
| **I: No Scaler** | ✗ | ✓ | ✓ | ✓ |
| **II: No Scaler + No SMOTE** | ✗ | ✗ | ✓ | ✓ |
| **III: No Metrics** | ✓ | ✓ | ✗ | ✓ |
| **IV: Source Code Only (Mashhadi RQ2)** | ✗ | ✗ | ✗ | ✓ |

### Models
All 4 scripts run across: `codebert`, `graphcodebert`, `unixcoder`, `codet5p`

---

## Prerequisites — Required Data Files

| File | Created by | Used in |
|---|---|---|
| `data/train_raw.csv` | `step1_preprocess.py` | Ablation IV |
| `data/test_raw.csv` | `step1_preprocess.py` | Ablation IV |
| `data/train_with_metrics.csv` | `step2_extract_metrics.py` | Ablation I, II |
| `data/test_with_metrics.csv` | `step2_extract_metrics.py` | Ablation I, II |
| `data/train_smote.csv` | `apply_smote.py` | Ablation III |
| `data/test_final.csv` | `step3_scale.py` | Ablation III |

If any are missing:
```bash
python src/step2_extract_metrics.py    # generates train/test_with_metrics.csv
python src/apply_smote.py              # generates train_smote.csv
```

---

## Execution

### Run all 4 ablation configurations (all models each)

```bash
# Ablation I — No RobustScaler
python ablation_i_no_scaler.py --models all

# Ablation II — No RobustScaler + No SMOTE
python ablation_ii_no_scaler_no_smote.py --models all

# Ablation III — No Code Metrics (embeddings only)
python ablation_iii_no_metrics.py --models all

# Ablation IV — Source Code Only (Mashhadi et al. RQ2 style)
python ablation_iv_source_code_only.py --models all

# Summary table + contribution analysis
python ablation_summary.py
```

### Run single model only
```bash
python ablation_i_no_scaler.py --models codebert
python ablation_iii_no_metrics.py --models graphcodebert unixcoder
```

---

## What Each Script Does

### `ablation_i_no_scaler.py`
- Loads `data/train_with_metrics.csv` (metrics extracted, NOT yet scaled)
- Applies SMOTE **on raw unscaled metrics** using same 60%-of-majority strategy
- Trains `ConcatClsModel` (768 + 10 = 778 → 4): same architecture as full pipeline
- **Isolates:** contribution of RobustScaler

### `ablation_ii_no_scaler_no_smote.py`
- Loads `data/train_with_metrics.csv` directly, NO scaling, NO SMOTE
- Raw, imbalanced class distribution preserved
- Same `ConcatClsModel` architecture
- **Isolates:** combined contribution of RobustScaler + SMOTE together

> Note: Your existing `{model}_results.json` (no SMOTE) used `train_final.csv`
> which HAS RobustScaling. Ablation II removes both, so it's a distinct point.

### `ablation_iii_no_metrics.py`
- Loads `data/train_smote.csv` — full preprocessing (scaler + SMOTE) kept
- Uses `EmbeddingOnlyModel`: classifier is `Linear(768, 4)` (NOT 778)
- Metrics are loaded by DataLoader for API compat but **completely ignored** by model
- **Isolates:** contribution of the 10 code metrics

### `ablation_iv_source_code_only.py`
- Loads `data/train_raw.csv` and `data/test_raw.csv` (only `method_code` + `label`)
- No metric extraction, no scaling, no SMOTE
- Uses `SourceCodeOnlyModel`: `Linear(768, 4)` — purely LLM-based classification
- **Directly replicates** Mashhadi et al. RQ2
- **Isolates:** combined contribution of ALL proposed components

---

## Output Files

```
results/
  {model}_smote_results.json           ← Full pipeline (already done)
  {model}_results.json                 ← Full pipeline no SMOTE (already done)
  ablation/
    ablation_i_no_scaler_{model}_results.json          × 4 models
    ablation_ii_no_scaler_no_smote_{model}_results.json × 4 models
    ablation_iii_no_metrics_{model}_results.json        × 4 models
    ablation_iv_source_code_only_{model}_results.json   × 4 models
    ablation_summary.csv                               ← unified table

checkpoints/
    ablation_i_no_scaler_{model}.pt
    ablation_ii_no_scaler_no_smote_{model}.pt
    ablation_iii_no_metrics_{model}.pt
    ablation_iv_source_code_only_{model}.pt
```

Full pipeline checkpoints (`best_{model}.pt`) are **never overwritten**.

---

## Hyperparameters

All ablation runs reuse **Optuna-tuned hyperparameters** from the full pipeline.
This is intentional — we isolate component contributions, not retune for each config.

| Model | LR | Dropout | Warmup |
|---|---|---|---|
| CodeBERT | 4.40e-05 | 0.0842 | 0.2969 |
| GraphCodeBERT | 7.93e-05 | 0.3278 | 0.1452 |
| UniXcoder | 5.37e-05 | 0.2895 | 0.0842 |
| CodeT5+ | 4.80e-05 | 0.3110 | 0.1046 |
