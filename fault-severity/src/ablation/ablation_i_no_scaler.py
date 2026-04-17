# ablation_i_no_scaler.py
"""
ABLATION STUDY — Configuration I: Remove RobustScaler Only
============================================================
Pipeline: Raw Metrics (no scaling) + SMOTE + LLM Embeddings + Classifier

What changes vs. full pipeline:
  - step3_scale.py is SKIPPED → metrics are NOT RobustScaled
  - train_with_metrics.csv is used directly (output of step2_extract_metrics.py)
  - SMOTE is still applied on the raw (unscaled) metrics
  - Everything else (fine-tuned LLM, classifier head) is identical

Output: results/ablation/ablation_i_no_scaler_{model}_results.json
"""

import os, sys, json
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    recall_score, precision_score, roc_auc_score,
    classification_report
)
from model import ConcatClsModel
from dataset import BugSeverityDataset
from trainer import train_model

# ── Metric columns ────────────────────────────────────────────────────────────
METRIC_COLS = [
    'SLOC', 'CyclomaticComplexity', 'McClureComplexity',
    'NestingDepth', 'ProxyIndentation', 'Readability',
    'FanOut', 'HalsteadDifficulty', 'HalsteadEffort',
    'MaintainabilityIndex'
]

# ── All 4 models with Optuna-tuned hyperparameters ────────────────────────────
MODELS = {
    "codebert": {
        "model_name":   "microsoft/codebert-base",
        "dropout":      0.08416125758393399,
        "lr":           4.404220810269438e-05,
        "weight_decay": 0.0002501181112418235,
        "warmup_ratio": 0.2968821991330362,
    },
    "graphcodebert": {
        "model_name":   "microsoft/graphcodebert-base",
        "dropout":      0.327789944814619,
        "lr":           7.928758212088177e-05,
        "weight_decay": 0.00029499719873019855,
        "warmup_ratio": 0.14522901132382177,
    },
    "unixcoder": {
        "model_name":   "microsoft/unixcoder-base",
        "dropout":      0.28948155927925495,
        "lr":           5.371004924854035e-05,
        "weight_decay": 1.6536937182824424e-05,
        "warmup_ratio": 0.08418523990223437,
    },
    "codet5p": {
        "model_name":   "Salesforce/codet5p-110m-embedding",
        "dropout":      0.31101859897600825,
        "lr":           4.8030133858163695e-05,
        "weight_decay": 2.3360213139915578e-05,
        "warmup_ratio": 0.10463821699577333,
    },
}

BATCH_SIZE         = 4
MAX_LENGTH         = 256
ACCUMULATION_STEPS = 4
NUM_WORKERS        = 0
EPOCHS             = 8

os.makedirs("results/ablation", exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def geometric_mean(y_true, y_pred):
    r = recall_score(y_true, y_pred, average=None, zero_division=0)
    r = r[r > 0]
    return float(np.prod(r) ** (1.0 / len(r))) if len(r) > 0 else 0.0


def compute_all_metrics(y_true, y_pred, y_prob):
    out = {
        "Accuracy":          round(accuracy_score(y_true, y_pred), 4),
        "Macro F1":          round(f1_score(y_true, y_pred, average='macro',    zero_division=0), 4),
        "Weighted F1":       round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        "MCC":               round(matthews_corrcoef(y_true, y_pred), 4),
        "G-Mean":            round(geometric_mean(y_true, y_pred), 4),
        "Precision (Macro)": round(precision_score(y_true, y_pred, average='macro',    zero_division=0), 4),
        "Recall (Macro)":    round(recall_score(y_true, y_pred,    average='macro',    zero_division=0), 4),
    }
    try:
        out["ROC-AUC (Macro)"] = round(
            roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro'), 4)
    except Exception:
        out["ROC-AUC (Macro)"] = 0.0
    return out


def load_tokenizer(model_key, model_name):
    if model_key == "codet5p":
        return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return AutoTokenizer.from_pretrained(model_name)


def apply_smote_on_df(train_df):
    """Apply SMOTE on raw (unscaled) metrics — 60% of majority class strategy."""
    X = train_df[METRIC_COLS].fillna(0.0).values
    y = train_df['label'].values

    original_counts   = dict(Counter(y))
    majority_count    = max(original_counts.values())
    target            = int(majority_count * 0.60)
    sampling_strategy = {cls: target for cls, cnt in original_counts.items() if cnt < target}

    print(f"  Before SMOTE : {dict(Counter(y))}")
    print(f"  Strategy     : {sampling_strategy}")

    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=5)
    try:
        X_res, y_res = smote.fit_resample(X, y)
    except ValueError:
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=3)
        X_res, y_res = smote.fit_resample(X, y)

    print(f"  After SMOTE  : {dict(Counter(y_res))}")

    n_orig       = len(train_df)
    n_synth      = len(y_res) - n_orig
    method_codes = list(train_df['method_code'].values) + [''] * n_synth

    df_out = pd.DataFrame(X_res, columns=METRIC_COLS)
    df_out['label']       = y_res
    df_out['method_code'] = method_codes
    return df_out[['method_code', 'label'] + METRIC_COLS]


def evaluate_with_probs(model, loader, device):
    model.eval()
    preds_, labels_, probs_ = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
                batch['metrics'].to(device),
            )
            probs_.extend(torch.softmax(logits, dim=-1).cpu().numpy())
            preds_.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            labels_.extend(batch['label'].numpy())
    return np.array(labels_), np.array(preds_), np.array(probs_)


# ── Per-model runner ──────────────────────────────────────────────────────────
def run_ablation_i(model_key):
    cfg    = MODELS[model_key]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*65}")
    print(f"ABLATION I — No RobustScaler  |  {model_key.upper()}")
    print(f"{'='*65}")

    # Load raw (unscaled) metrics — output of step2, BEFORE step3_scale.py
    train_raw = pd.read_csv("data/train_with_metrics.csv")
    test_df   = pd.read_csv("data/test_with_metrics.csv")
    print(f"  Train (raw): {len(train_raw)}  |  Test: {len(test_df)}")

    # SMOTE on raw metrics
    train_df = apply_smote_on_df(train_raw)
    print(f"  Train (after SMOTE): {len(train_df)}")

    # Datasets & loaders
    tokenizer    = load_tokenizer(model_key, cfg["model_name"])
    train_loader = DataLoader(
        BugSeverityDataset(train_df, tokenizer, MAX_LENGTH),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(
        BugSeverityDataset(test_df, tokenizer, MAX_LENGTH),
        batch_size=8, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Model & training
    model  = ConcatClsModel(model_name=cfg["model_name"], dropout_prob=cfg["dropout"]).to(device)
    config = {
        'learning_rate': cfg["lr"], 'weight_decay': cfg["weight_decay"],
        'warmup_ratio':  cfg["warmup_ratio"], 'dropout': cfg["dropout"],
        'epochs': EPOCHS, 'accumulation_steps': ACCUMULATION_STEPS,
    }
    save_path = f"checkpoints/ablation_i_no_scaler_{model_key}.pt"
    history   = train_model(model, train_loader, test_loader, config, device, save_path=save_path)

    # Evaluate
    model.load_state_dict(torch.load(save_path, map_location=device))
    y_true, y_pred, y_prob = evaluate_with_probs(model, test_loader, device)
    metrics = compute_all_metrics(y_true, y_pred, y_prob)

    print(f"\n  Results — Ablation I — {model_key.upper()}")
    for k, v in metrics.items():
        print(f"    {k:<25}: {v:.4f}")
    print(classification_report(y_true, y_pred, labels=[0,1,2,3], zero_division=0,
          target_names=['Critical(0)','Major(1)','Medium(2)','Low(3)']))

    # Save
    out  = {"ablation": "i_no_scaler",
            "description": "No RobustScaler; SMOTE on raw metrics; LLM embeddings + metrics",
            "model": model_key, "model_name": cfg["model_name"], **metrics, "history": history}
    path = f"results/ablation/ablation_i_no_scaler_{model_key}_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved → {path}")

    del model; torch.cuda.empty_cache()
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        choices=list(MODELS.keys()) + ["all"], default=["all"])
    args    = parser.parse_args()
    keys    = list(MODELS.keys()) if "all" in args.models else args.models
    results = {k: run_ablation_i(k) for k in keys}

    print(f"\n{'='*65}\nABLATION I SUMMARY — No RobustScaler\n{'='*65}")
    print(f"{'Model':<15} {'Acc':>8} {'MacroF1':>9} {'WtdF1':>9} {'MCC':>8} {'GMean':>8}")
    print("─" * 62)
    for m, r in results.items():
        print(f"{m:<15} {r['Accuracy']:>8.4f} {r['Macro F1']:>9.4f} "
              f"{r['Weighted F1']:>9.4f} {r['MCC']:>8.4f} {r['G-Mean']:>8.4f}")
