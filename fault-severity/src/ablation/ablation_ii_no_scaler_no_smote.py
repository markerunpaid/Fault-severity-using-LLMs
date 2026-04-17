# ablation_ii_no_scaler_no_smote.py
"""
ABLATION STUDY — Configuration II: Remove RobustScaler + SMOTE
===============================================================
Pipeline: Raw Metrics (no scaling, no oversampling) + LLM Embeddings + Classifier

What changes vs. full pipeline:
  - step3_scale.py is SKIPPED → metrics NOT RobustScaled
  - apply_smote.py is SKIPPED  → NO oversampling, class imbalance kept as-is
  - train_with_metrics.csv is used directly (output of step2_extract_metrics.py)
  - Everything else (fine-tuned LLM, classifier head) is identical

NOTE: Your existing {model}_results.json files (without SMOTE) used
      train_final.csv which HAS RobustScaling. This config removes BOTH
      the scaler AND SMOTE, giving a separate ablation point.

Output: results/ablation/ablation_ii_no_scaler_no_smote_{model}_results.json
"""

import os, sys, json
import pandas as pd
import numpy as np

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
def run_ablation_ii(model_key):
    cfg    = MODELS[model_key]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*65}")
    print(f"ABLATION II — No RobustScaler + No SMOTE  |  {model_key.upper()}")
    print(f"{'='*65}")

    # Load raw data: metrics extracted but NOT scaled, NOT oversampled
    train_df = pd.read_csv("data/train_with_metrics.csv")
    test_df  = pd.read_csv("data/test_with_metrics.csv")
    print(f"  Train: {len(train_df)}  |  Test: {len(test_df)}")
    print(f"  Class distribution (imbalanced):\n"
          f"{train_df['label'].value_counts().sort_index().to_string()}")

    # Datasets & loaders — NO SMOTE, NO scaling, direct raw metrics
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
    save_path = f"checkpoints/ablation_ii_no_scaler_no_smote_{model_key}.pt"
    history   = train_model(model, train_loader, test_loader, config, device, save_path=save_path)

    # Evaluate
    model.load_state_dict(torch.load(save_path, map_location=device))
    y_true, y_pred, y_prob = evaluate_with_probs(model, test_loader, device)
    metrics = compute_all_metrics(y_true, y_pred, y_prob)

    print(f"\n  Results — Ablation II — {model_key.upper()}")
    for k, v in metrics.items():
        print(f"    {k:<25}: {v:.4f}")
    print(classification_report(y_true, y_pred, labels=[0,1,2,3], zero_division=0,
          target_names=['Critical(0)','Major(1)','Medium(2)','Low(3)']))

    # Save
    out  = {"ablation": "ii_no_scaler_no_smote",
            "description": "No RobustScaler + No SMOTE; raw imbalanced metrics; LLM embeddings + metrics",
            "model": model_key, "model_name": cfg["model_name"], **metrics, "history": history}
    path = f"results/ablation/ablation_ii_no_scaler_no_smote_{model_key}_results.json"
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
    results = {k: run_ablation_ii(k) for k in keys}

    print(f"\n{'='*65}\nABLATION II SUMMARY — No RobustScaler + No SMOTE\n{'='*65}")
    print(f"{'Model':<15} {'Acc':>8} {'MacroF1':>9} {'WtdF1':>9} {'MCC':>8} {'GMean':>8}")
    print("─" * 62)
    for m, r in results.items():
        print(f"{m:<15} {r['Accuracy']:>8.4f} {r['Macro F1']:>9.4f} "
              f"{r['Weighted F1']:>9.4f} {r['MCC']:>8.4f} {r['G-Mean']:>8.4f}")
