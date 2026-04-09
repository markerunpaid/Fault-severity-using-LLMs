# src/full_evaluation.py

import os, json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score
)
import sys
sys.path.insert(0, 'src')
from model import ConcatClsModel
from dataset import BugSeverityDataset

# ── Config ────────────────────────────────────────────────────────────────────
# Fill dropout values from results/{model}_best_params.json
MODELS = {
    "codebert":      ("microsoft/codebert-base",            0.084),
    "graphcodebert": ("microsoft/graphcodebert-base",        0.328),
    "unixcoder":     ("microsoft/unixcoder-base",            0.289),
    "codet5p":       ("Salesforce/codet5p-110m-embedding",   0.31102),  # ← fill dropout
}

MAX_LENGTH   = 256
NUM_WORKERS  = 0
CLASS_NAMES  = ['Critical(0)', 'Major(1)', 'Medium(2)', 'Low(3)']


# ── Tokenizer loader ──────────────────────────────────────────────────────────
def load_tokenizer(model_key, model_name):
    if model_key == "codet5p":
        return AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
    return AutoTokenizer.from_pretrained(model_name)


# ── Helpers ───────────────────────────────────────────────────────────────────
def geometric_mean(y_true, y_pred):
    from sklearn.metrics import recall_score
    r = recall_score(y_true, y_pred, average=None, zero_division=0)
    r = r[r > 0]
    return float(np.prod(r) ** (1.0 / len(r))) if len(r) > 0 else 0.0


def evaluate_with_probs(model, loader, device):
    """Run inference and return labels, predictions, and softmax probs."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            metrics        = batch['metrics'].to(device)
            labels         = batch['label'].to(device)

            logits = model(input_ids, attention_mask, metrics)
            probs  = torch.softmax(logits, dim=-1)
            preds  = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs))


def compute_all_metrics(y_true, y_pred, y_prob):
    """Compute every metric."""

    # ── Overall ───────────────────────────────────────────────────────────────
    accuracy      = accuracy_score(y_true, y_pred)
    prec_macro    = precision_score(y_true, y_pred, average='macro',    zero_division=0)
    prec_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec_macro     = recall_score(y_true, y_pred,    average='macro',    zero_division=0)
    rec_weighted  = recall_score(y_true, y_pred,    average='weighted', zero_division=0)
    f1_macro      = f1_score(y_true, y_pred,        average='macro',    zero_division=0)
    f1_weighted   = f1_score(y_true, y_pred,        average='weighted', zero_division=0)
    mcc           = matthews_corrcoef(y_true, y_pred)
    gmean         = geometric_mean(y_true, y_pred)

    # ── ROC-AUC ───────────────────────────────────────────────────────────────
    try:
        roc_auc_macro    = roc_auc_score(
            y_true, y_prob, multi_class='ovr', average='macro'
        )
        roc_auc_weighted = roc_auc_score(
            y_true, y_prob, multi_class='ovr', average='weighted'
        )
    except Exception as e:
        print(f"  ROC-AUC warning: {e}")
        roc_auc_macro    = 0.0
        roc_auc_weighted = 0.0

    # ── Per-class ─────────────────────────────────────────────────────────────
    prec_per  = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per   = recall_score(y_true, y_pred,    average=None, zero_division=0)
    f1_per    = f1_score(y_true, y_pred,        average=None, zero_division=0)

    auc_per = []
    for c in range(4):
        try:
            binary = (np.array(y_true) == c).astype(int)
            auc_per.append(roc_auc_score(binary, y_prob[:, c]))
        except:
            auc_per.append(0.0)

    return {
        "accuracy":           round(accuracy,       4),
        "precision_macro":    round(prec_macro,      4),
        "precision_weighted": round(prec_weighted,   4),
        "recall_macro":       round(rec_macro,       4),
        "recall_weighted":    round(rec_weighted,    4),
        "f1_macro":           round(f1_macro,        4),
        "f1_weighted":        round(f1_weighted,     4),
        "roc_auc_macro":      round(roc_auc_macro,   4),
        "roc_auc_weighted":   round(roc_auc_weighted,4),
        "mcc":                round(mcc,             4),
        "g_mean":             round(gmean,           4),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": round(float(prec_per[i]), 4) if i < len(prec_per) else 0.0,
                "recall":    round(float(rec_per[i]),  4) if i < len(rec_per)  else 0.0,
                "f1":        round(float(f1_per[i]),   4) if i < len(f1_per)   else 0.0,
                "roc_auc":   round(float(auc_per[i]),  4),
            }
            for i in range(4)
        }
    }


def print_results(model_key, metrics):
    print(f"\n{'='*62}")
    print(f"  {model_key.upper()}")
    print(f"{'='*62}")
    print(f"  {'Metric':<25} {'Macro':>10} {'Weighted':>10}")
    print(f"  {'-'*47}")
    print(f"  {'Accuracy':<25} {metrics['accuracy']:>10.4f}")
    print(f"  {'Precision':<25} {metrics['precision_macro']:>10.4f} {metrics['precision_weighted']:>10.4f}")
    print(f"  {'Recall':<25} {metrics['recall_macro']:>10.4f} {metrics['recall_weighted']:>10.4f}")
    print(f"  {'F1 Score':<25} {metrics['f1_macro']:>10.4f} {metrics['f1_weighted']:>10.4f}")
    print(f"  {'ROC-AUC':<25} {metrics['roc_auc_macro']:>10.4f} {metrics['roc_auc_weighted']:>10.4f}")
    print(f"  {'MCC':<25} {metrics['mcc']:>10.4f}")
    print(f"  {'G-Mean':<25} {metrics['g_mean']:>10.4f}")
    print(f"\n  Per-Class Breakdown:")
    print(f"  {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
    print(f"  {'-'*57}")
    for cls, vals in metrics['per_class'].items():
        print(f"  {cls:<15} {vals['precision']:>10.4f} {vals['recall']:>10.4f} "
              f"{vals['f1']:>10.4f} {vals['roc_auc']:>10.4f}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    test_df     = pd.read_csv("data/test_final.csv")
    all_results = {}

    for model_key, (model_name, dropout) in MODELS.items():
        ckpt_path = f"checkpoints/best_{model_key}.pt"
        if not os.path.exists(ckpt_path):
            print(f"\nSkipping {model_key} — checkpoint not found: {ckpt_path}")
            continue

        print(f"\nLoading {model_key} ({model_name})...")

        tokenizer   = load_tokenizer(model_key, model_name)
        test_ds     = BugSeverityDataset(test_df, tokenizer, MAX_LENGTH)
        test_loader = DataLoader(
            test_ds, batch_size=8,
            shuffle=False, num_workers=NUM_WORKERS
        )

        model = ConcatClsModel(
            model_name   = model_name,
            dropout_prob = dropout
        ).to(device)
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device)
        )

        y_true, y_pred, y_prob = evaluate_with_probs(model, test_loader, device)
        metrics = compute_all_metrics(y_true, y_pred, y_prob)
        all_results[model_key] = metrics

        print_results(model_key, metrics)

        del model
        torch.cuda.empty_cache()

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    with open("results/full_evaluation.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nSaved → results/full_evaluation.json")

    # ── Comparison table ──────────────────────────────────────────────────────
    models_done = list(all_results.keys())
    col_w       = 13

    print(f"\n\n{'='*75}")
    print("FINAL COMPARISON TABLE — ALL MODELS")
    print(f"{'='*75}")

    header = f"{'Metric':<25}" + "".join(f"{m.upper():>{col_w}}" for m in models_done)
    print(header)
    print("-"*75)

    metrics_to_show = [
        ("Accuracy",           "accuracy"),
        ("Precision (Macro)",  "precision_macro"),
        ("Precision (Wtd)",    "precision_weighted"),
        ("Recall (Macro)",     "recall_macro"),
        ("Recall (Wtd)",       "recall_weighted"),
        ("F1 (Macro)",         "f1_macro"),
        ("F1 (Weighted)",      "f1_weighted"),
        ("ROC-AUC (Macro)",    "roc_auc_macro"),
        ("ROC-AUC (Weighted)", "roc_auc_weighted"),
        ("MCC",                "mcc"),
        ("G-Mean",             "g_mean"),
    ]

    for label, key in metrics_to_show:
        row = f"{label:<25}"
        for m in models_done:
            val = all_results.get(m, {}).get(key, 0.0)
            row += f"{val:>{col_w}.4f}"
        print(row)

    print(f"\n{'Per-Class F1':}")
    print(f"{'='*75}")
    header2 = f"{'Class':<20}" + "".join(f"{m.upper():>{col_w}}" for m in models_done)
    print(header2)
    print("-"*75)
    for cls in CLASS_NAMES:
        row = f"{cls:<20}"
        for m in models_done:
            val = all_results.get(m,{}).get("per_class",{}).get(cls,{}).get("f1", 0.0)
            row += f"{val:>{col_w}.4f}"
        print(row)

    print(f"\n{'Per-Class ROC-AUC':}")
    print(f"{'='*75}")
    print(header2)
    print("-"*75)
    for cls in CLASS_NAMES:
        row = f"{cls:<20}"
        for m in models_done:
            val = all_results.get(m,{}).get("per_class",{}).get(cls,{}).get("roc_auc", 0.0)
            row += f"{val:>{col_w}.4f}"
        print(row)

    print(f"\n{'='*75}")
    best_model = max(all_results, key=lambda m: all_results[m]['f1_macro'])
    best_f1    = all_results[best_model]['f1_macro']
    print(f"BEST MODEL: {best_model.upper()}  |  Macro F1 = {best_f1:.4f}")
    print(f"{'='*75}")


if __name__ == "__main__":
    main()