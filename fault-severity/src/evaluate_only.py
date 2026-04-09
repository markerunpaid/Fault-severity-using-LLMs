# save as src/evaluate_only.py

import torch
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    classification_report, f1_score,
    accuracy_score, matthews_corrcoef
)
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

import sys
sys.path.append('src')
from model import ConcatClsModel
from dataset import BugSeverityDataset, METRIC_COLS
from trainer import evaluate

def geometric_mean(y_true, y_pred, num_classes=4):
    from sklearn.metrics import recall_score
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    return np.prod(recalls) ** (1 / num_classes)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data
    test_df  = pd.read_csv("data/test_with_metrics.csv")
    scaler   = joblib.load("checkpoints/robust_scaler.pkl")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    test_ds = BugSeverityDataset(test_df, tokenizer, scaler, max_length=256)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=2)

    # Load saved model
    model = ConcatClsModel(dropout_prob=0.163).to(device)
    model.load_state_dict(torch.load("checkpoints/best_model.pt", map_location=device))
    print("Model loaded from checkpoints/best_model.pt")

    # Evaluate
    _, y_pred, y_true = evaluate(model, test_loader, device)

    # Report
    present_labels = sorted(list(set(y_true.tolist()) | set(y_pred.tolist())))
    all_names = {0: 'Critical(0)', 1: 'Major(1)', 2: 'Medium(2)', 3: 'Low(3)'}
    present_names = [all_names[l] for l in present_labels]

    print("\n" + "="*60)
    print("FINAL TEST SET RESULTS")
    print("="*60)
    print(classification_report(
        y_true, y_pred,
        labels=present_labels,
        target_names=present_names,
        zero_division=0
    ))
    print(f"Accuracy:    {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1:    {f1_score(y_true, y_pred, average='macro',    zero_division=0):.4f}")
    print(f"Weighted F1: {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"MCC:         {matthews_corrcoef(y_true, y_pred):.4f}")
    print(f"G-Mean:      {geometric_mean(y_true, y_pred):.4f}")

    print("\nClass distribution in test set:")
    unique, counts = np.unique(y_true, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  {all_names[cls]}: {cnt} samples")

if __name__ == "__main__":
    main()