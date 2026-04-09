# src/train_final.py
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, f1_score,
    accuracy_score, matthews_corrcoef
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model import ConcatClsModel
from dataset import BugSeverityDataset
from trainer import train_model, evaluate

# ── RTX 3050 6GB settings ────────────────────────────────────────────────────
BATCH_SIZE         = 4
ACCUMULATION_STEPS = 4      # effective batch = 16
MAX_LENGTH         = 256
NUM_WORKERS        = 2

# ── Paper's best hyperparameters ─────────────────────────────────────────────
BEST_CONFIG = {
    'learning_rate':      4.82e-5,
    'weight_decay':       1.0e-4,
    'warmup_ratio':       0.286,
    'dropout':            0.163,
    'epochs':             8,
    'accumulation_steps': ACCUMULATION_STEPS,
}

MODEL_NAME = "microsoft/codebert-base"

def geometric_mean(y_true, y_pred):
    from sklearn.metrics import recall_score
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    recalls = recalls[recalls > 0]
    return float(np.prod(recalls) ** (1.0 / len(recalls))) if len(recalls) > 0 else 0.0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load freshly processed data ──────────────────────────────────────────
    train_df = pd.read_csv("data/train_final.csv")
    test_df  = pd.read_csv("data/test_final.csv")

    print(f"\nTrain: {len(train_df)} | Test: {len(test_df)}")
    print(f"Train class distribution:\n{train_df['label'].value_counts().sort_index()}")
    print(f"Test class distribution:\n{test_df['label'].value_counts().sort_index()}")

    # ── Tokenizer & Datasets ─────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = BugSeverityDataset(train_df, tokenizer, MAX_LENGTH)
    test_ds  = BugSeverityDataset(test_df,  tokenizer, MAX_LENGTH)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,  batch_size=8,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = ConcatClsModel(
        model_name   = MODEL_NAME,
        dropout_prob = BEST_CONFIG['dropout']
    ).to(device)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Train ────────────────────────────────────────────────────────────────
    history = train_model(
        model, train_loader, test_loader,
        BEST_CONFIG, device,
        save_path="checkpoints/best_codebert.pt"
    )

    # ── Evaluate ─────────────────────────────────────────────────────────────
    model.load_state_dict(
        torch.load("checkpoints/best_codebert.pt", map_location=device)
    )
    _, y_pred, y_true = evaluate(model, test_loader, device)

    present_labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    all_names      = {0:'Critical(0)', 1:'Major(1)', 2:'Medium(2)', 3:'Low(3)'}
    present_names  = [all_names[l] for l in present_labels]

    print("\n" + "="*60)
    print("FINAL TEST RESULTS — CodeBERT")
    print("="*60)
    print(classification_report(-
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

if __name__ == "__main__":
    main()