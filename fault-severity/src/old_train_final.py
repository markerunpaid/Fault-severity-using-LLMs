import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    classification_report, f1_score,
    accuracy_score, matthews_corrcoef
)
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
import joblib

from model import ConcatClsModel
from dataset import BugSeverityDataset, METRIC_COLS
from trainer import train_model, evaluate

# ── GPU Memory optimized settings for RTX 3050 6GB ──────────────────────────
BATCH_SIZE         = 4      # fits in 6GB VRAM
ACCUMULATION_STEPS = 4      # effective batch = 4×4 = 16 (same as paper)
MAX_LENGTH         = 256    # biggest VRAM saver; minor accuracy trade-off
NUM_WORKERS        = 2      # safe for laptop

BEST_CONFIG = {
    'learning_rate':    4.82e-5,
    'weight_decay':     1.0e-4,
    'warmup_ratio':     0.286,
    'dropout':          0.163,
    'epochs':           8,
    'accumulation_steps': ACCUMULATION_STEPS,
}

def geometric_mean(y_true, y_pred, num_classes=4):
    from sklearn.metrics import recall_score
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    return np.prod(recalls) ** (1 / num_classes)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load data ────────────────────────────────────────────────────────────
    train_df = pd.read_csv("data/train_with_metrics.csv")
    test_df  = pd.read_csv("data/test_with_metrics.csv")
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")

    # ── Fit RobustScaler on training data only ───────────────────────────────
    scaler = RobustScaler()
    scaler.fit(train_df[METRIC_COLS].values)
    joblib.dump(scaler, "checkpoints/robust_scaler.pkl")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = BugSeverityDataset(train_df, tokenizer, scaler, MAX_LENGTH)
    test_ds  = BugSeverityDataset(test_df,  tokenizer, scaler, MAX_LENGTH)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=True          # faster CPU→GPU transfers
    )
    test_loader = DataLoader(
        test_ds, batch_size=8,   # can be slightly larger for eval (no gradients)
        shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = ConcatClsModel(dropout_prob=BEST_CONFIG['dropout']).to(device)

    # Enable TF32 for slight speedup on Ampere GPUs (RTX 3050 is Ampere)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Train ────────────────────────────────────────────────────────────────
    history = train_model(
        model, train_loader, test_loader,
        BEST_CONFIG, device,
        save_path="checkpoints/best_model.pt"
    )

   # ── Load best checkpoint & evaluate ─────────────────────────────────────
model.load_state_dict(torch.load("checkpoints/best_model.pt"))
_, y_pred, y_true = evaluate(model, test_loader, device)

print("\n" + "="*60)
print("FINAL TEST SET RESULTS")
print("="*60)

# Dynamically detect which classes are present in predictions
import numpy as np
present_labels = sorted(list(set(y_true.tolist()) | set(y_pred.tolist())))
all_names = {0: 'Critical(0)', 1: 'Major(1)', 2: 'Medium(2)', 3: 'Low(3)'}
present_names = [all_names[l] for l in present_labels]

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

# Show class distribution in test set
print("\nClass distribution in test set:")
unique, counts = np.unique(y_true, return_counts=True)
for cls, cnt in zip(unique, counts):
    print(f"  {all_names[cls]}: {cnt} samples")
if __name__ == "__main__":
    main()