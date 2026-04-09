# src/train_smote_only.py
# Uses existing best hyperparameters — no retuning needed
# Usage:
#   python src/train_smote_only.py --model codebert
#   python src/train_smote_only.py --model graphcodebert
#   python src/train_smote_only.py --model unixcoder
#   python src/train_smote_only.py --model codet5p

import os, sys, json, argparse
import torch, numpy as np, pandas as pd
from sklearn.metrics import (
    classification_report, f1_score,
    accuracy_score, matthews_corrcoef
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model import ConcatClsModel
from dataset import BugSeverityDataset
from trainer import train_model, evaluate

MODELS = {
    "codebert":      "microsoft/codebert-base",
    "graphcodebert": "microsoft/graphcodebert-base",
    "unixcoder":     "microsoft/unixcoder-base",
    "codet5p":       "Salesforce/codet5p-110m-embedding",
}

BATCH_SIZE         = 4
MAX_LENGTH         = 256
NUM_WORKERS        = 0
ACCUMULATION_STEPS = 4

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results",     exist_ok=True)


def load_tokenizer(model_key, model_name):
    if model_key == "codet5p":
        return AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
    return AutoTokenizer.from_pretrained(model_name)


def geometric_mean(y_true, y_pred):
    from sklearn.metrics import recall_score
    r = recall_score(y_true, y_pred, average=None, zero_division=0)
    r = r[r > 0]
    return float(np.prod(r) ** (1.0 / len(r))) if len(r) > 0 else 0.0


def main(model_key):
    model_name = MODELS[model_key]
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*60)
    print(f"MODEL  : {model_name}  [SMOTE — no retuning]")
    print(f"DEVICE : {device}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print("="*60)

    # ── Load existing best params ─────────────────────────────────────────────
    params_path = f"results/{model_key}_best_params.json"
    if not os.path.exists(params_path):
        print(f"ERROR: {params_path} not found.")
        print(f"Run tune_and_train.py --model {model_key} first.")
        sys.exit(1)

    with open(params_path) as f:
        saved = json.load(f)

    best_config = {
        'learning_rate':      saved['learning_rate'],
        'weight_decay':       saved['weight_decay'],
        'warmup_ratio':       saved['warmup_ratio'],
        'dropout':            saved['dropout'],
        'epochs':             8,
        'accumulation_steps': ACCUMULATION_STEPS,
    }

    print(f"\nUsing existing hyperparameters from {params_path}:")
    for k, v in best_config.items():
        if k != 'accumulation_steps':
            print(f"  {k:20s}: {v}")

    # ── Load data — SMOTE train, original test ────────────────────────────────
    train_df = pd.read_csv("data/train_smote.csv")
    test_df  = pd.read_csv("data/test_final.csv")

    print(f"\nSMOTE train distribution:")
    print(train_df['label'].value_counts().sort_index())
    print(f"\nTest distribution (original, unchanged):")
    print(test_df['label'].value_counts().sort_index())
    print(f"\nTrain: {len(train_df)} | Test: {len(test_df)}")

    # ── Tokenizer & DataLoaders ───────────────────────────────────────────────
    tokenizer    = load_tokenizer(model_key, model_name)

    train_loader = DataLoader(
        BugSeverityDataset(train_df, tokenizer, MAX_LENGTH),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        BugSeverityDataset(test_df, tokenizer, MAX_LENGTH),
        batch_size=8, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ConcatClsModel(
        model_name   = model_name,
        dropout_prob = best_config['dropout']
    ).to(device)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Train ─────────────────────────────────────────────────────────────────
    save_path = f"checkpoints/best_smote_{model_key}.pt"

    history = train_model(
        model, train_loader, test_loader,
        best_config, device, save_path=save_path
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    model.load_state_dict(torch.load(save_path, map_location=device))
    _, y_pred, y_true = evaluate(model, test_loader, device)

    present_labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    all_names      = {0:'Critical(0)', 1:'Major(1)', 2:'Medium(2)', 3:'Low(3)'}

    print("\n" + "="*60)
    print(f"FINAL TEST RESULTS (SMOTE) — {model_key.upper()}")
    print("="*60)
    print(classification_report(
        y_true, y_pred,
        labels        = present_labels,
        target_names  = [all_names[l] for l in present_labels],
        zero_division = 0
    ))

    final = {
        "model":       f"{model_key}_smote",
        "model_name":  model_name,
        "Accuracy":    round(accuracy_score(y_true, y_pred), 4),
        "Macro F1":    round(f1_score(y_true, y_pred, average='macro',    zero_division=0), 4),
        "Weighted F1": round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        "MCC":         round(matthews_corrcoef(y_true, y_pred), 4),
        "G-Mean":      round(geometric_mean(y_true, y_pred), 4),
        "used_params": best_config,
        "history":     history,
    }

    print(f"  Accuracy    : {final['Accuracy']}")
    print(f"  Macro F1    : {final['Macro F1']}")
    print(f"  Weighted F1 : {final['Weighted F1']}")
    print(f"  MCC         : {final['MCC']}")
    print(f"  G-Mean      : {final['G-Mean']}")

    with open(f"results/{model_key}_smote_results.json", "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved → results/{model_key}_smote_results.json")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()), required=True)
    args = parser.parse_args()
    main(args.model)