# src/tune_and_train.py
# Usage:
#   python src/tune_and_train.py --model codebert
#   python src/tune_and_train.py --model graphcodebert
#   python src/tune_and_train.py --model unixcoder
#   python src/tune_and_train.py --model codet5p

import os, sys, json, argparse
import torch, optuna, numpy as np, pandas as pd
from sklearn.metrics import (
    classification_report, f1_score,
    accuracy_score, matthews_corrcoef
)
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TRANSFORMERS_VERBOSITY"]          = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"]   = "1"
import transformers
transformers.logging.set_verbosity_error()
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model import ConcatClsModel
from dataset import BugSeverityDataset
from trainer import train_model, evaluate

# ── Model registry ────────────────────────────────────────────────────────────
MODELS = {
    "codebert":      "microsoft/codebert-base",
    "graphcodebert": "microsoft/graphcodebert-base",
    "unixcoder":     "microsoft/unixcoder-base",
    "codet5p":       "Salesforce/codet5p-110m-embedding",
}

# ── Fixed settings ────────────────────────────────────────────────────────────
BATCH_SIZE         = 4
MAX_LENGTH         = 256
NUM_WORKERS        = 0      
ACCUMULATION_STEPS = 4
N_TRIALS           = 20
PATIENCE           = 8

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results",     exist_ok=True)


# ── Tokenizer loader ──────────────────────────────────────────────────────────
def load_tokenizer(model_key, model_name):
    """Load tokenizer with model-specific kwargs."""
    if model_key == "codet5p":
        return AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
    return AutoTokenizer.from_pretrained(model_name)


# ── Early stopping ────────────────────────────────────────────────────────────
class EarlyStoppingCallback:
    def __init__(self, patience: int = 8):
        self.patience   = patience
        self.best_value = None
        self.no_improve = 0

    def __call__(self, study, trial):
        if self.best_value is None or study.best_value > self.best_value:
            self.best_value = study.best_value
            self.no_improve = 0
        else:
            self.no_improve += 1
            print(f"  No improvement for {self.no_improve}/{self.patience} "
                  f"(best={self.best_value:.4f})")
            if self.no_improve >= self.patience:
                print(f"\n  Early stopping! Best F1={self.best_value:.4f}")
                study.stop()


# ── Helpers ───────────────────────────────────────────────────────────────────
def geometric_mean(y_true, y_pred):
    from sklearn.metrics import recall_score
    r = recall_score(y_true, y_pred, average=None, zero_division=0)
    r = r[r > 0]
    return float(np.prod(r) ** (1.0 / len(r))) if len(r) > 0 else 0.0


# ── Optuna objective ──────────────────────────────────────────────────────────
def objective(trial, model_key, model_name, train_df, val_df, device):
    config = {
        'learning_rate':      trial.suggest_float("learning_rate", 1e-5, 8e-5, log=True),
        'weight_decay':       trial.suggest_float("weight_decay",  1e-6, 1e-2, log=True),
        'warmup_ratio':       trial.suggest_float("warmup_ratio",  0.05, 0.40),
        'dropout':            trial.suggest_float("dropout",       0.05, 0.40),
        'epochs':             trial.suggest_int("epochs",          3,    8),
        'accumulation_steps': ACCUMULATION_STEPS,
    }

    print(f"\n  Trial {trial.number:02d}/{N_TRIALS-1} | "
          f"lr={config['learning_rate']:.2e} | "
          f"wd={config['weight_decay']:.2e} | "
          f"warmup={config['warmup_ratio']:.3f} | "
          f"dropout={config['dropout']:.3f} | "
          f"epochs={config['epochs']}")

    tokenizer    = load_tokenizer(model_key, model_name)
    train_loader = DataLoader(
        BugSeverityDataset(train_df, tokenizer, MAX_LENGTH, use_metrics=use_metrics),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        BugSeverityDataset(val_df, tokenizer, MAX_LENGTH, use_metrics=use_metrics),
        batch_size=8, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    model     = ConcatClsModel(
        model_name   = model_name,
        dropout_prob = config['dropout']
    ).to(device)

    save_path = f"checkpoints/{model_key}_trial_{trial.number}.pt"

    history   = train_model(
        model, train_loader, val_loader,
        config, device, save_path=save_path
    )
    best_f1 = max(h['val_macro_f1'] for h in history)

    # Cleanup
    if os.path.exists(save_path):
        os.remove(save_path)
    del model
    torch.cuda.empty_cache()

    print(f"  Trial {trial.number:02d} → Best F1: {best_f1:.4f}")
    return best_f1


# ── Final training ────────────────────────────────────────────────────────────
def final_training(model_key, model_name, best_params, train_df, test_df, device):
    print("\n" + "="*60)
    print("PHASE 2: FINAL TRAINING WITH BEST PARAMS")
    print("="*60)
    for k, v in best_params.items():
        print(f"  {k:20s}: {v}")

    best_config = {
        'learning_rate':      best_params['learning_rate'],
        'weight_decay':       best_params['weight_decay'],
        'warmup_ratio':       best_params['warmup_ratio'],
        'dropout':            best_params['dropout'],
        'epochs':             8,               # full epochs, best ckpt auto-saved
        'accumulation_steps': ACCUMULATION_STEPS,
    }

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

    model = ConcatClsModel(
        model_name   = model_name,
        dropout_prob = best_config['dropout']
    ).to(device)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    save_path = f"checkpoints/best_{model_key}.pt"

    history = train_model(
        model, train_loader, test_loader,
        best_config, device, save_path=save_path
    )

    # Load best checkpoint (not last epoch)
    model.load_state_dict(torch.load(save_path, map_location=device))
    _, y_pred, y_true = evaluate(model, test_loader, device)

    present_labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    all_names      = {0:'Critical(0)', 1:'Major(1)', 2:'Medium(2)', 3:'Low(3)'}

    print("\n" + "="*60)
    print(f"FINAL TEST RESULTS — {model_key.upper()}")
    print("="*60)
    print(classification_report(
        y_true, y_pred,
        labels       = present_labels,
        target_names = [all_names[l] for l in present_labels],
        zero_division = 0
    ))

    final = {
        "model":       model_key,
        "model_name":  model_name,
        "Accuracy":    round(accuracy_score(y_true, y_pred), 4),
        "Macro F1":    round(f1_score(y_true, y_pred, average='macro',    zero_division=0), 4),
        "Weighted F1": round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4),
        "MCC":         round(matthews_corrcoef(y_true, y_pred), 4),
        "G-Mean":      round(geometric_mean(y_true, y_pred), 4),
        "best_params": best_params,
        "history":     history,
    }

    print(f"  Accuracy    : {final['Accuracy']}")
    print(f"  Macro F1    : {final['Macro F1']}")
    print(f"  Weighted F1 : {final['Weighted F1']}")
    print(f"  MCC         : {final['MCC']}")
    print(f"  G-Mean      : {final['G-Mean']}")

    with open(f"results/{model_key}_results.json", "w") as f:
        json.dump(final, f, indent=2)
    print(f"\nSaved → results/{model_key}_results.json")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return final


# ── Main ──────────────────────────────────────────────────────────────────────
def main(model_key):
    model_name = MODELS[model_key]
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "="*60)
    print(f"MODEL  : {model_name}")
    print(f"DEVICE : {device}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print("="*60)

    # ── Load data ─────────────────────────────────────────────────────────────
    train_df = pd.read_csv("data/train_final.csv")
    test_df  = pd.read_csv("data/test_final.csv")

    inner_train_df, val_df = train_test_split(
        train_df, test_size=0.15,
        stratify=train_df['label'], random_state=42
    )
    inner_train_df = inner_train_df.reset_index(drop=True)
    val_df         = val_df.reset_index(drop=True)

    print(f"Inner train : {len(inner_train_df)} | "
          f"Val: {len(val_df)} | Test: {len(test_df)}")

    # ── Verify tokenizer loads before starting trials ─────────────────────────
    print(f"\nVerifying tokenizer for {model_key}...")
    try:
        tok     = load_tokenizer(model_key, model_name)
        test_enc = tok(
            "public void test() {}",
            return_tensors="pt", max_length=16,
            padding="max_length", truncation=True
        )
        print(f"  Tokenizer OK — type: {type(tok).__name__} | "
              f"input shape: {test_enc['input_ids'].shape}")
        del tok
    except Exception as e:
        print(f"  Tokenizer FAILED: {e}")
        print(f"  Cannot proceed.")
        sys.exit(1)

    # ── Phase 1: Optuna search ────────────────────────────────────────────────
    print(f"\nPHASE 1: Optuna search")
    print(f"  {N_TRIALS} trials max | early stop after {PATIENCE} no-improve\n")

    study = optuna.create_study(
        direction  = "maximize",
        sampler    = optuna.samplers.TPESampler(seed=42),
        pruner     = optuna.pruners.MedianPruner(
                         n_startup_trials = 5,
                         n_warmup_steps   = 1,
                     ),
        study_name = f"{model_key}_study",
    )

    early_stop = EarlyStoppingCallback(patience=PATIENCE)

    study.optimize(
        lambda t: objective(t, model_key, model_name,
                            inner_train_df, val_df, device),
        n_trials       = N_TRIALS,
        gc_after_trial = True,
        callbacks      = [early_stop],
    )

    best_params = study.best_params
    print(f"\n{'='*60}")
    print(f"OPTUNA DONE — Best Val F1: {study.best_value:.4f}")
    print(f"Best params:")
    for k, v in best_params.items():
        print(f"  {k:20s}: {v}")

    with open(f"results/{model_key}_best_params.json", "w") as f:
        json.dump({"best_val_f1": study.best_value, **best_params}, f, indent=2)

    # ── Phase 2: Final training ───────────────────────────────────────────────
    final_training(model_key, model_name, best_params, train_df, test_df, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        required=True,
        help="Model to tune and train"
    )
    args = parser.parse_args()
    main(args.model)
