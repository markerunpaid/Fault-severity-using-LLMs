# src/extract_embeddings.py
"""
Extracts frozen embeddings from all 4 fine-tuned transformer models.

For each model, this script:
  1. Loads the fine-tuned checkpoint (best_{model_key}.pt)
  2. Runs inference on train_final.csv and test_final.csv
  3. Extracts the [CLS] or mean-pooled embedding (768-d) per method
  4. Saves as .npy files under embeddings/

Output files:
  embeddings/{model_key}_train_embeddings.npy   shape: (N_train, 768)
  embeddings/{model_key}_train_metrics.npy      shape: (N_train, 10)
  embeddings/{model_key}_train_labels.npy       shape: (N_train,)
  embeddings/{model_key}_test_embeddings.npy    shape: (N_test,  768)
  embeddings/{model_key}_test_metrics.npy       shape: (N_test,  10)
  embeddings/{model_key}_test_labels.npy        shape: (N_test,)

These are combined downstream in train_ml_models.py as:
  X = [embedding (768-d)] + [metrics (10-d)]  →  778-d feature vector
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# ── Make src imports work regardless of working directory ─────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from model import ConcatClsModel
from dataset import BugSeverityDataset, METRIC_COLS

# ── Model registry ────────────────────────────────────────────────────────────
MODELS = {
    "codebert":      {
        "model_name":  "microsoft/codebert-base",
        "dropout":     0.084,
        "hidden_size": 768,
    },
    "graphcodebert": {
        "model_name":  "microsoft/graphcodebert-base",
        "dropout":     0.328,
        "hidden_size": 768,
    },
    "unixcoder":     {
        "model_name":  "microsoft/unixcoder-base",
        "dropout":     0.289,
        "hidden_size": 768,
    },
    "codet5p":       {
        "model_name":  "Salesforce/codet5p-110m-embedding",
        "dropout":     0.311,
        "hidden_size": 768,
    },
}

MAX_LENGTH  = 256
BATCH_SIZE  = 16     # larger than training batch — no gradients needed
NUM_WORKERS = 0


# ── Tokenizer loader ──────────────────────────────────────────────────────────
def load_tokenizer(model_key, model_name):
    if model_key == "codet5p":
        return AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
    return AutoTokenizer.from_pretrained(model_name)


# ── Embedding extractor ───────────────────────────────────────────────────────
def extract_embeddings(model, loader, device):
    """
    Run forward pass through encoder only (no classifier head).
    Returns:
        embeddings : np.ndarray  shape (N, hidden_size)
        metrics    : np.ndarray  shape (N, 10)
        labels     : np.ndarray  shape (N,)
    """
    model.eval()
    all_embeddings = []
    all_metrics    = []
    all_labels     = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            metrics        = batch['metrics']          # keep on CPU for now
            labels         = batch['label']

            # ── Encoder forward ───────────────────────────────────────────────
            outputs = model.encoder(
                input_ids      = input_ids,
                attention_mask = attention_mask,
            )

            if model.pool_type == "cls":
                # RoBERTa-style: [CLS] is first token
                embedding = outputs.last_hidden_state[:, 0, :]        # [B, 768]

            else:
                # T5-style: mean pool over non-padding tokens
                token_emb = outputs.last_hidden_state                  # [B, seq, hidden]
                mask_exp  = attention_mask.unsqueeze(-1).float()       # [B, seq, 1]
                sum_emb   = torch.sum(token_emb * mask_exp, dim=1)    # [B, hidden]
                sum_mask  = torch.clamp(mask_exp.sum(dim=1), min=1e-9)
                embedding = sum_emb / sum_mask                         # [B, hidden]

            all_embeddings.append(embedding.cpu().numpy())
            all_metrics.append(metrics.numpy())
            all_labels.append(labels.numpy())

    return (
        np.vstack(all_embeddings).astype(np.float32),
        np.vstack(all_metrics).astype(np.float32),
        np.concatenate(all_labels).astype(np.int64),
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main(model_keys):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if torch.cuda.is_available():
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    os.makedirs("embeddings", exist_ok=True)

    train_df = pd.read_csv("data/train_final.csv")
    test_df  = pd.read_csv("data/test_final.csv")
    print(f"Train  : {len(train_df)} samples")
    print(f"Test   : {len(test_df)} samples")

    for model_key in model_keys:
        cfg        = MODELS[model_key]
        model_name = cfg["model_name"]
        dropout    = cfg["dropout"]
        ckpt_path  = f"checkpoints/best_{model_key}.pt"

        if not os.path.exists(ckpt_path):
            print(f"\n[SKIP] {model_key} — checkpoint not found: {ckpt_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Extracting embeddings: {model_key.upper()}")
        print(f"  Model    : {model_name}")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"{'='*60}")

        # ── Load model ────────────────────────────────────────────────────────
        tokenizer = load_tokenizer(model_key, model_name)

        model = ConcatClsModel(
            model_name   = model_name,
            dropout_prob = dropout,
        ).to(device)
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device)
        )
        model.eval()
        print(f"  Loaded fine-tuned checkpoint ✓")
        print(f"  Embedding size : {cfg['hidden_size']}-d | Pool: {model.pool_type}")

        # ── Build DataLoaders ─────────────────────────────────────────────────
        train_ds = BugSeverityDataset(train_df, tokenizer, MAX_LENGTH)
        test_ds  = BugSeverityDataset(test_df,  tokenizer, MAX_LENGTH)

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE,
            shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
        )
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE,
            shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
        )

        # ── Extract train embeddings ──────────────────────────────────────────
        print(f"\n  Extracting TRAIN embeddings ({len(train_df)} samples)...")
        from tqdm import tqdm
        model.eval()

        # Wrap loader in tqdm for progress
        train_loader_tqdm = tqdm(train_loader, desc=f"  [{model_key}] train")
        test_loader_tqdm  = tqdm(test_loader,  desc=f"  [{model_key}] test ")

        train_emb, train_met, train_lbl = extract_embeddings(
            model, train_loader_tqdm, device
        )
        print(f"  Train embeddings shape : {train_emb.shape}")

        # ── Extract test embeddings ───────────────────────────────────────────
        print(f"  Extracting TEST embeddings ({len(test_df)} samples)...")
        test_emb, test_met, test_lbl = extract_embeddings(
            model, test_loader_tqdm, device
        )
        print(f"  Test embeddings shape  : {test_emb.shape}")

        # ── Save ──────────────────────────────────────────────────────────────
        prefix_train = f"embeddings/{model_key}_train"
        prefix_test  = f"embeddings/{model_key}_test"

        np.save(f"{prefix_train}_embeddings.npy", train_emb)
        np.save(f"{prefix_train}_metrics.npy",    train_met)
        np.save(f"{prefix_train}_labels.npy",     train_lbl)

        np.save(f"{prefix_test}_embeddings.npy",  test_emb)
        np.save(f"{prefix_test}_metrics.npy",     test_met)
        np.save(f"{prefix_test}_labels.npy",      test_lbl)

        print(f"\n  Saved:")
        print(f"    {prefix_train}_embeddings.npy  {train_emb.shape}")
        print(f"    {prefix_train}_metrics.npy     {train_met.shape}")
        print(f"    {prefix_train}_labels.npy      {train_lbl.shape}")
        print(f"    {prefix_test}_embeddings.npy   {test_emb.shape}")
        print(f"    {prefix_test}_metrics.npy      {test_met.shape}")
        print(f"    {prefix_test}_labels.npy       {test_lbl.shape}")

        # ── Free GPU memory before next model ─────────────────────────────────
        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("All embeddings extracted.")
    print("Next step: python src/train_ml_models.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract embeddings from fine-tuned transformer models"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()) + ["all"],
        default=["all"],
        help="Which models to extract. Use 'all' for all 4.",
    )
    args = parser.parse_args()

    keys = list(MODELS.keys()) if "all" in args.models else args.models
    main(keys)
