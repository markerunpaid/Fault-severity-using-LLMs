# src/apply_smote.py
"""
Apply SMOTE oversampling to training data to fix class imbalance.
IMPORTANT:
  - SMOTE is applied ONLY to training data
  - Test data is NEVER touched
  - SMOTE operates on the 10 scaled code metrics
  - CodeBERT embeddings cannot be SMOTEd (too high dimensional)
  - Strategy: upsample minority classes to 50% of majority class count
"""

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter
import os

METRIC_COLS = [
    'SLOC', 'CyclomaticComplexity', 'McClureComplexity',
    'NestingDepth', 'ProxyIndentation', 'Readability',
    'FanOut', 'HalsteadDifficulty', 'HalsteadEffort',
    'MaintainabilityIndex'
]

def apply_smote(train_path, output_path, strategy='auto'):
    df = pd.read_csv(train_path)

    print("Before SMOTE:")
    print(df['label'].value_counts().sort_index())
    original_counts = dict(df['label'].value_counts())
    majority_count  = max(original_counts.values())

    # ── SMOTE strategy ────────────────────────────────────────────────────────
    # Target: bring each minority class to 60% of majority class
    # Class 1 (Major) = 1665 → target others at ~1000
    target = int(majority_count * 0.60)

    sampling_strategy = {}
    for cls, count in original_counts.items():
        if count < target:
            sampling_strategy[cls] = target
            print(f"  Class {cls}: {count} → {target} (oversampled)")
        else:
            print(f"  Class {cls}: {count} (unchanged — majority)")

    # ── Apply SMOTE on metrics only ───────────────────────────────────────────
    X = df[METRIC_COLS].values
    y = df['label'].values

    smote = SMOTE(
        sampling_strategy = sampling_strategy,
        random_state      = 42,
        k_neighbors       = 5,   # use 3 if any class has < 6 samples
    )

    try:
        X_res, y_res = smote.fit_resample(X, y)
    except ValueError as e:
        print(f"SMOTE with k=5 failed: {e}")
        print("Retrying with k_neighbors=3...")
        smote = SMOTE(
            sampling_strategy = sampling_strategy,
            random_state      = 42,
            k_neighbors       = 3,
        )
        X_res, y_res = smote.fit_resample(X, y)

    print(f"\nAfter SMOTE:")
    print(pd.Series(y_res).value_counts().sort_index())

    # ── Rebuild dataframe ─────────────────────────────────────────────────────
    # Original rows: keep method_code for real samples
    # New synthetic rows: method_code will be None (metrics only)
    n_original = len(df)
    n_synthetic = len(y_res) - n_original

    # Metrics for all rows (original + synthetic)
    metrics_df = pd.DataFrame(X_res, columns=METRIC_COLS)
    labels_ser = pd.Series(y_res, name='label')

    # method_code: keep original, fill synthetic with empty string
    # (synthetic samples won't have real code — CodeBERT will get empty string)
    method_codes = list(df['method_code'].values) + [''] * n_synthetic

    result_df = metrics_df.copy()
    result_df['label']       = labels_ser
    result_df['method_code'] = method_codes

    # Reorder columns to match original
    result_df = result_df[['method_code', 'label'] + METRIC_COLS]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)

    print(f"\nSaved → {output_path}")
    print(f"Original samples : {n_original}")
    print(f"Synthetic samples: {n_synthetic}")
    print(f"Total            : {len(result_df)}")
    return result_df

if __name__ == "__main__":
    apply_smote(
        train_path  = "data/train_final.csv",
        output_path = "data/train_smote.csv",
    )