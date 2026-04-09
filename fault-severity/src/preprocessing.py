import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ─── Severity label → integer class mapping ───────────────────────────────────
SEVERITY_MAP = {
    # Class 0: Critical
    'blocker':   0, 'critical': 0,
    # Class 1: Major/High
    'major':     1, 'high':     1,
    # Class 2: Medium
    'medium':    2, 'normal':   2,
    # Class 3: Low/Minor/Trivial
    'low':       3, 'minor':    3, 'trivial': 3,
}

def load_and_map(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.strip()

    # Normalize severity strings
    df['severity_label'] = df['severity_label'].str.lower().str.strip()
    df['label'] = df['severity_label'].map(SEVERITY_MAP)

    # Drop unmapped rows
    before = len(df)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    after = len(df)
    print(f"Dropped {before - after} rows with unknown severity labels.")
    print(f"Class distribution:\n{df['label'].value_counts().sort_index()}")
    return df

def split_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    train_df, test_df = train_test_split(
        df, test_size=test_size,
        stratify=df['label'],
        random_state=random_state
    )
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

if __name__ == "__main__":
    df = load_and_map("data/buggy_methods.csv")
    train_df, test_df = split_data(df)
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)
    print("Saved train.csv and test.csv")   