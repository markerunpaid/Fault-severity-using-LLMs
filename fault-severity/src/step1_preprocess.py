# src/step1_preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
import os

os.makedirs("data", exist_ok=True)

df = pd.read_csv("data/raw_combined.csv")

print(f"Total: {len(df)}")
print(f"Class distribution:\n{df['label'].value_counts().sort_index()}")

# Stratified 80/20 split
train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df['label'],
    random_state=42
)

train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

print(f"\nTrain: {len(train_df)}")
print(f"Test:  {len(test_df)}")
print(f"\nTrain class distribution:\n{train_df['label'].value_counts().sort_index()}")
print(f"\nTest class distribution:\n{test_df['label'].value_counts().sort_index()}")

train_df.to_csv("data/train_raw.csv", index=False)
test_df.to_csv("data/test_raw.csv",   index=False)
print("\nSaved → data/train_raw.csv and data/test_raw.csv")