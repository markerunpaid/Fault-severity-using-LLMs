# src/step3_scale.py
import pandas as pd
import joblib
import os
from sklearn.preprocessing import RobustScaler

METRIC_COLS = [
    'SLOC', 'CyclomaticComplexity', 'McClureComplexity',
    'NestingDepth', 'ProxyIndentation', 'Readability',
    'FanOut', 'HalsteadDifficulty', 'HalsteadEffort',
    'MaintainabilityIndex'
]

os.makedirs("checkpoints", exist_ok=True)

train_df = pd.read_csv("data/train_with_metrics.csv")
test_df  = pd.read_csv("data/test_with_metrics.csv")

print("Before scaling — Train metric stats:")
print(train_df[METRIC_COLS].describe().round(2))

# Fit ONLY on train, transform both
scaler = RobustScaler()
train_df[METRIC_COLS] = scaler.fit_transform(train_df[METRIC_COLS].values)
test_df[METRIC_COLS]  = scaler.transform(test_df[METRIC_COLS].values)

print("\nAfter scaling — Train metric stats:")
print(train_df[METRIC_COLS].describe().round(2))

# Save
joblib.dump(scaler, "checkpoints/robust_scaler.pkl")
train_df.to_csv("data/train_final.csv", index=False)
test_df.to_csv("data/test_final.csv",   index=False)

print("\nSaved:")
print("  checkpoints/robust_scaler.pkl")
print("  data/train_final.csv")
print("  data/test_final.csv")