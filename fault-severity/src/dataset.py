# src/dataset.py
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import numpy as np

METRIC_COLS = [
    'SLOC', 'CyclomaticComplexity', 'McClureComplexity',
    'NestingDepth', 'ProxyIndentation', 'Readability',
    'FanOut', 'HalsteadDifficulty', 'HalsteadEffort',
    'MaintainabilityIndex'
]

# Placeholder for synthetic SMOTE samples with no real code
EMPTY_CODE_PLACEHOLDER = "public void unknown() {}"

class BugSeverityDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=256):
        self.df         = df.reset_index(drop=True)
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.metrics    = df[METRIC_COLS].fillna(0.0).values.astype(np.float32)
        self.labels     = df['label'].values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        code = str(self.df.loc[idx, 'method_code']).strip()

        # Handle synthetic SMOTE samples (empty code)
        if not code or code == 'nan':
            code = EMPTY_CODE_PLACEHOLDER

        enc = self.tokenizer(
            code,
            max_length  = self.max_length,
            padding     = 'max_length',
            truncation  = True,
            return_tensors = 'pt',
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'metrics':        torch.tensor(self.metrics[idx], dtype=torch.float32),
            'label':          torch.tensor(self.labels[idx],  dtype=torch.long),
        }