import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer
import pandas as pd
import numpy as np

METRIC_COLS = [
    'SLOC', 'CyclomaticComplexity', 'McClureComplexity',
    'NestingDepth', 'ProxyIndentation', 'Readability',
    'FanOut', 'HalsteadDifficulty', 'HalsteadEffort',
    'MaintainabilityIndex'
]

class BugSeverityDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: RobertaTokenizer,
        metric_scaler,           # fitted sklearn RobustScaler
        max_length: int = 512,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.scaler = metric_scaler
        self.max_length = max_length

        # Pre-scale metrics
        raw_metrics = df[METRIC_COLS].values.astype(np.float32)
        self.metrics = self.scaler.transform(raw_metrics).astype(np.float32)
        self.labels = df['label'].values.astype(np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        code = str(self.df.loc[idx, 'method_code'])
        encoding = self.tokenizer(
            code,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'metrics':        torch.tensor(self.metrics[idx], dtype=torch.float32),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long),
        }