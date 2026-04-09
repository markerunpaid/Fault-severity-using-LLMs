# src/model.py

import torch
import torch.nn as nn
from transformers import AutoModel, T5EncoderModel


class ConcatClsModel(nn.Module):

    def __init__(
        self,
        model_name:   str   = "microsoft/codebert-base",
        num_metrics:  int   = 10,
        num_classes:  int   = 4,
        dropout_prob: float = 0.1,
    ):
        super(ConcatClsModel, self).__init__()
        self.model_name = model_name

        if "codet5p-220m" in model_name.lower():
            # Encoder-decoder T5 — encoder only, mean pooling
            self.encoder   = T5EncoderModel.from_pretrained(model_name)
            hidden_size    = self.encoder.config.d_model
            self.pool_type = "mean"

        elif "codet5p-110m-embedding" in model_name.lower():
            # Custom T5-based model — must use T5EncoderModel, not AutoModel
            self.encoder   = T5EncoderModel.from_pretrained(
                model_name,
                trust_remote_code       = True,
                ignore_mismatched_sizes = True,
            )
            hidden_size    = self.encoder.config.d_model
            self.pool_type = "mean"

        else:
            # CodeBERT, GraphCodeBERT, UniXcoder — all RoBERTa-style
            self.encoder   = AutoModel.from_pretrained(model_name)
            hidden_size    = self.encoder.config.hidden_size
            self.pool_type = "cls"

        self.dropout    = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Linear(hidden_size + num_metrics, num_classes)

        print(f"  Encoder : {model_name}")
        print(f"  Hidden  : {hidden_size}  |  Pooling: {self.pool_type}")
        print(f"  Classes : {num_classes}  |  Metrics: {num_metrics}")

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        metrics:        torch.Tensor,
    ):
        outputs = self.encoder(
            input_ids      = input_ids,
            attention_mask = attention_mask,
        )

        if self.pool_type == "cls":
            # RoBERTa-style: first token is [CLS]
            embedding = outputs.last_hidden_state[:, 0, :]

        else:
            # T5-style: mean pool over all non-padding tokens
            token_emb = outputs.last_hidden_state               # [B, seq, hidden]
            mask_exp  = attention_mask.unsqueeze(-1).float()    # [B, seq, 1]
            sum_emb   = torch.sum(token_emb * mask_exp, dim=1)  # [B, hidden]
            sum_mask  = torch.clamp(mask_exp.sum(dim=1), min=1e-9)
            embedding = sum_emb / sum_mask                      # [B, hidden]

        combined = torch.cat([embedding, metrics], dim=1)
        combined = self.dropout(combined)
        logits   = self.classifier(combined)

        return logits