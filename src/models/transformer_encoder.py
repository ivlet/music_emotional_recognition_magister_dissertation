"""Transformer encoder classifier for MFCC sequences."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerEncoderClassifier(nn.Module):
    """MFCC sequence classifier using a Transformer encoder."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_encoder(x)

        batch_size, max_len, _ = x.shape
        device = x.device
        key_padding_mask = torch.arange(max_len, device=device).unsqueeze(0) >= lengths.unsqueeze(1)

        encoded = self.transformer(x, src_key_padding_mask=key_padding_mask)
        mask = (~key_padding_mask).unsqueeze(-1).float()
        pooled = (encoded * mask).sum(dim=1) / lengths.unsqueeze(1).clamp(min=1).float()
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


def build_transformer_model(
    input_dim: int,
    num_classes: int,
    configs: dict[str, Any],
) -> TransformerEncoderClassifier:
    cfg = configs["models"]["sequence"]["transformer"]
    return TransformerEncoderClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]),
        num_layers=int(cfg["num_layers"]),
        dim_feedforward=int(cfg["dim_feedforward"]),
        dropout=float(cfg["dropout"]),
    )
