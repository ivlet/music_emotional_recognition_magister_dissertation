"""RNN and attention-based sequence classifiers."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class LSTMSequenceClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        direction = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * direction, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        if self.lstm.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        hidden = self.dropout(hidden)
        return self.classifier(hidden)


class GRUSequenceClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        direction = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * direction, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        if self.gru.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        hidden = self.dropout(hidden)
        return self.classifier(hidden)


class AttentionSequenceClassifier(nn.Module):
    """GRU encoder with learned attention pooling over time steps."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        direction = 2 if bidirectional else 1
        self.attention = nn.Linear(hidden_size * direction, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * direction, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.gru(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        batch_size, max_len, hidden_dim = outputs.shape
        device = outputs.device
        mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)

        scores = self.attention(outputs).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        context = (outputs * weights).sum(dim=1)
        context = self.dropout(context)
        return self.classifier(context)


def build_lstm_model(input_dim: int, num_classes: int, configs: dict[str, Any]) -> LSTMSequenceClassifier:
    cfg = configs["models"]["sequence"]["lstm"]
    return LSTMSequenceClassifier(
        input_dim=input_dim,
        hidden_size=int(cfg["hidden_size"]),
        num_layers=int(cfg["num_layers"]),
        num_classes=num_classes,
        dropout=float(cfg["dropout"]),
    )


def build_gru_model(input_dim: int, num_classes: int, configs: dict[str, Any]) -> GRUSequenceClassifier:
    cfg = configs["models"]["sequence"]["gru"]
    return GRUSequenceClassifier(
        input_dim=input_dim,
        hidden_size=int(cfg["hidden_size"]),
        num_layers=int(cfg["num_layers"]),
        num_classes=num_classes,
        dropout=float(cfg["dropout"]),
    )


def build_attention_model(
    input_dim: int,
    num_classes: int,
    configs: dict[str, Any],
) -> AttentionSequenceClassifier:
    cfg = configs["models"]["sequence"]["attention"]
    return AttentionSequenceClassifier(
        input_dim=input_dim,
        hidden_size=int(cfg["hidden_size"]),
        num_layers=int(cfg["num_layers"]),
        num_classes=num_classes,
        dropout=float(cfg["dropout"]),
    )
