"""CRNN classifier for Mel-spectrogram inputs."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class MelSpectrogramCRNN(nn.Module):
    """CNN feature extractor followed by bidirectional GRU/LSTM."""

    def __init__(
        self,
        n_mels: int,
        num_classes: int,
        cnn_channels: list[int],
        kernel_size: int = 3,
        rnn_hidden_size: int = 128,
        rnn_num_layers: int = 2,
        rnn_type: str = "gru",
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 1
        padding = kernel_size // 2

        for out_channels in cnn_channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
                    nn.Dropout2d(dropout),
                ]
            )
            in_channels = out_channels

        self.cnn = nn.Sequential(*layers)
        self.rnn_type = rnn_type.lower()
        rnn_cls = nn.GRU if self.rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=in_channels,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
            dropout=dropout if rnn_num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(rnn_hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        features = features.mean(dim=2)
        features = features.permute(0, 2, 1).contiguous()

        if self.rnn_type == "gru":
            outputs, _ = self.rnn(features)
        else:
            outputs, _ = self.rnn(features)

        pooled = outputs[:, -1, :]
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


def build_crnn_model(
    n_mels: int,
    num_classes: int,
    configs: dict[str, Any],
) -> MelSpectrogramCRNN:
    cfg = configs["models"]["spectrogram"]["crnn"]
    return MelSpectrogramCRNN(
        n_mels=n_mels,
        num_classes=num_classes,
        cnn_channels=[int(c) for c in cfg["cnn_channels"]],
        kernel_size=int(cfg["kernel_size"]),
        rnn_hidden_size=int(cfg["rnn_hidden_size"]),
        rnn_num_layers=int(cfg["rnn_num_layers"]),
        rnn_type=str(cfg.get("rnn_type", "gru")),
        dropout=float(cfg["dropout"]),
    )
