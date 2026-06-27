"""CNN classifier for Mel-spectrogram inputs."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class MelSpectrogramCNN(nn.Module):
    """Simple CNN baseline with global average pooling."""

    def __init__(
        self,
        n_mels: int,
        num_classes: int,
        channels: list[int],
        kernel_size: int = 3,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = 1
        padding = kernel_size // 2

        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Dropout2d(dropout),
                ]
            )
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_cnn_model(
    n_mels: int,
    num_classes: int,
    configs: dict[str, Any],
) -> MelSpectrogramCNN:
    cfg = configs["models"]["spectrogram"]["cnn"]
    return MelSpectrogramCNN(
        n_mels=n_mels,
        num_classes=num_classes,
        channels=[int(c) for c in cfg["channels"]],
        kernel_size=int(cfg["kernel_size"]),
        dropout=float(cfg["dropout"]),
    )
