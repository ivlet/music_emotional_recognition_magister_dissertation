"""MLP baseline for aggregated sequence features."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class SequenceMLP(nn.Module):
    """MLP classifier on aggregated (mean/std) MFCC vectors."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list[int],
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_mlp_model(
    input_dim: int,
    num_classes: int,
    configs: dict[str, Any],
) -> SequenceMLP:
    cfg = configs["models"]["sequence"]["mlp"]
    return SequenceMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=[int(d) for d in cfg["hidden_dims"]],
        dropout=float(cfg["dropout"]),
    )


def aggregate_sequence_features(sequences: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    Aggregate padded sequences with mean and std (mask-aware).

    ``sequences``: (batch, max_len, feat_dim)
    """
    batch_size, max_len, feat_dim = sequences.shape
    device = sequences.device
    mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1).float()

    summed = (sequences * mask).sum(dim=1)
    mean = summed / lengths.unsqueeze(1).clamp(min=1).float()

    var_sum = ((sequences - mean.unsqueeze(1)) ** 2 * mask).sum(dim=1)
    std = torch.sqrt(var_sum / lengths.unsqueeze(1).clamp(min=1).float() + 1e-6)

    return torch.cat([mean, std], dim=1)
