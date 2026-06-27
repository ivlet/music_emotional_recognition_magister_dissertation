"""Shared helpers for feature extraction."""

from __future__ import annotations

from typing import Callable

import numpy as np

AGGREGATION_FUNCTIONS: dict[str, Callable[[np.ndarray], float]] = {
    "mean": lambda x: float(np.mean(x)),
    "std": lambda x: float(np.std(x)),
    "min": lambda x: float(np.min(x)),
    "max": lambda x: float(np.max(x)),
    "median": lambda x: float(np.median(x)),
}


def aggregate_feature_matrix(
    matrix: np.ndarray,
    prefix: str,
    aggregations: tuple[str, ...] | list[str],
) -> dict[str, float]:
    """
    Aggregate a 2-D feature matrix (n_features, n_frames) into a flat dict.

    Column names follow ``{prefix}_{feature_idx}_{aggregation}`` with 1-based indices.
    """
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D feature array for '{prefix}', got shape {matrix.shape}")

    features: dict[str, float] = {}
    for idx in range(matrix.shape[0]):
        values = matrix[idx]
        for agg_name in aggregations:
            if agg_name not in AGGREGATION_FUNCTIONS:
                raise ValueError(f"Unknown aggregation: {agg_name}")
            features[f"{prefix}_{idx + 1}_{agg_name}"] = AGGREGATION_FUNCTIONS[agg_name](values)
    return features


def aggregate_scalar_feature(
    value: float | np.ndarray,
    prefix: str,
    aggregations: tuple[str, ...] | list[str],
) -> dict[str, float]:
    """Aggregate a scalar or 1-D sequence (e.g. tempo estimates)."""
    array = np.atleast_1d(np.asarray(value, dtype=float))
    return aggregate_feature_matrix(array, prefix, aggregations)
