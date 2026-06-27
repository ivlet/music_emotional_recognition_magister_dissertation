"""Collect experiment metric summaries from results/metrics/."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import get_project_root, load_configs, resolve_path

logger = logging.getLogger(__name__)

STANDARD_COLUMNS = [
    "experiment_group",
    "task_type",
    "model_name",
    "feature_type",
    "target_type",
    "eval_split",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "mae",
    "rmse",
    "r2",
    "comments",
]

EXPERIMENT_SOURCES: dict[str, dict[str, str]] = {
    "static_baselines": {
        "filename": "static_baselines_summary.csv",
        "experiment_group": "static",
        "task_type": "static_classification",
        "feature_type": "spectral_aggregated",
        "target_type": "emotion_quadrant",
        "comments": "Track-level classical ML on aggregated spectral features.",
    },
    "dynamic_window_classification": {
        "filename": "dynamic_window_classification_summary.csv",
        "experiment_group": "dynamic_window",
        "task_type": "dynamic_window_classification",
        "feature_type": "dynamic_window_spectral",
        "target_type": "dynamic_emotion_quadrant",
        "comments": "Window-level classical ML.",
    },
    "dynamic_window_regression": {
        "filename": "dynamic_window_regression_summary.csv",
        "experiment_group": "dynamic_window",
        "task_type": "dynamic_window_regression",
        "feature_type": "dynamic_window_spectral",
        "target_type": "valence_arousal",
        "comments": "Window-level regression.",
    },
    "sequence_models": {
        "filename": "sequence_models_summary.csv",
        "experiment_group": "sequence",
        "task_type": "sequence_classification",
        "feature_type": "mfcc_sequence",
        "target_type": "emotion_quadrant",
        "comments": "Track-level sequence models on MFCC.",
    },
    "spectrogram_models": {
        "filename": "spectrogram_models_summary.csv",
        "experiment_group": "spectrogram",
        "task_type": "spectrogram_classification",
        "feature_type": "mel_spectrogram",
        "target_type": "emotion_quadrant",
        "comments": "Mel-spectrogram CNN/CRNN.",
    },
    "pretrained_audio_models": {
        "filename": "pretrained_audio_models_summary.csv",
        "experiment_group": "pretrained",
        "task_type": "pretrained_embedding_classification",
        "feature_type": "pretrained_audio_embedding",
        "target_type": "emotion_quadrant",
        "comments": "Classifier on frozen pretrained embeddings.",
    },
}


def _standardize_summary(df: pd.DataFrame, meta: dict[str, str]) -> pd.DataFrame:
    frame = df.copy()

    for key in ("experiment_group", "task_type", "feature_type", "target_type", "comments"):
        if key not in frame.columns:
            frame[key] = meta.get(key)

    if "target_type" not in frame.columns and "target" in frame.columns:
        frame["target_type"] = frame["target"]

    if "classifier" in frame.columns:
        if "model_alias" in frame.columns:
            frame["model_name"] = (
                frame["model_alias"].astype(str) + "_" + frame["classifier"].astype(str)
            )
        elif "model_name" in frame.columns:
            frame["model_name"] = (
                frame["model_name"].astype(str) + "_" + frame["classifier"].astype(str)
            )

    for col in STANDARD_COLUMNS:
        if col not in frame.columns:
            if col in {"accuracy", "macro_f1", "weighted_f1", "mae", "rmse", "r2"}:
                frame[col] = np.nan
            else:
                frame[col] = None

    return frame


def collect_experiment_summaries(
    configs: dict[str, dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Load all available experiment summary CSV files.

    Returns combined dataframe and list of warning messages for missing files.
    """
    if configs is None:
        configs = load_configs()

    root = get_project_root()
    metrics_dir = resolve_path(root, configs["paths"]["results"]["metrics"])
    reports_tables = resolve_path(root, configs["paths"]["reports"]["tables"])

    frames: list[pd.DataFrame] = []
    warnings: list[str] = []

    for source_name, meta in EXPERIMENT_SOURCES.items():
        candidates = [
            metrics_dir / meta["filename"],
            reports_tables / meta["filename"],
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            msg = f"Missing summary file for '{source_name}': {meta['filename']}"
            logger.warning(msg)
            warnings.append(msg)
            continue

        df = pd.read_csv(path)
        df = _standardize_summary(df, meta)
        frames.append(df)
        logger.info("Loaded %s from %s (%d rows)", source_name, path, len(df))

    if not frames:
        empty = pd.DataFrame(columns=STANDARD_COLUMNS)
        warnings.append("No experiment summary files were found.")
        return empty, warnings

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.reindex(columns=STANDARD_COLUMNS, fill_value=np.nan)
    return combined, warnings
