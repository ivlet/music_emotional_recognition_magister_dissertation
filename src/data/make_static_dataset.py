"""Build static (track-level) label datasets from DEAM metadata."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.load_deam import build_metadata_table
from src.utils.config import ensure_dir, get_project_root, load_configs, resolve_path

logger = logging.getLogger(__name__)

QUADRANT_LABELS = {
    "Q1": "high_valence_high_arousal",
    "Q2": "high_valence_low_arousal",
    "Q3": "low_valence_high_arousal",
    "Q4": "low_valence_low_arousal",
}


def _resolve_threshold(
    values: pd.Series,
    threshold: str | float,
) -> float:
    """Resolve a threshold specification to a numeric cut point."""
    if isinstance(threshold, str) and threshold.lower() == "median":
        return float(values.median())
    return float(threshold)


def assign_emotion_quadrant(
    valence: pd.Series | np.ndarray,
    arousal: pd.Series | np.ndarray,
    valence_threshold: str | float = "median",
    arousal_threshold: str | float = "median",
) -> pd.DataFrame:
    """
    Assign Russell-circumplex quadrant labels from valence and arousal.

    Quadrants:
      Q1 — high valence, high arousal
      Q2 — high valence, low arousal
      Q3 — low valence, high arousal
      Q4 — low valence, low arousal
    """
    valence_series = pd.Series(valence)
    arousal_series = pd.Series(arousal)

    v_cut = _resolve_threshold(valence_series.dropna(), valence_threshold)
    a_cut = _resolve_threshold(arousal_series.dropna(), arousal_threshold)

    high_v = valence_series >= v_cut
    high_a = arousal_series >= a_cut

    quadrant = pd.Series(index=valence_series.index, dtype="object")
    quadrant[high_v & high_a] = "Q1"
    quadrant[high_v & ~high_a] = "Q2"
    quadrant[~high_v & high_a] = "Q3"
    quadrant[~high_v & ~high_a] = "Q4"

    return pd.DataFrame(
        {
            "valence_threshold": v_cut,
            "arousal_threshold": a_cut,
            "emotion_quadrant": quadrant,
            "emotion_class": quadrant.map(QUADRANT_LABELS),
        }
    )


def build_static_label_dataset(
    metadata: pd.DataFrame | None = None,
    configs: dict[str, dict[str, Any]] | None = None,
    complete_only: bool = True,
) -> pd.DataFrame:
    """
    Create a static label table with one row per track.

    Uses mean valence/arousal from static annotations and assigns quadrant labels.
    """
    if configs is None:
        configs = load_configs()
    if metadata is None:
        metadata = build_metadata_table(configs)

    frame = metadata.copy()
    if complete_only:
        frame = frame[frame["is_complete"]].copy()

    emotion_cfg = configs["features"]["emotion"]
    labels = assign_emotion_quadrant(
        frame["valence"],
        frame["arousal"],
        valence_threshold=emotion_cfg["valence_threshold"],
        arousal_threshold=emotion_cfg["arousal_threshold"],
    )

    static_df = pd.concat(
        [
            frame[
                [
                    "song_id",
                    "valence",
                    "arousal",
                    "audio_path",
                    "has_static_annotation",
                    "has_dynamic_annotation",
                    "has_audio",
                ]
            ].reset_index(drop=True),
            labels.reset_index(drop=True),
        ],
        axis=1,
    )
    return static_df


def summarize_static_labels(static_df: pd.DataFrame) -> pd.DataFrame:
    """Return class counts and proportions for static emotion labels."""
    counts = static_df["emotion_quadrant"].value_counts().sort_index()
    summary = counts.rename("count").to_frame()
    summary["proportion"] = summary["count"] / summary["count"].sum()
    summary["emotion_class"] = summary.index.map(QUADRANT_LABELS)
    return summary.reset_index().rename(columns={"index": "emotion_quadrant"})


def save_static_dataset(
    static_df: pd.DataFrame,
    configs: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """Save static label dataset to the configured processed path."""
    if configs is None:
        configs = load_configs()
    root = get_project_root()
    output_path = resolve_path(root, configs["paths"]["processed"]["static_labels"])
    ensure_dir(output_path.parent)
    static_df.to_parquet(output_path, index=False)
    logger.info("Saved static label dataset to %s (%d tracks)", output_path, len(static_df))
    return output_path
