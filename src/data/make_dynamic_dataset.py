"""Build dynamic (window-level) label datasets from DEAM annotations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.data.load_deam import build_metadata_table, load_dynamic_annotations
from src.data.make_static_dataset import QUADRANT_LABELS, assign_emotion_quadrant
from src.data.splits import (
    assign_splits_to_dataframe,
    create_track_split_table,
    load_track_splits,
    verify_window_splits,
)
from src.utils.config import ensure_dir, get_project_root, load_configs, resolve_path

logger = logging.getLogger(__name__)

STANDARD_DYNAMIC_WINDOW_COLUMNS = (
    "song_id",
    "window_index",
    "window_start_sec",
    "window_end_sec",
    "annotation_time_sec",
    "valence",
    "arousal",
    "dynamic_emotion_quadrant",
    "split",
)

_LEGACY_COLUMN_RENAMES = {
    "track_id": "song_id",
    "window_start": "window_start_sec",
    "window_end": "window_end_sec",
    "start_sec": "window_start_sec",
    "end_sec": "window_end_sec",
    "emotion_quadrant": "dynamic_emotion_quadrant",
    "annotation_time": "annotation_time_sec",
}


def standardize_dynamic_window_columns(window_df: pd.DataFrame) -> pd.DataFrame:
    """Rename legacy dynamic window columns to the standard schema."""
    window_df = window_df.copy()

    rename_map = {
        old: new
        for old, new in _LEGACY_COLUMN_RENAMES.items()
        if old in window_df.columns and new not in window_df.columns
    }
    if rename_map:
        window_df = window_df.rename(columns=rename_map)

    if "annotation_time_sec" not in window_df.columns:
        if "time_sec" in window_df.columns:
            window_df = window_df.rename(columns={"time_sec": "annotation_time_sec"})
        elif {"window_start_sec", "window_end_sec"}.issubset(window_df.columns):
            window_df["annotation_time_sec"] = (
                window_df["window_start_sec"] + window_df["window_end_sec"]
            ) / 2.0

    if "song_id" in window_df.columns:
        window_df["song_id"] = pd.to_numeric(window_df["song_id"], errors="coerce")
        if window_df["song_id"].isna().any():
            raise ValueError("Dynamic windows contain invalid song_id values.")
        window_df["song_id"] = window_df["song_id"].astype(int)

    return window_df


def _dynamic_windows_need_resave(before: pd.DataFrame, after: pd.DataFrame) -> bool:
    return list(before.columns) != list(after.columns) or any(
        col in before.columns for col in _LEGACY_COLUMN_RENAMES
    )


def _window_starts(duration_sec: float, window_size_sec: float, hop_size_sec: float) -> np.ndarray:
    if duration_sec < window_size_sec:
        return np.array([], dtype=float)
    max_start = duration_sec - window_size_sec
    return np.arange(0.0, max_start + 1e-9, hop_size_sec)


def _resolve_threshold_value(values: pd.Series, threshold: str | float) -> float:
    if isinstance(threshold, str) and threshold.lower() == "median":
        return float(values.median())
    return float(threshold)


def _align_window_to_annotations(
    track_dynamic: pd.DataFrame,
    window_start: float,
    window_end: float,
    skip_initial_sec: float,
) -> tuple[float | None, float | None, float | None]:
    """
    Align an audio window with overlapping or nearest dynamic annotations.

    Returns (valence, arousal, annotation_time_sec).
    """
    track_dynamic = track_dynamic.sort_values("time_sec")
    window_center = (window_start + window_end) / 2.0

    overlapping_mask = (
        (track_dynamic["time_sec"] >= window_start)
        & (track_dynamic["time_sec"] < window_end)
        & (track_dynamic["time_sec"] >= skip_initial_sec)
    )
    overlapping = track_dynamic.loc[
        overlapping_mask, ["time_sec", "valence", "arousal"]
    ].dropna(subset=["valence", "arousal"])

    if not overlapping.empty:
        return (
            float(overlapping["valence"].mean()),
            float(overlapping["arousal"].mean()),
            float(overlapping["time_sec"].median()),
        )

    usable = track_dynamic[track_dynamic["time_sec"] >= skip_initial_sec].dropna(
        subset=["valence", "arousal"]
    )
    if usable.empty:
        return None, None, None

    nearest_idx = (usable["time_sec"] - window_center).abs().idxmin()
    row = usable.loc[nearest_idx]
    return float(row["valence"]), float(row["arousal"]), float(row["time_sec"])


def _get_track_split_table(
    metadata: pd.DataFrame,
    configs: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Load persisted track splits or create them from static labels."""
    try:
        split_df = load_track_splits(configs)
        logger.info("Loaded existing track splits from config.")
        return split_df
    except FileNotFoundError:
        from src.data.make_static_dataset import build_static_label_dataset

        static_df = build_static_label_dataset(metadata, configs)
        split_df = create_track_split_table(
            static_df,
            track_col="song_id",
            label_col="emotion_quadrant",
            configs=configs,
        )
        logger.info("Created track splits from static label dataset.")
        return split_df


def build_dynamic_window_dataset(
    metadata: pd.DataFrame | None = None,
    dynamic_df: pd.DataFrame | None = None,
    configs: dict[str, dict[str, Any]] | None = None,
    complete_only: bool = True,
) -> pd.DataFrame:
    """
    Create a window-level dataset aligned with dynamic valence/arousal labels.

    Each row represents one time window within a track with columns:
    song_id, window_index, window_start_sec, window_end_sec, annotation_time_sec,
    valence, arousal, dynamic_emotion_quadrant, split
    """
    if configs is None:
        configs = load_configs()
    if metadata is None:
        metadata = build_metadata_table(configs)
    if dynamic_df is None:
        dynamic_df = load_dynamic_annotations(configs)

    dynamic_cfg = configs["features"]["dynamic"]
    emotion_cfg = configs["features"]["emotion"]
    window_size_sec = float(dynamic_cfg["window_size_sec"])
    hop_size_sec = float(dynamic_cfg["hop_size_sec"])
    skip_initial_sec = float(dynamic_cfg["skip_initial_sec"])

    tracks = metadata.copy()
    if complete_only:
        tracks = tracks[tracks["is_complete"]].copy()

    split_df = _get_track_split_table(metadata, configs)

    usable = dynamic_df[dynamic_df["time_sec"] >= skip_initial_sec].dropna(subset=["valence", "arousal"])
    v_cut = _resolve_threshold_value(usable["valence"], emotion_cfg["valence_threshold"])
    a_cut = _resolve_threshold_value(usable["arousal"], emotion_cfg["arousal_threshold"])

    rows: list[dict[str, Any]] = []
    dynamic_by_song = {song_id: group for song_id, group in dynamic_df.groupby("song_id")}

    for _, track in tqdm(tracks.iterrows(), total=len(tracks), desc="Building dynamic windows"):
        song_id = int(track["song_id"])
        track_dynamic = dynamic_by_song.get(song_id)
        if track_dynamic is None or track_dynamic.empty:
            logger.warning("No dynamic annotations for song_id=%s", song_id)
            continue

        duration_sec = float(track_dynamic["time_sec"].max()) + (
            track_dynamic["time_sec"].diff().median() if len(track_dynamic) > 1 else 1.0
        )
        starts = _window_starts(duration_sec, window_size_sec, hop_size_sec)

        for window_index, window_start in enumerate(starts):
            window_end = window_start + window_size_sec
            valence, arousal, annotation_time_sec = _align_window_to_annotations(
                track_dynamic,
                window_start,
                window_end,
                skip_initial_sec,
            )
            if valence is None or arousal is None or annotation_time_sec is None:
                continue

            quadrant_df = assign_emotion_quadrant(
                pd.Series([valence]),
                pd.Series([arousal]),
                valence_threshold=v_cut,
                arousal_threshold=a_cut,
            )
            rows.append(
                {
                    "song_id": song_id,
                    "window_index": window_index,
                    "window_start_sec": window_start,
                    "window_end_sec": window_end,
                    "annotation_time_sec": annotation_time_sec,
                    "valence": valence,
                    "arousal": arousal,
                    "dynamic_emotion_quadrant": quadrant_df["emotion_quadrant"].iloc[0],
                }
            )

    window_df = pd.DataFrame(rows)
    if window_df.empty:
        logger.warning("Dynamic window dataset is empty. Check DEAM paths and window settings.")
        return window_df

    window_df = assign_splits_to_dataframe(window_df, split_df, track_col="song_id")
    verify_window_splits(window_df, track_col="song_id")
    return window_df


def build_dynamic_trajectory_summary(
    dynamic_df: pd.DataFrame | None = None,
    window_df: pd.DataFrame | None = None,
    configs: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """
    Summarize per-track dynamic emotion trajectories.

    Includes quadrant transitions, stability flags, dominant quadrant, and
    percentage of time spent in each quadrant.
    """
    if configs is None:
        configs = load_configs()

    emotion_cfg = configs["features"]["emotion"]
    dynamic_cfg = configs["features"]["dynamic"]
    skip_initial_sec = float(dynamic_cfg["skip_initial_sec"])
    quadrant_order = tuple(emotion_cfg.get("class_order", ("Q1", "Q2", "Q3", "Q4")))

    if dynamic_df is None:
        dynamic_df = load_dynamic_annotations(configs)

    usable = dynamic_df[dynamic_df["time_sec"] >= skip_initial_sec].copy()
    v_cut = _resolve_threshold_value(usable["valence"], emotion_cfg["valence_threshold"])
    a_cut = _resolve_threshold_value(usable["arousal"], emotion_cfg["arousal_threshold"])

    summaries: list[dict[str, Any]] = []
    for song_id, group in usable.groupby("song_id"):
        group = group.sort_values("time_sec")
        labels = assign_emotion_quadrant(
            group["valence"],
            group["arousal"],
            valence_threshold=v_cut,
            arousal_threshold=a_cut,
        )
        quadrants = labels["emotion_quadrant"].tolist()
        transitions = sum(q1 != q2 for q1, q2 in zip(quadrants, quadrants[1:]))
        quadrant_counts = pd.Series(quadrants).value_counts()
        total = len(quadrants)

        summary: dict[str, Any] = {
            "song_id": int(song_id),
            "n_samples": total,
            "n_quadrant_transitions": transitions,
            "is_emotionally_stable": transitions == 0,
            "is_emotionally_changing": transitions > 0,
            "dominant_emotion_quadrant": quadrant_counts.idxmax() if total else None,
            "dominant_emotion_class": (
                QUADRANT_LABELS.get(quadrant_counts.idxmax()) if total else None
            ),
        }
        for quadrant in quadrant_order:
            count = int(quadrant_counts.get(quadrant, 0))
            summary[f"pct_time_{quadrant}"] = (count / total * 100.0) if total else 0.0

        summaries.append(summary)

    summary_df = pd.DataFrame(summaries).sort_values("song_id").reset_index(drop=True)

    if window_df is not None and "split" in window_df.columns:
        split_map = window_df.groupby("song_id")["split"].first()
        summary_df["split"] = summary_df["song_id"].map(split_map)

    # Backward-compatible aliases used in notebook 01
    summary_df["track_id"] = summary_df["song_id"]
    summary_df["n_transitions"] = summary_df["n_quadrant_transitions"]
    summary_df["is_stable"] = summary_df["is_emotionally_stable"]
    summary_df["dominant_quadrant"] = summary_df["dominant_emotion_quadrant"]

    return summary_df


def save_dynamic_dataset(
    window_df: pd.DataFrame,
    configs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Path]:
    """Save dynamic window dataset to CSV (always) and parquet when available."""
    if configs is None:
        configs = load_configs()
    root = get_project_root()
    paths_cfg = configs["paths"]["processed"]

    csv_path = resolve_path(root, paths_cfg["dynamic_windows_csv"])
    parquet_path = resolve_path(root, paths_cfg["dynamic_windows"])
    ensure_dir(csv_path.parent)

    window_df = standardize_dynamic_window_columns(window_df)
    window_df.to_csv(csv_path, index=False)
    saved: dict[str, Path] = {"csv": csv_path}

    try:
        window_df.to_parquet(parquet_path, index=False)
        saved["parquet"] = parquet_path
        logger.info(
            "Saved dynamic window dataset to %s and %s (%d windows)",
            parquet_path,
            csv_path,
            len(window_df),
        )
    except Exception as exc:
        logger.warning("Could not save dynamic windows parquet to %s: %s", parquet_path, exc)
        logger.info("Saved dynamic window dataset to %s (%d windows)", csv_path, len(window_df))

    return saved


def save_dynamic_trajectory_summary(
    summary_df: pd.DataFrame,
    configs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Path]:
    """Save per-track dynamic emotion change summary to CSV."""
    if configs is None:
        configs = load_configs()
    root = get_project_root()

    csv_path = resolve_path(root, configs["paths"]["reports"]["dynamic_emotion_change_summary"])
    parquet_path = resolve_path(root, configs["paths"]["processed"]["dynamic_trajectories"])
    ensure_dir(csv_path.parent)

    summary_df.to_csv(csv_path, index=False)
    saved: dict[str, Path] = {"csv": csv_path}

    try:
        summary_df.to_parquet(parquet_path, index=False)
        saved["parquet"] = parquet_path
        logger.info("Saved dynamic trajectory summary to %s and %s", parquet_path, csv_path)
    except Exception as exc:
        logger.warning("Could not save trajectory summary parquet to %s: %s", parquet_path, exc)
        logger.info("Saved dynamic trajectory summary to %s", csv_path)

    return saved


def attach_track_splits_to_dynamic_windows(
    window_df: pd.DataFrame,
    configs: dict[str, dict[str, Any]] | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Merge track-level train/val/test splits into dynamic windows by song_id.

    Loads splits from the static track split file when the window table has no
    ``split`` column or contains null split values.
    """
    if configs is None:
        configs = load_configs()

    before = window_df.copy()
    window_df = standardize_dynamic_window_columns(window_df)

    needs_split = "split" not in window_df.columns or window_df["split"].isna().any()
    if needs_split:
        split_df = load_track_splits(configs)
        if "split" in window_df.columns:
            window_df = window_df.drop(columns=["split"])
        window_df = window_df.merge(split_df[["song_id", "split"]], on="song_id", how="left")

    should_save = save and (needs_split or _dynamic_windows_need_resave(before, window_df))
    if should_save:
        save_dynamic_dataset(window_df, configs)
        if needs_split:
            logger.info("Saved dynamic windows with track splits.")
        else:
            logger.info("Saved dynamic windows with standardized column names.")

    if window_df["split"].isna().any():
        missing = window_df.loc[window_df["split"].isna(), "song_id"].unique()
        raise ValueError(f"Tracks without split assignment: {missing[:10]}")

    verify_window_splits(window_df, track_col="song_id")
    return window_df


def load_dynamic_windows(
    configs: dict[str, dict[str, Any]] | None = None,
    attach_splits: bool = True,
    save: bool = True,
) -> pd.DataFrame:
    """Load dynamic window dataset and attach track splits when missing."""
    if configs is None:
        configs = load_configs()
    root = get_project_root()
    paths_cfg = configs["paths"]["processed"]

    parquet_path = resolve_path(root, paths_cfg["dynamic_windows"])
    csv_path = resolve_path(root, paths_cfg["dynamic_windows_csv"])

    if parquet_path.exists():
        window_df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        window_df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(
            f"Dynamic windows not found at {parquet_path} or {csv_path}. "
            "Run dynamic window dataset construction first."
        )

    if attach_splits:
        window_df = attach_track_splits_to_dynamic_windows(window_df, configs, save=save)
    else:
        before = window_df.copy()
        window_df = standardize_dynamic_window_columns(window_df)
        if save and _dynamic_windows_need_resave(before, window_df):
            save_dynamic_dataset(window_df, configs)

    return window_df
