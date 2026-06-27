"""Track-level train/validation/test splits without window leakage."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.config import ensure_dir, get_project_root, load_configs, resolve_path

logger = logging.getLogger(__name__)

SplitName = Literal["train", "val", "test"]


def _validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")
    for name, ratio in (("train", train_ratio), ("val", val_ratio), ("test", test_ratio)):
        if ratio < 0:
            raise ValueError(f"{name}_ratio must be non-negative, got {ratio}")


def split_track_ids(
    track_ids: np.ndarray | list[int],
    labels: np.ndarray | list[Any] | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify: bool = True,
    random_state: int = 42,
) -> dict[SplitName, list[int]]:
    """
    Split unique track IDs into train/val/test partitions.

    Splitting is always performed at the track level so that all windows or
    segments from the same track stay in one partition.
    """
    _validate_split_ratios(train_ratio, val_ratio, test_ratio)

    track_ids = np.asarray(track_ids)
    unique_ids = np.unique(track_ids)

    if len(unique_ids) < 3:
        raise ValueError("Need at least 3 tracks to create train/val/test splits.")

    stratify_labels = None
    if stratify and labels is not None:
        labels = np.asarray(labels)
        if len(labels) != len(unique_ids):
            raise ValueError("labels must have one entry per unique track_id.")
        stratify_labels = labels

    temp_ratio = val_ratio + test_ratio
    train_ids, temp_ids = train_test_split(
        unique_ids,
        test_size=temp_ratio,
        random_state=random_state,
        stratify=stratify_labels,
    )

    if stratify and labels is not None:
        label_map = dict(zip(unique_ids, labels))
        temp_labels = np.array([label_map[tid] for tid in temp_ids])
    else:
        temp_labels = None

    val_fraction = val_ratio / temp_ratio if temp_ratio > 0 else 0.0
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=(1.0 - val_fraction),
        random_state=random_state,
        stratify=temp_labels,
    )

    splits = {
        "train": sorted(int(t) for t in train_ids),
        "val": sorted(int(t) for t in val_ids),
        "test": sorted(int(t) for t in test_ids),
    }
    assert_no_track_leakage(splits)
    return splits


def assert_no_track_leakage(splits: dict[str, list[int]]) -> None:
    """Raise if any track_id appears in more than one split."""
    seen: dict[int, str] = {}
    for split_name, ids in splits.items():
        for track_id in ids:
            if track_id in seen:
                raise ValueError(
                    f"Track leakage detected: track_id={track_id} appears in "
                    f"both '{seen[track_id]}' and '{split_name}'."
                )
            seen[track_id] = split_name


def create_track_split_table(
    df: pd.DataFrame,
    track_col: str = "song_id",
    label_col: str | None = "emotion_quadrant",
    configs: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """
    Assign each track in ``df`` to train, val, or test.

    Returns a two-column DataFrame: ``song_id``, ``split``.
    """
    if configs is None:
        configs = load_configs()

    split_cfg = configs["training"]["splits"]
    random_state = int(configs["training"]["general"]["random_state"])

    track_table = df[[track_col]].drop_duplicates().sort_values(track_col)
    track_ids = track_table[track_col].to_numpy()

    labels = None
    if split_cfg.get("stratify", True) and label_col is not None and label_col in df.columns:
        labels = (
            df.groupby(track_col, as_index=False)[label_col]
            .first()
            .sort_values(track_col)[label_col]
            .to_numpy()
        )

    splits = split_track_ids(
        track_ids=track_ids,
        labels=labels,
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        stratify=bool(split_cfg.get("stratify", True) and labels is not None),
        random_state=random_state,
    )

    rows: list[dict[str, Any]] = []
    for split_name, ids in splits.items():
        rows.extend({"song_id": track_id, "split": split_name} for track_id in ids)

    split_df = pd.DataFrame(rows).sort_values("song_id").reset_index(drop=True)
    logger.info(
        "Track splits — train: %d, val: %d, test: %d",
        len(splits["train"]),
        len(splits["val"]),
        len(splits["test"]),
    )
    return split_df


def assign_splits_to_dataframe(
    df: pd.DataFrame,
    split_df: pd.DataFrame,
    track_col: str = "song_id",
) -> pd.DataFrame:
    """Attach a ``split`` column to ``df`` by merging on track id."""
    split_subset = split_df.copy()
    if track_col != "song_id":
        split_subset = split_subset.rename(columns={"song_id": track_col})
    merged = df.merge(split_subset[[track_col, "split"]], on=track_col, how="left")
    if merged["split"].isna().any():
        missing = merged.loc[merged["split"].isna(), track_col].unique()
        raise ValueError(f"Tracks without split assignment: {missing[:10]}")
    return merged


def assign_splits_to_windows(
    windows_df: pd.DataFrame,
    split_df: pd.DataFrame,
    track_col: str = "track_id",
) -> pd.DataFrame:
    """
    Assign train/val/test labels to window-level rows via their parent track.

    Ensures windows inherit the split of their track — no window-level random splitting.
    """
    renamed = split_df.rename(columns={"song_id": track_col})
    merged = windows_df.merge(renamed, on=track_col, how="left")
    if merged["split"].isna().any():
        missing = merged.loc[merged["split"].isna(), track_col].unique()
        raise ValueError(f"Window tracks without split assignment: {missing[:10]}")
    return merged


def verify_window_splits(windows_df: pd.DataFrame, track_col: str = "track_id") -> None:
    """Verify that each track maps to exactly one split in a window dataset."""
    track_splits = windows_df.groupby(track_col)["split"].nunique()
    leaky = track_splits[track_splits > 1]
    if not leaky.empty:
        raise ValueError(
            f"Window leakage detected: {len(leaky)} tracks appear in multiple splits."
        )


def save_track_splits(
    split_df: pd.DataFrame,
    configs: dict[str, dict[str, Any]] | None = None,
    output_key: str = "static_track_splits",
) -> Path:
    """Persist track split assignments."""
    if configs is None:
        configs = load_configs()
    root = get_project_root()
    output_path = resolve_path(root, configs["paths"]["processed"][output_key])
    ensure_dir(output_path.parent)
    split_df.to_parquet(output_path, index=False)
    logger.info("Saved track splits to %s", output_path)
    return output_path


def load_track_splits(
    configs: dict[str, dict[str, Any]] | None = None,
    output_key: str = "static_track_splits",
) -> pd.DataFrame:
    """Load persisted track split assignments from parquet or CSV."""
    if configs is None:
        configs = load_configs()
    root = get_project_root()
    paths_cfg = configs["paths"]["processed"]

    parquet_path = resolve_path(root, paths_cfg[output_key])
    csv_key = f"{output_key}_csv"
    csv_path = resolve_path(root, paths_cfg[csv_key]) if csv_key in paths_cfg else parquet_path.with_suffix(".csv")

    if parquet_path.exists():
        split_df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        split_df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"Track splits not found at {parquet_path} or {csv_path}")

    split_df = split_df.copy()
    split_df["song_id"] = pd.to_numeric(split_df["song_id"], errors="coerce").astype("Int64")
    if split_df["song_id"].isna().any():
        raise ValueError("Track splits contain invalid song_id values.")
    split_df["song_id"] = split_df["song_id"].astype(int)
    return split_df
