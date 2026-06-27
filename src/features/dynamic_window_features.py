"""Window-level spectral feature extraction for dynamic emotion analysis."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.features.feature_utils import aggregate_feature_matrix
from src.utils.config import ensure_dir, get_project_root, load_configs, resolve_path

logger = logging.getLogger(__name__)

REQUIRED_WINDOW_COLUMNS = (
    "song_id",
    "window_index",
    "window_start_sec",
    "window_end_sec",
)


def _validate_window_columns(windows_df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_WINDOW_COLUMNS if col not in windows_df.columns]
    if missing:
        raise ValueError(
            "Dynamic windows are missing required columns: "
            f"{missing}. Available columns: {list(windows_df.columns)}. "
            "Reload with load_dynamic_windows(configs) to apply column standardization, "
            "or rebuild the dynamic window dataset."
        )


def extract_window_spectral_features(
    y: np.ndarray,
    sr: int,
    configs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, float]:
    """
    Extract aggregated spectral features from a single audio window segment.

    Features: MFCC, chroma, spectral centroid/bandwidth/rolloff, ZCR, RMS.
    Each feature group is aggregated with mean, std, min, max, median.
    """
    if configs is None:
        configs = load_configs()

    dynamic_cfg = configs["features"]["dynamic"]
    aggregations = tuple(dynamic_cfg["aggregations"])
    groups = dynamic_cfg["feature_groups"]

    if y.size == 0:
        raise ValueError("Audio window segment is empty.")

    features: dict[str, float] = {}

    n_mfcc = int(groups["mfcc"]["n_mfcc"])
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    features.update(aggregate_feature_matrix(mfcc, "mfcc", aggregations))

    n_chroma = int(groups["chroma"]["n_chroma"])
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma)
    features.update(aggregate_feature_matrix(chroma, "chroma", aggregations))

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.update(aggregate_feature_matrix(spectral_centroid, "spectral_centroid", aggregations))

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.update(aggregate_feature_matrix(spectral_bandwidth, "spectral_bandwidth", aggregations))

    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.update(aggregate_feature_matrix(spectral_rolloff, "spectral_rolloff", aggregations))

    zcr = librosa.feature.zero_crossing_rate(y)
    features.update(aggregate_feature_matrix(zcr, "zcr", aggregations))

    rms = librosa.feature.rms(y=y)
    features.update(aggregate_feature_matrix(rms, "rms", aggregations))

    return features


def _slice_window(y: np.ndarray, sr: int, start_sec: float, end_sec: float) -> np.ndarray:
    start_sample = max(0, int(start_sec * sr))
    end_sample = min(len(y), int(end_sec * sr))
    if end_sample <= start_sample:
        return np.array([], dtype=y.dtype)
    return y[start_sample:end_sample]


def extract_dynamic_window_features_dataset(
    windows_df: pd.DataFrame,
    metadata: pd.DataFrame,
    configs: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """
    Extract spectral features for each dynamic window.

    Returns a DataFrame keyed by song_id and window_index with feature columns.
    """
    if configs is None:
        configs = load_configs()

    from src.data.make_dynamic_dataset import standardize_dynamic_window_columns

    windows_df = standardize_dynamic_window_columns(windows_df)
    _validate_window_columns(windows_df)

    dynamic_cfg = configs["features"]["dynamic"]
    sample_rate = int(dynamic_cfg["sample_rate"])

    audio_map = (
        metadata[["song_id", "audio_path"]]
        .dropna(subset=["audio_path"])
        .drop_duplicates("song_id")
        .set_index("song_id")["audio_path"]
        .to_dict()
    )

    rows: list[dict[str, Any]] = []
    failed_windows = 0

    grouped = windows_df.groupby("song_id", sort=False)
    for song_id, song_windows in tqdm(grouped, total=grouped.ngroups, desc="Extracting window features"):
        song_id = int(song_id)
        audio_path = audio_map.get(song_id)
        if audio_path is None:
            logger.warning("Missing audio path for song_id=%s — skipping %d windows", song_id, len(song_windows))
            failed_windows += len(song_windows)
            continue

        audio_path = Path(audio_path)
        if not audio_path.exists():
            logger.warning("Audio file not found for song_id=%s: %s", song_id, audio_path)
            failed_windows += len(song_windows)
            continue

        try:
            y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        except Exception as exc:
            logger.warning("Failed to load audio for song_id=%s: %s", song_id, exc)
            failed_windows += len(song_windows)
            continue

        if y.size == 0:
            logger.warning("Empty audio for song_id=%s", song_id)
            failed_windows += len(song_windows)
            continue

        for _, window_row in song_windows.iterrows():
            segment = _slice_window(
                y,
                sr,
                float(window_row["window_start_sec"]),
                float(window_row["window_end_sec"]),
            )
            try:
                feature_dict = extract_window_spectral_features(segment, sr, configs)
            except Exception as exc:
                logger.warning(
                    "Feature extraction failed for song_id=%s window_index=%s: %s",
                    song_id,
                    window_row.get("window_index"),
                    exc,
                )
                failed_windows += 1
                continue

            feature_dict["song_id"] = song_id
            feature_dict["window_index"] = int(window_row["window_index"])
            rows.append(feature_dict)

    if not rows:
        raise RuntimeError("No dynamic window features were extracted. Check audio paths and windows.")

    features_df = pd.DataFrame(rows)
    id_cols = ["song_id", "window_index"]
    feature_cols = sorted(col for col in features_df.columns if col not in id_cols)
    features_df = features_df[id_cols + feature_cols]

    if failed_windows:
        logger.info("Skipped %d windows due to missing or failed audio.", failed_windows)

    return features_df


def save_dynamic_window_features(
    features_df: pd.DataFrame,
    configs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Path]:
    """Save extracted dynamic window features to CSV (always) and parquet when available."""
    if configs is None:
        configs = load_configs()
    root = get_project_root()
    paths_cfg = configs["paths"]["features"]

    csv_path = resolve_path(root, paths_cfg["dynamic_window_features_csv"])
    parquet_path = resolve_path(root, paths_cfg["dynamic_window_features"])
    ensure_dir(csv_path.parent)

    features_df.to_csv(csv_path, index=False)
    saved: dict[str, Path] = {"csv": csv_path}

    try:
        features_df.to_parquet(parquet_path, index=False)
        saved["parquet"] = parquet_path
        logger.info("Saved dynamic window features to %s and %s", parquet_path, csv_path)
    except Exception as exc:
        logger.warning("Could not save dynamic window features parquet to %s: %s", parquet_path, exc)
        logger.info("Saved dynamic window features to %s", csv_path)

    return saved


def load_dynamic_window_features(
    configs: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Load previously extracted dynamic window features."""
    if configs is None:
        configs = load_configs()
    root = get_project_root()
    paths_cfg = configs["paths"]["features"]

    parquet_path = resolve_path(root, paths_cfg["dynamic_window_features"])
    csv_path = resolve_path(root, paths_cfg["dynamic_window_features_csv"])

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(
        f"Dynamic window features not found at {parquet_path} or {csv_path}. "
        "Run dynamic window feature extraction first."
    )
