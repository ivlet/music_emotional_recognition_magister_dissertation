"""Static (track-level) spectral feature extraction."""

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


def extract_track_spectral_features(
    audio_path: str | Path,
    configs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, float]:
    """
    Extract aggregated spectral features from a single audio file.

    Features: MFCC, chroma, spectral centroid/bandwidth/rolloff, ZCR, RMS, tempo.
    Each feature group is aggregated with mean, std, min, max, median.
    """
    if configs is None:
        configs = load_configs()

    static_cfg = configs["features"]["static"]
    sample_rate = int(static_cfg["sample_rate"])
    aggregations = tuple(static_cfg["aggregations"])
    groups = static_cfg["feature_groups"]

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    if y.size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")

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

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
    features["tempo_bpm"] = float(np.atleast_1d(tempo)[0])

    return features


def extract_static_features_dataset(
    labels_df: pd.DataFrame,
    configs: dict[str, dict[str, Any]] | None = None,
    song_id_col: str = "song_id",
    audio_path_col: str = "audio_path",
) -> pd.DataFrame:
    """
    Extract static spectral features for all tracks in ``labels_df``.

    Returns a DataFrame with ``song_id`` plus one column per feature.
    Tracks with missing or unreadable audio are skipped and logged.
    """
    if configs is None:
        configs = load_configs()

    rows: list[dict[str, Any]] = []
    failed: list[int] = []

    for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc="Extracting static features"):
        song_id = int(row[song_id_col])
        audio_path = row.get(audio_path_col)
        if pd.isna(audio_path):
            logger.warning("Missing audio path for song_id=%s", song_id)
            failed.append(song_id)
            continue

        try:
            feature_dict = extract_track_spectral_features(audio_path, configs)
        except FileNotFoundError:
            logger.warning("Audio file not found for song_id=%s: %s", song_id, audio_path)
            failed.append(song_id)
            continue
        except Exception as exc:
            logger.warning("Feature extraction failed for song_id=%s: %s", song_id, exc)
            failed.append(song_id)
            continue

        feature_dict[song_id_col] = song_id
        rows.append(feature_dict)

    if not rows:
        raise RuntimeError("No features were extracted. Check audio paths and DEAM setup.")

    features_df = pd.DataFrame(rows)
    cols = [song_id_col] + sorted(col for col in features_df.columns if col != song_id_col)
    features_df = features_df[cols]

    if failed:
        logger.info("Skipped %d tracks due to missing or failed audio.", len(failed))

    return features_df


def save_static_features(
    features_df: pd.DataFrame,
    configs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Path]:
    """Save extracted static features to parquet and csv."""
    if configs is None:
        configs = load_configs()
    root = get_project_root()
    paths_cfg = configs["paths"]["features"]

    parquet_path = resolve_path(root, paths_cfg["static_features"])
    csv_path = resolve_path(root, paths_cfg["static_features_csv"])
    ensure_dir(parquet_path.parent)

    features_df.to_parquet(parquet_path, index=False)
    features_df.to_csv(csv_path, index=False)
    logger.info("Saved static features to %s and %s", parquet_path, csv_path)
    return {"parquet": parquet_path, "csv": csv_path}


def load_static_features(
    configs: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Load previously extracted static features."""
    if configs is None:
        configs = load_configs()
    root = get_project_root()
    parquet_path = resolve_path(root, configs["paths"]["features"]["static_features"])
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Static features not found at {parquet_path}. "
            "Run static feature extraction first."
        )
    return pd.read_parquet(parquet_path)
