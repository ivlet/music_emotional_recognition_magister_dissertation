"""Per-track Mel-spectrogram extraction for CNN/CRNN models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.data.load_deam import build_metadata_table
from src.data.make_static_dataset import assign_emotion_quadrant
from src.data.splits import assign_splits_to_dataframe, load_track_splits
from src.utils.config import ensure_dir, get_project_root, load_configs, resolve_path

logger = logging.getLogger(__name__)


def _mel_paths(configs: dict[str, dict[str, Any]]) -> tuple[Path, Path]:
    root = get_project_root()
    paths_cfg = configs["paths"]["features"]
    mel_dir = resolve_path(root, paths_cfg["mel_spectrogram_dir"])
    index_path = resolve_path(root, paths_cfg["mel_spectrogram_index"])
    return mel_dir, index_path


def _max_time_frames(configs: dict[str, dict[str, Any]]) -> int:
    spec_cfg = configs["features"]["spectrogram"]
    sample_rate = int(spec_cfg["sample_rate"])
    hop_length = int(spec_cfg["hop_length"])
    max_duration_sec = float(spec_cfg["max_duration_sec"])
    max_samples = int(max_duration_sec * sample_rate)
    return int(np.ceil(max_samples / hop_length))


def extract_track_mel_spectrogram(
    audio_path: str | Path,
    configs: dict[str, dict[str, Any]] | None = None,
) -> np.ndarray:
    """
    Extract a fixed-size log-Mel spectrogram for one track.

    Returns array of shape (n_mels, n_frames).
    """
    if configs is None:
        configs = load_configs()

    spec_cfg = configs["features"]["spectrogram"]
    sample_rate = int(spec_cfg["sample_rate"])
    n_fft = int(spec_cfg["n_fft"])
    hop_length = int(spec_cfg["hop_length"])
    n_mels = int(spec_cfg["n_mels"])
    fmin = float(spec_cfg["fmin"])
    fmax = float(spec_cfg["fmax"])
    max_duration_sec = float(spec_cfg["max_duration_sec"])
    normalize = bool(spec_cfg.get("normalize", True))
    target_frames = _max_time_frames(configs)

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    y, sr = librosa.load(audio_path, sr=sample_rate, mono=True, duration=max_duration_sec)
    if y.size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] < target_frames:
        pad_width = target_frames - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant")
    elif mel_db.shape[1] > target_frames:
        mel_db = mel_db[:, :target_frames]

    if normalize:
        mean = float(mel_db.mean())
        std = float(mel_db.std())
        mel_db = (mel_db - mean) / (std + 1e-6)

    return mel_db.astype(np.float32)


def build_mel_spectrogram_index(
    metadata: pd.DataFrame,
    labels_df: pd.DataFrame,
    split_df: pd.DataFrame,
    configs: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Assign splits and train-fitted emotion quadrants for spectrogram index rows."""
    label_cols = ["song_id", "valence", "arousal"]
    frame = metadata[["song_id", "audio_path"]].merge(labels_df[label_cols], on="song_id", how="inner")
    frame = assign_splits_to_dataframe(frame, split_df, track_col="song_id")

    train_mask = frame["split"] == "train"
    train_labels = assign_emotion_quadrant(
        frame.loc[train_mask, "valence"],
        frame.loc[train_mask, "arousal"],
        valence_threshold=configs["features"]["emotion"]["valence_threshold"],
        arousal_threshold=configs["features"]["emotion"]["arousal_threshold"],
    )
    v_cut = float(train_labels["valence_threshold"].iloc[0])
    a_cut = float(train_labels["arousal_threshold"].iloc[0])
    all_labels = assign_emotion_quadrant(
        frame["valence"],
        frame["arousal"],
        valence_threshold=v_cut,
        arousal_threshold=a_cut,
    )
    frame["emotion_quadrant"] = all_labels["emotion_quadrant"].values
    return frame


def extract_mel_spectrograms_dataset(
    metadata: pd.DataFrame | None = None,
    configs: dict[str, dict[str, Any]] | None = None,
    complete_only: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """
    Extract Mel-spectrograms for all tracks and write index CSV.

    Saves ``.npy`` files under ``data/features/mel_spectrograms/``.
    """
    if configs is None:
        configs = load_configs()
    if metadata is None:
        metadata = build_metadata_table(configs)

    mel_dir, index_path = _mel_paths(configs)
    ensure_dir(mel_dir)

    root = get_project_root()
    labels_path = resolve_path(root, configs["paths"]["processed"]["static_labels"])
    labels_df = pd.read_parquet(labels_path) if labels_path.exists() else pd.read_csv(
        labels_path.with_suffix(".csv")
    )
    split_df = load_track_splits(configs)

    tracks = metadata.copy()
    if complete_only:
        tracks = tracks[tracks["is_complete"]].copy()

    if index_path.exists() and not force:
        index_df = pd.read_csv(index_path)
        if len(index_df) >= len(tracks):
            logger.info("Mel-spectrogram index already exists at %s (%d tracks).", index_path, len(index_df))
            return index_df

    index_frame = build_mel_spectrogram_index(tracks, labels_df, split_df, configs)
    rows: list[dict[str, Any]] = []
    failed: list[int] = []

    for _, track in tqdm(index_frame.iterrows(), total=len(index_frame), desc="Extracting Mel-spectrograms"):
        song_id = int(track["song_id"])
        audio_path = track["audio_path"]
        if pd.isna(audio_path):
            failed.append(song_id)
            continue
        try:
            mel = extract_track_mel_spectrogram(audio_path, configs)
        except Exception as exc:
            logger.warning("Mel extraction failed for song_id=%s: %s", song_id, exc)
            failed.append(song_id)
            continue

        rel_path = Path("mel_spectrograms") / f"{song_id}.npy"
        np.save(mel_dir / f"{song_id}.npy", mel)
        rows.append(
            {
                "song_id": song_id,
                "spectrogram_path": str(rel_path).replace("\\", "/"),
                "split": track["split"],
                "emotion_quadrant": track["emotion_quadrant"],
                "n_mels": int(mel.shape[0]),
                "n_frames": int(mel.shape[1]),
            }
        )

    if not rows:
        raise RuntimeError("No Mel-spectrograms extracted. Check audio paths.")

    index_df = pd.DataFrame(rows).sort_values("song_id").reset_index(drop=True)
    index_df.to_csv(index_path, index=False)
    logger.info("Saved Mel-spectrogram index to %s (%d tracks)", index_path, len(index_df))

    if failed:
        logger.info("Skipped %d tracks due to missing or failed audio.", len(failed))
    return index_df


def load_mel_spectrogram_index(
    configs: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Load the Mel-spectrogram index table."""
    if configs is None:
        configs = load_configs()
    _, index_path = _mel_paths(configs)
    if not index_path.exists():
        raise FileNotFoundError(
            f"Mel-spectrogram index not found at {index_path}. "
            "Run extract_mel_spectrograms_dataset() first."
        )
    return pd.read_csv(index_path)


def get_mel_spectrogram_dir(configs: dict[str, dict[str, Any]] | None = None) -> Path:
    if configs is None:
        configs = load_configs()
    mel_dir, _ = _mel_paths(configs)
    return mel_dir
