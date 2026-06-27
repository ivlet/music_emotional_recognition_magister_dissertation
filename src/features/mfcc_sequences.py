"""Per-track MFCC sequence extraction for sequence models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.data.load_deam import build_metadata_table
from src.utils.config import ensure_dir, get_project_root, load_configs, resolve_path

logger = logging.getLogger(__name__)


def extract_track_mfcc_sequence(
    audio_path: str | Path,
    configs: dict[str, dict[str, Any]] | None = None,
) -> np.ndarray:
    """
    Extract MFCC sequence for one track.

    Returns array of shape (n_frames, n_mfcc).
    """
    if configs is None:
        configs = load_configs()

    seq_cfg = configs["features"]["sequence"]
    sample_rate = int(seq_cfg["sample_rate"])
    n_mfcc = int(seq_cfg["n_mfcc"])
    frame_length = int(seq_cfg["frame_length"])
    hop_length = int(seq_cfg["hop_length"])

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
    if y.size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=frame_length,
        hop_length=hop_length,
    )
    return mfcc.T.astype(np.float32)


def _sequence_paths(configs: dict[str, dict[str, Any]]) -> tuple[Path, Path]:
    root = get_project_root()
    paths_cfg = configs["paths"]["features"]
    mfcc_dir = resolve_path(root, paths_cfg["sequence_mfcc_dir"])
    manifest_path = resolve_path(root, paths_cfg["sequence_mfcc_manifest"])
    return mfcc_dir, manifest_path


def extract_mfcc_sequences_dataset(
    metadata: pd.DataFrame | None = None,
    configs: dict[str, dict[str, Any]] | None = None,
    complete_only: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """
    Extract and save MFCC sequences for all tracks.

    Saves one ``.npy`` file per song under ``data/features/sequence/mfcc/``
    and writes ``mfcc_manifest.csv``.
    """
    if configs is None:
        configs = load_configs()
    if metadata is None:
        metadata = build_metadata_table(configs)

    mfcc_dir, manifest_path = _sequence_paths(configs)
    ensure_dir(mfcc_dir)

    tracks = metadata.copy()
    if complete_only:
        tracks = tracks[tracks["is_complete"]].copy()

    if manifest_path.exists() and not force:
        manifest = pd.read_csv(manifest_path)
        expected = len(tracks)
        if len(manifest) >= expected:
            logger.info("MFCC manifest already exists at %s (%d tracks).", manifest_path, len(manifest))
            return manifest

    rows: list[dict[str, Any]] = []
    failed: list[int] = []

    for _, track in tqdm(tracks.iterrows(), total=len(tracks), desc="Extracting MFCC sequences"):
        song_id = int(track["song_id"])
        audio_path = track.get("audio_path")
        if pd.isna(audio_path):
            logger.warning("Missing audio path for song_id=%s", song_id)
            failed.append(song_id)
            continue

        try:
            sequence = extract_track_mfcc_sequence(audio_path, configs)
        except Exception as exc:
            logger.warning("MFCC extraction failed for song_id=%s: %s", song_id, exc)
            failed.append(song_id)
            continue

        out_path = mfcc_dir / f"{song_id}.npy"
        np.save(out_path, sequence)
        rows.append(
            {
                "song_id": song_id,
                "n_frames": int(sequence.shape[0]),
                "n_mfcc": int(sequence.shape[1]),
                "sequence_path": str(Path("mfcc") / f"{song_id}.npy"),
            }
        )

    if not rows:
        raise RuntimeError("No MFCC sequences extracted. Check audio paths.")

    manifest = pd.DataFrame(rows).sort_values("song_id").reset_index(drop=True)
    manifest.to_csv(manifest_path, index=False)
    logger.info("Saved MFCC manifest to %s (%d tracks)", manifest_path, len(manifest))

    if failed:
        logger.info("Skipped %d tracks due to missing or failed audio.", len(failed))

    return manifest


def load_mfcc_manifest(
    configs: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Load the MFCC sequence manifest."""
    if configs is None:
        configs = load_configs()
    _, manifest_path = _sequence_paths(configs)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"MFCC manifest not found at {manifest_path}. "
            "Run extract_mfcc_sequences_dataset() first."
        )
    return pd.read_csv(manifest_path)


def load_track_mfcc_sequence(
    song_id: int,
    configs: dict[str, dict[str, Any]] | None = None,
) -> np.ndarray:
    """Load a single track MFCC sequence from disk."""
    if configs is None:
        configs = load_configs()
    mfcc_dir, manifest_path = _sequence_paths(configs)
    path = mfcc_dir / f"{song_id}.npy"
    if not path.exists():
        raise FileNotFoundError(f"MFCC sequence not found for song_id={song_id} at {path}")
    return np.load(path)
