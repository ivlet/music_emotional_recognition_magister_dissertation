"""Load and audit the DEAM dataset annotations and audio file inventory."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.utils.config import ensure_dir, get_project_root, load_configs, resolve_path

logger = logging.getLogger(__name__)

STATIC_ID_CANDIDATES = ("song_id", "Song_ID", "songId", "musicId", "id", "ID")
VALENCE_CANDIDATES = ("valence_mean", "Valence.mean.", "valence", "Valence", "mean_valence")
AROUSAL_CANDIDATES = ("arousal_mean", "Arousal.mean.", "arousal", "Arousal", "mean_arousal")


def _pick_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {col.strip().lower(): col for col in columns}
    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized:
            return normalized[key]
    return None


def _normalize_static_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize static annotation column names."""
    frame = df.copy()
    if frame.columns[0] in ("Unnamed: 0", "") or frame.iloc[:, 0].dtype in ("int64", "float64"):
        first_col = frame.columns[0]
        if first_col.startswith("Unnamed") or first_col == "":
            frame = frame.rename(columns={first_col: "song_id"})

    id_col = _pick_column(list(frame.columns), STATIC_ID_CANDIDATES)
    valence_col = _pick_column(list(frame.columns), VALENCE_CANDIDATES)
    arousal_col = _pick_column(list(frame.columns), AROUSAL_CANDIDATES)

    missing = [
        name
        for name, col in (("song_id", id_col), ("valence", valence_col), ("arousal", arousal_col))
        if col is None
    ]
    if missing:
        raise ValueError(
            f"Could not identify required static columns: {missing}. "
            f"Available columns: {list(frame.columns)}"
        )

    out = frame[[id_col, valence_col, arousal_col]].copy()
    out.columns = ["song_id", "valence", "arousal"]
    out["song_id"] = pd.to_numeric(out["song_id"], errors="coerce").astype("Int64")
    out["valence"] = pd.to_numeric(out["valence"], errors="coerce")
    out["arousal"] = pd.to_numeric(out["arousal"], errors="coerce")
    return out.dropna(subset=["song_id"]).astype({"song_id": int})


def _resolve_existing_dir(base: Path, candidates: tuple[str, ...]) -> Path:
    """Return the first existing subdirectory under ``base``."""
    for name in candidates:
        path = base / name
        if path.is_dir():
            return path
    raise FileNotFoundError(
        f"Could not find DEAM subdirectory under {base}. Tried: {list(candidates)}"
    )


def _resolve_deam_subdir(base: Path, configured_name: str, fallbacks: tuple[str, ...]) -> Path:
    """Resolve a DEAM subfolder, trying common naming variants."""
    seen: set[str] = set()
    candidates: list[str] = []
    for name in (configured_name, *fallbacks):
        if name not in seen:
            candidates.append(name)
            seen.add(name)
    return _resolve_existing_dir(base, tuple(candidates))


def _deam_root(project_root: Path, deam_cfg: dict[str, Any]) -> Path:
    """Resolved DEAM dataset root directory."""
    return resolve_path(project_root, deam_cfg["root"])


def _deam_subpath(project_root: Path, deam_cfg: dict[str, Any], *parts: str) -> Path:
    """Path under ``deam.root``."""
    return _deam_root(project_root, deam_cfg).joinpath(*parts)


def _resolve_annotations_root(project_root: Path, deam_cfg: dict[str, Any]) -> Path:
    """Resolve the annotations directory (contains averaged / per-rater subfolders)."""
    ann_cfg = deam_cfg["annotations"]
    averaged_names = (
        ann_cfg["averaged_per_song"],
        "annotations_averaged_per_song",
        "annotations averaged per song",
    )
    candidates = [
        _deam_subpath(project_root, deam_cfg, ann_cfg["dir"]),
        _deam_subpath(project_root, deam_cfg, "DEAM_annotations/annotations"),
        _deam_subpath(project_root, deam_cfg, "DEAM_Annotations/annotations"),
    ]
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.is_dir():
            continue
        seen.add(path)
        if any((path / name).is_dir() for name in averaged_names):
            if path != candidates[0]:
                logger.info("Using annotations directory: %s", path)
            return path

    tried = list(seen) if seen else candidates
    raise FileNotFoundError(
        "DEAM annotations directory not found (expected a folder containing "
        f"{averaged_names[0]}). Tried: {tried}"
    )


def _dynamic_folder_name_candidates(deam_cfg: dict[str, Any]) -> tuple[str, ...]:
    configured = deam_cfg["annotations"].get("dynamic_per_second", "dynamic_per_second_annotations")
    return (
        configured,
        "dynamic_per_second_annotations",
        "dynamic (per second annotations)",
    )


def _per_rater_folder_name_candidates(deam_cfg: dict[str, Any]) -> tuple[str, ...]:
    configured = deam_cfg["annotations"].get("per_each_rater", "annotations_per_each_rater")
    return (
        configured,
        "annotations_per_each_rater",
        "annotations per each rater",
    )


def _resolve_dynamic_dir(
    annotations_dir: Path,
    averaged_dir: Path,
    deam_cfg: dict[str, Any],
) -> Path:
    """
    Find dynamic annotations folder.

    Search order:
      1. annotations_averaged_per_song / dynamic_per_second_annotations
      2. annotations_averaged_per_song / dynamic (per second annotations)
      3. annotations_per_each_rater / dynamic_per_second_annotations
      4. annotations_per_each_rater / dynamic (per second annotations)
    """
    dynamic_names = _dynamic_folder_name_candidates(deam_cfg)
    per_rater_names = _per_rater_folder_name_candidates(deam_cfg)

    candidates: list[Path] = []
    for dynamic_name in dynamic_names:
        candidates.append(averaged_dir / dynamic_name)
    for per_rater_name in per_rater_names:
        per_rater_dir = annotations_dir / per_rater_name
        for dynamic_name in dynamic_names:
            candidates.append(per_rater_dir / dynamic_name)

    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.is_dir():
            logger.info("Found dynamic annotation folder: %s", path)
            return path

    raise FileNotFoundError(
        "Could not find dynamic annotations. Tried:\n"
        + "\n".join(f"  - {p}" for p in candidates)
    )


def _parse_sample_ms_column(column_name: str) -> float | None:
    """Parse DEAM columns like ``sample_15000ms`` into seconds."""
    match = re.match(r"sample_(\d+)ms", str(column_name).strip(), re.IGNORECASE)
    if match:
        return int(match.group(1)) / 1000.0
    return None


def _is_dynamic_songs_as_rows_format(df: pd.DataFrame) -> bool:
    """Detect DEAM format: one row per song, ``sample_*ms`` columns as time axis."""
    first_col = str(df.columns[0]).strip().lower()
    if first_col in {"song_id", "songid", "id", "musicid"}:
        return True
    return any(str(col).lower().startswith("sample_") for col in df.columns[1:6])


def _normalize_dynamic_songs_as_rows(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """Convert song-row / sample-column dynamic CSVs to long format."""
    frame = df.copy()
    id_col = frame.columns[0]
    if str(id_col).lower() != "song_id":
        frame = frame.rename(columns={id_col: "song_id"})

    long_df = frame.melt(id_vars="song_id", var_name="time_col", value_name=value_name)
    long_df["song_id"] = pd.to_numeric(long_df["song_id"], errors="coerce")
    long_df[value_name] = pd.to_numeric(long_df[value_name], errors="coerce")
    long_df["time_sec"] = long_df["time_col"].map(_parse_sample_ms_column)
    long_df = long_df.dropna(subset=["song_id", "time_sec", value_name])
    long_df["song_id"] = long_df["song_id"].astype(int)
    return long_df[["song_id", "time_sec", value_name]]


def _load_averaged_dynamic_csvs(
    dynamic_dir: Path,
    valence_name: str,
    arousal_name: str,
    fallback_hz: float | None,
) -> pd.DataFrame:
    """Load averaged dynamic valence/arousal (wide CSV files)."""
    valence_path = dynamic_dir / valence_name
    arousal_path = dynamic_dir / arousal_name
    if not valence_path.exists() or not arousal_path.exists():
        raise FileNotFoundError(
            f"Averaged dynamic files not found in {dynamic_dir}. "
            f"Expected {valence_name} and {arousal_name}."
        )

    valence_wide = pd.read_csv(valence_path)
    arousal_wide = pd.read_csv(arousal_path)

    if _is_dynamic_songs_as_rows_format(valence_wide):
        logger.info("Loading dynamic annotations from song-row / sample-column format.")
        valence_long = _normalize_dynamic_songs_as_rows(valence_wide, "valence")
        arousal_long = _normalize_dynamic_songs_as_rows(arousal_wide, "arousal")
        merge_keys = ["song_id", "time_sec"]
    else:
        time_grid = infer_dynamic_time_grid(valence_wide, fallback_hz=fallback_hz)
        logger.info(
            "Inferred dynamic annotation grid: mode=%s, step_sec=%.4f, hz=%s",
            time_grid["mode"],
            time_grid["step_sec"],
            time_grid["inferred_hz"],
        )
        valence_long = _normalize_dynamic_matrix(valence_wide, "valence", time_grid=time_grid)
        arousal_long = _normalize_dynamic_matrix(arousal_wide, "arousal", time_grid=time_grid)
        merge_keys = ["song_id", "time_sec"]

    merged = valence_long.merge(arousal_long, on=merge_keys, how="outer")
    return merged[["song_id", "time_sec", "valence", "arousal"]]


def _load_per_rater_dynamic_annotations(dynamic_dir: Path) -> pd.DataFrame:
    """Load per-rater dynamic CSVs (one file per song under valence/ and arousal/)."""
    valence_dir = dynamic_dir / "valence"
    arousal_dir = dynamic_dir / "arousal"
    if not valence_dir.is_dir() or not arousal_dir.is_dir():
        raise FileNotFoundError(
            f"Per-rater dynamic folders not found under {dynamic_dir}. "
            "Expected valence/ and arousal/ subdirectories."
        )

    rows: list[dict[str, Any]] = []
    for valence_path in tqdm(list(valence_dir.glob("*.csv")), desc="Loading per-rater dynamic", leave=False):
        try:
            song_id = int(valence_path.stem)
        except ValueError:
            continue
        arousal_path = arousal_dir / valence_path.name
        if not arousal_path.exists():
            logger.warning("Missing arousal file for song_id=%s", song_id)
            continue

        valence_df = pd.read_csv(valence_path)
        arousal_df = pd.read_csv(arousal_path)

        for row_idx in range(len(valence_df)):
            for col in valence_df.columns:
                time_sec = _parse_sample_ms_column(col)
                if time_sec is None or col not in arousal_df.columns:
                    continue
                valence = pd.to_numeric(valence_df.iloc[row_idx][col], errors="coerce")
                arousal = pd.to_numeric(arousal_df.iloc[row_idx][col], errors="coerce")
                if pd.isna(valence) or pd.isna(arousal):
                    continue
                rows.append(
                    {
                        "song_id": song_id,
                        "time_sec": time_sec,
                        "valence": float(valence),
                        "arousal": float(arousal),
                    }
                )

    if not rows:
        raise FileNotFoundError(f"No per-rater dynamic annotations loaded from {dynamic_dir}")

    return pd.DataFrame(rows)


def _finalize_dynamic_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure int song_id, average across raters, and return standard columns."""
    frame = df.copy()
    frame["song_id"] = pd.to_numeric(frame["song_id"], errors="coerce")
    frame["time_sec"] = pd.to_numeric(frame["time_sec"], errors="coerce")
    frame["valence"] = pd.to_numeric(frame["valence"], errors="coerce")
    frame["arousal"] = pd.to_numeric(frame["arousal"], errors="coerce")
    frame = frame.dropna(subset=["song_id", "time_sec", "valence", "arousal"])
    frame["song_id"] = frame["song_id"].astype(int)

    frame = frame.groupby(["song_id", "time_sec"], as_index=False).agg(
        valence=("valence", "mean"),
        arousal=("arousal", "mean"),
    )
    return frame.sort_values(["song_id", "time_sec"]).reset_index(drop=True)


def get_deam_paths(configs: dict[str, dict[str, Any]] | None = None) -> dict[str, Path]:
    """Return resolved DEAM-related paths from config."""
    if configs is None:
        configs = load_configs()
    root = get_project_root()
    paths_cfg = configs["paths"]
    deam_cfg = paths_cfg["deam"]

    annotations_dir = _resolve_annotations_root(root, deam_cfg)
    averaged_dir = _resolve_deam_subdir(
        annotations_dir,
        deam_cfg["annotations"]["averaged_per_song"],
        ("annotations_averaged_per_song", "annotations averaged per song"),
    )
    song_level_dir = _resolve_deam_subdir(
        averaged_dir,
        deam_cfg["annotations"]["song_level"],
        ("song_level", "static"),
    )
    dynamic_dir = _resolve_dynamic_dir(annotations_dir, averaged_dir, deam_cfg)
    audio_dir = _deam_subpath(root, deam_cfg, deam_cfg["audio"]["dir"])

    return {
        "project_root": root,
        "annotations_dir": annotations_dir,
        "averaged_dir": averaged_dir,
        "song_level_dir": song_level_dir,
        "dynamic_dir": dynamic_dir,
        "audio_dir": audio_dir,
        "static_files": deam_cfg["annotations"]["static_files"],
        "dynamic_valence": deam_cfg["annotations"]["dynamic_valence"],
        "dynamic_arousal": deam_cfg["annotations"]["dynamic_arousal"],
        "audio_extensions": tuple(deam_cfg["audio"]["extensions"]),
    }


def load_static_annotations(
    configs: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Load and merge static (song-level) valence/arousal annotations."""
    deam_paths = get_deam_paths(configs)
    song_level_dir = deam_paths["song_level_dir"]
    frames: list[pd.DataFrame] = []

    for filename in deam_paths["static_files"]:
        filepath = song_level_dir / filename
        if not filepath.exists():
            logger.warning("Static annotation file missing: %s", filepath)
            continue
        raw = pd.read_csv(filepath)
        frames.append(_normalize_static_frame(raw))
        logger.info("Loaded static annotations from %s (%d rows)", filename, len(raw))

    if not frames:
        raise FileNotFoundError(
            f"No static annotation files found in {song_level_dir}. "
            "Download DEAM annotations and update configs/paths.yaml."
        )

    static_df = pd.concat(frames, ignore_index=True)
    static_df = static_df.drop_duplicates(subset=["song_id"], keep="first")
    static_df = static_df.sort_values("song_id").reset_index(drop=True)
    return static_df


def _normalize_dynamic_matrix(
    df: pd.DataFrame,
    value_name: str,
    time_grid: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Convert wide dynamic matrix (time x songs) to long format."""
    frame = df.copy()
    time_col = frame.columns[0]
    if str(time_col).lower() in {"sample", "time", "second", "seconds", "timestamp", "time_sec"}:
        frame = frame.rename(columns={time_col: "time_index"})
    else:
        frame = frame.rename(columns={time_col: "time_index"})

    long_df = frame.melt(id_vars="time_index", var_name="song_id", value_name=value_name)
    long_df["song_id"] = pd.to_numeric(long_df["song_id"], errors="coerce").astype("Int64")
    long_df["time_index"] = pd.to_numeric(long_df["time_index"], errors="coerce")
    long_df[value_name] = pd.to_numeric(long_df[value_name], errors="coerce")
    long_df = long_df.dropna(subset=["song_id", "time_index"])
    long_df["song_id"] = long_df["song_id"].astype(int)

    if time_grid is not None:
        long_df["time_sec"] = _time_index_to_seconds(long_df["time_index"], time_grid)

    return long_df.sort_values(["song_id", "time_index"]).reset_index(drop=True)


def _time_index_to_seconds(time_index: pd.Series, time_grid: dict[str, Any]) -> pd.Series:
    """Convert raw annotation indices to seconds using inferred grid metadata."""
    values = time_index.astype(float)
    mode = time_grid["mode"]
    if mode in {
        "absolute_seconds_column",
        "fractional_values_as_seconds",
        "integer_values_as_seconds",
    }:
        return values
    if mode == "integer_sample_index":
        return values / float(time_grid["inferred_hz"])
    return values * float(time_grid["step_sec"])


def infer_dynamic_time_grid(
    wide_df: pd.DataFrame,
    fallback_hz: float | None = None,
) -> dict[str, Any]:
    """
    Infer the annotation time grid from a wide-format dynamic annotation file.

    Returns a dictionary with:
      - step_sec: median spacing between consecutive annotation times (seconds)
      - inferred_hz: 1 / step_sec when step_sec > 0
      - mode: how the grid was interpreted
      - index_to_sec: mapping from raw time_index values to seconds
    """
    time_col = wide_df.columns[0]
    time_values = pd.to_numeric(wide_df[time_col], errors="coerce").dropna().to_numpy(dtype=float)
    unique_sorted = np.sort(np.unique(time_values))

    if len(unique_sorted) < 2:
        step_sec = 1.0
    else:
        step_sec = float(np.median(np.diff(unique_sorted)))

    time_col_lower = str(time_col).strip().lower()
    absolute_seconds_names = {"second", "seconds", "time", "timestamp", "time_sec"}

    if time_col_lower in absolute_seconds_names:
        index_to_sec = {float(v): float(v) for v in unique_sorted}
        return {
            "step_sec": step_sec,
            "inferred_hz": 1.0 / step_sec if step_sec > 0 else None,
            "mode": "absolute_seconds_column",
            "time_column": time_col,
            "index_to_sec": index_to_sec,
        }

    # Values like 0.0, 0.5, 1.0 — already seconds on a sub-second grid
    if step_sec < 1.0 and not np.allclose(unique_sorted, np.round(unique_sorted)):
        index_to_sec = {float(v): float(v) for v in unique_sorted}
        return {
            "step_sec": step_sec,
            "inferred_hz": 1.0 / step_sec if step_sec > 0 else None,
            "mode": "fractional_values_as_seconds",
            "time_column": time_col,
            "index_to_sec": index_to_sec,
        }

    # Integer steps of ~1.0 — treat index values as seconds (e.g. 0, 1, 2 at 1 Hz)
    if step_sec >= 0.9 and np.allclose(unique_sorted, np.round(unique_sorted)):
        index_to_sec = {float(v): float(v) for v in unique_sorted}
        return {
            "step_sec": step_sec,
            "inferred_hz": 1.0 / step_sec if step_sec > 0 else None,
            "mode": "integer_values_as_seconds",
            "time_column": time_col,
            "index_to_sec": index_to_sec,
        }

    # Integer sample indices 0, 1, 2, ... — infer Hz from plausible track durations
    max_index = float(unique_sorted.max())
    inferred_hz = _infer_hz_from_sample_count(max_index, fallback_hz)
    step_sec = 1.0 / inferred_hz
    index_to_sec = {float(v): float(v) / inferred_hz for v in unique_sorted}
    return {
        "step_sec": step_sec,
        "inferred_hz": inferred_hz,
        "mode": "integer_sample_index",
        "time_column": time_col,
        "index_to_sec": index_to_sec,
    }


def _infer_hz_from_sample_count(max_index: float, fallback_hz: float | None) -> float:
    """Infer annotation rate from the maximum sample index and typical song lengths."""
    if fallback_hz is not None and fallback_hz > 0:
        return float(fallback_hz)

    n_samples = max_index + 1.0
    candidate_durations = [30.0, 45.0, 60.0, 90.0, 120.0, 180.0, 240.0]
    best_hz = None
    best_score = float("inf")

    for duration in candidate_durations:
        hz = n_samples / duration
        if hz < 0.5 or hz > 10.0:
            continue
        rounded_hz = round(hz)
        score = abs(hz - rounded_hz)
        if score < best_score:
            best_score = score
            best_hz = rounded_hz if score < 0.15 else hz

    if best_hz is not None:
        return float(best_hz)

    logger.warning(
        "Could not infer annotation Hz from sample count (max_index=%.0f). Using 1 Hz fallback.",
        max_index,
    )
    return 1.0


def load_dynamic_annotations(
    configs: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """
    Load dynamic valence and arousal annotations.

    Returns a long-format DataFrame with columns:
    song_id, time_sec, valence, arousal
    """
    if configs is None:
        configs = load_configs()

    deam_paths = get_deam_paths(configs)
    dynamic_dir = deam_paths["dynamic_dir"]
    dynamic_cfg = configs["features"]["dynamic"]
    fallback_hz = dynamic_cfg.get("fallback_hz")
    fallback_hz = float(fallback_hz) if fallback_hz is not None else None

    valence_file = deam_paths["dynamic_valence"]
    arousal_file = deam_paths["dynamic_arousal"]

    logger.info("Dynamic annotation folder: %s", dynamic_dir)

    if (dynamic_dir / valence_file).exists() and (dynamic_dir / arousal_file).exists():
        logger.info("Loading averaged dynamic CSVs: %s, %s", valence_file, arousal_file)
        dynamic_df = _load_averaged_dynamic_csvs(
            dynamic_dir, valence_file, arousal_file, fallback_hz
        )
    else:
        logger.info("Averaged dynamic CSVs not found; loading per-rater files from %s", dynamic_dir)
        dynamic_df = _load_per_rater_dynamic_annotations(dynamic_dir)

    dynamic_df = _finalize_dynamic_df(dynamic_df)
    n_tracks = int(dynamic_df["song_id"].nunique())
    logger.info(
        "Loaded %d dynamic rows for %d unique tracks.",
        len(dynamic_df),
        n_tracks,
    )
    return dynamic_df


def discover_audio_files(
    audio_dir: Path,
    extensions: tuple[str, ...] = (".mp3", ".wav"),
) -> dict[int, Path]:
    """Map song_id -> audio file path for all matching files in ``audio_dir``."""
    if not audio_dir.exists():
        logger.warning("Audio directory does not exist: %s", audio_dir)
        return {}

    mapping: dict[int, Path] = {}
    normalized_ext = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}

    for path in tqdm(list(audio_dir.iterdir()), desc="Scanning audio files", leave=False):
        if not path.is_file():
            continue
        if path.suffix.lower() not in normalized_ext:
            continue
        try:
            song_id = int(path.stem)
        except ValueError:
            logger.debug("Skipping non-numeric audio filename: %s", path.name)
            continue
        mapping[song_id] = path

    return mapping


def build_metadata_table(
    configs: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """
    Build a unified metadata table linking songs, annotations, and audio files.

    Columns include valence/arousal (static), dynamic annotation availability,
    and audio path flags.
    """
    if configs is None:
        configs = load_configs()

    deam_paths = get_deam_paths(configs)
    static_df = load_static_annotations(configs)

    try:
        dynamic_df = load_dynamic_annotations(configs)
        dynamic_df["song_id"] = dynamic_df["song_id"].astype(int)
        dynamic_song_ids = set(dynamic_df["song_id"].unique())
        dynamic_counts = dynamic_df.groupby("song_id").size().rename("n_dynamic_samples")
    except Exception as exc:
        logger.warning("Dynamic annotations unavailable: %s", exc)
        dynamic_df = pd.DataFrame(columns=["song_id", "time_sec", "valence", "arousal"])
        dynamic_song_ids = set()
        dynamic_counts = pd.Series(dtype=int)

    audio_map = discover_audio_files(deam_paths["audio_dir"], deam_paths["audio_extensions"])

    metadata = static_df.copy()
    metadata["has_static_annotation"] = metadata["valence"].notna() & metadata["arousal"].notna()
    metadata["has_dynamic_annotation"] = metadata["song_id"].isin(dynamic_song_ids)
    metadata["has_audio"] = metadata["song_id"].isin(audio_map)
    metadata["audio_path"] = metadata["song_id"].map(lambda sid: str(audio_map[sid]) if sid in audio_map else pd.NA)
    metadata = metadata.merge(dynamic_counts, on="song_id", how="left")
    metadata["n_dynamic_samples"] = metadata["n_dynamic_samples"].fillna(0).astype(int)
    metadata["is_complete"] = (
        metadata["has_static_annotation"]
        & metadata["has_dynamic_annotation"]
        & metadata["has_audio"]
    )
    return metadata.sort_values("song_id").reset_index(drop=True)


def audit_deam_dataset(
    configs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run a dataset audit and return summary statistics."""
    metadata = build_metadata_table(configs)

    valence = metadata["valence"].dropna()
    arousal = metadata["arousal"].dropna()

    summary: dict[str, Any] = {
        "n_tracks_static": int(len(metadata)),
        "n_tracks_with_audio": int(metadata["has_audio"].sum()),
        "n_tracks_with_dynamic": int(metadata["has_dynamic_annotation"].sum()),
        "n_tracks_complete": int(metadata["is_complete"].sum()),
        "n_missing_static_valence": int(metadata["valence"].isna().sum()),
        "n_missing_static_arousal": int(metadata["arousal"].isna().sum()),
        "n_missing_audio": int((~metadata["has_audio"]).sum()),
        "n_missing_dynamic": int((~metadata["has_dynamic_annotation"]).sum()),
        "valence_mean": float(valence.mean()) if len(valence) else None,
        "valence_std": float(valence.std()) if len(valence) else None,
        "valence_min": float(valence.min()) if len(valence) else None,
        "valence_max": float(valence.max()) if len(valence) else None,
        "arousal_mean": float(arousal.mean()) if len(arousal) else None,
        "arousal_std": float(arousal.std()) if len(arousal) else None,
        "arousal_min": float(arousal.min()) if len(arousal) else None,
        "arousal_max": float(arousal.max()) if len(arousal) else None,
        "metadata": metadata,
    }
    return summary


def save_metadata_table(
    metadata: pd.DataFrame,
    configs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Path]:
    """Persist metadata to processed CSV and Parquet paths from config."""
    if configs is None:
        configs = load_configs()
    root = get_project_root()
    paths_cfg = configs["paths"]["processed"]

    parquet_path = resolve_path(root, paths_cfg["metadata"])
    csv_path = resolve_path(root, paths_cfg["metadata_csv"])
    ensure_dir(parquet_path.parent)

    metadata.to_csv(csv_path, index=False)
    saved: dict[str, Path] = {"csv": csv_path}

    try:
        metadata.to_parquet(parquet_path, index=False)
        saved["parquet"] = parquet_path
        logger.info("Saved metadata to %s and %s", parquet_path, csv_path)
    except Exception as exc:
        logger.warning("Could not save metadata parquet to %s: %s", parquet_path, exc)
        logger.info("Saved metadata to %s", csv_path)

    return saved
