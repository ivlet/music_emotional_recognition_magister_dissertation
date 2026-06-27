"""Plotting utilities for dynamic emotion trajectories over time."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data.load_deam import load_dynamic_annotations
from src.data.make_static_dataset import QUADRANT_LABELS, assign_emotion_quadrant
from src.utils.config import ensure_dir, get_project_root, load_configs, resolve_path

logger = logging.getLogger(__name__)

QUADRANT_COLORS = {
    "Q1": "#E45756",
    "Q2": "#4C78A8",
    "Q3": "#F58518",
    "Q4": "#72B7B2",
}


def _resolve_output_dir(configs: dict[str, dict[str, Any]] | None = None) -> Path:
    if configs is None:
        configs = load_configs()
    root = get_project_root()
    return resolve_path(root, configs["paths"]["reports"]["dynamic_trajectories"])


def _track_dynamic_data(
    song_id: int,
    dynamic_df: pd.DataFrame | None,
    configs: dict[str, dict[str, Any]] | None,
) -> pd.DataFrame:
    if dynamic_df is None:
        dynamic_df = load_dynamic_annotations(configs)
    if configs is None:
        configs = load_configs()

    dynamic_cfg = configs["features"]["dynamic"]
    emotion_cfg = configs["features"]["emotion"]
    skip_initial_sec = float(dynamic_cfg["skip_initial_sec"])

    track = dynamic_df.loc[dynamic_df["song_id"] == song_id].copy()
    if track.empty:
        raise ValueError(f"No dynamic annotations found for song_id={song_id}")

    track = track.sort_values("time_sec")
    track = track[track["time_sec"] >= skip_initial_sec].dropna(subset=["valence", "arousal"])
    if track.empty:
        raise ValueError(f"No usable dynamic annotations for song_id={song_id} after skip_initial_sec")

    labels = assign_emotion_quadrant(
        track["valence"],
        track["arousal"],
        valence_threshold=emotion_cfg["valence_threshold"],
        arousal_threshold=emotion_cfg["arousal_threshold"],
    )
    track["dynamic_emotion_quadrant"] = labels["emotion_quadrant"].values
    track["dynamic_emotion_class"] = labels["emotion_class"].values
    return track


def _save_figure(fig: plt.Figure, output_path: Path | str | None) -> Path | None:
    if output_path is None:
        return None
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return output_path


def plot_valence_arousal_over_time(
    song_id: int,
    dynamic_df: pd.DataFrame | None = None,
    configs: dict[str, dict[str, Any]] | None = None,
    output_path: Path | str | None = None,
) -> plt.Figure:
    """Plot valence and arousal over time for a single track."""
    if configs is None:
        configs = load_configs()
    track = _track_dynamic_data(song_id, dynamic_df, configs)

    if output_path is None:
        output_path = _resolve_output_dir(configs) / f"song_{song_id}_valence_arousal.png"

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(track["time_sec"], track["valence"], label="Valence", color="#4C78A8", linewidth=1.5)
    ax.plot(track["time_sec"], track["arousal"], label="Arousal", color="#E45756", linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Rating")
    ax.set_title(f"Valence and arousal over time — song {song_id}")
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, output_path)
    return fig


def plot_emotion_quadrant_over_time(
    song_id: int,
    dynamic_df: pd.DataFrame | None = None,
    configs: dict[str, dict[str, Any]] | None = None,
    output_path: Path | str | None = None,
) -> plt.Figure:
    """Plot emotion quadrant assignments over time for a single track."""
    if configs is None:
        configs = load_configs()
    track = _track_dynamic_data(song_id, dynamic_df, configs)

    if output_path is None:
        output_path = _resolve_output_dir(configs) / f"song_{song_id}_quadrant_over_time.png"

    quadrant_order = list(configs["features"]["emotion"].get("class_order", QUADRANT_COLORS.keys()))
    track["dynamic_emotion_quadrant"] = pd.Categorical(
        track["dynamic_emotion_quadrant"],
        categories=quadrant_order,
        ordered=True,
    )

    fig, ax = plt.subplots(figsize=(9, 3))
    colors = [QUADRANT_COLORS.get(q, "gray") for q in track["dynamic_emotion_quadrant"]]
    ax.scatter(track["time_sec"], track["dynamic_emotion_quadrant"], c=colors, s=18, alpha=0.85)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Emotion quadrant")
    ax.set_title(f"Emotion quadrant over time — song {song_id}")
    fig.tight_layout()
    _save_figure(fig, output_path)
    return fig


def plot_emotion_trajectory_2d(
    song_id: int,
    dynamic_df: pd.DataFrame | None = None,
    configs: dict[str, dict[str, Any]] | None = None,
    output_path: Path | str | None = None,
) -> plt.Figure:
    """Plot a 2-D valence–arousal trajectory with time progression for one track."""
    if configs is None:
        configs = load_configs()
    track = _track_dynamic_data(song_id, dynamic_df, configs)

    if output_path is None:
        output_path = _resolve_output_dir(configs) / f"song_{song_id}_trajectory_2d.png"

    emotion_cfg = configs["features"]["emotion"]
    labels = assign_emotion_quadrant(
        track["valence"],
        track["arousal"],
        valence_threshold=emotion_cfg["valence_threshold"],
        arousal_threshold=emotion_cfg["arousal_threshold"],
    )
    v_cut = float(labels["valence_threshold"].iloc[0])
    a_cut = float(labels["arousal_threshold"].iloc[0])

    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter(
        track["valence"],
        track["arousal"],
        c=track["time_sec"],
        cmap="viridis",
        s=25,
        alpha=0.8,
    )
    ax.plot(track["valence"], track["arousal"], color="gray", alpha=0.35, linewidth=1)
    ax.axvline(v_cut, color="gray", linestyle="--", linewidth=1)
    ax.axhline(a_cut, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_title(f"2-D emotion trajectory — song {song_id}")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Time (s)")
    fig.tight_layout()
    _save_figure(fig, output_path)
    return fig


def plot_example_trajectories(
    song_ids: list[int],
    dynamic_df: pd.DataFrame | None = None,
    configs: dict[str, dict[str, Any]] | None = None,
) -> dict[int, dict[str, Path | None]]:
    """Generate all trajectory plots for a list of song IDs."""
    if configs is None:
        configs = load_configs()
    output_dir = _resolve_output_dir(configs)
    ensure_dir(output_dir)

    saved: dict[int, dict[str, Path | None]] = {}
    for song_id in song_ids:
        va_path = output_dir / f"song_{song_id}_valence_arousal.png"
        q_path = output_dir / f"song_{song_id}_quadrant_over_time.png"
        t_path = output_dir / f"song_{song_id}_trajectory_2d.png"
        try:
            plot_valence_arousal_over_time(song_id, dynamic_df, configs, va_path)
            plot_emotion_quadrant_over_time(song_id, dynamic_df, configs, q_path)
            plot_emotion_trajectory_2d(song_id, dynamic_df, configs, t_path)
            saved[song_id] = {
                "valence_arousal": va_path,
                "quadrant": q_path,
                "trajectory_2d": t_path,
            }
        except ValueError as exc:
            logger.warning("Skipping song_id=%s: %s", song_id, exc)
            saved[song_id] = {"valence_arousal": None, "quadrant": None, "trajectory_2d": None}

    return saved
