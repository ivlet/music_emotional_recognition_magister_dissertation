"""Plotting utilities for dataset audit and experiment reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.config import ensure_dir


def plot_valence_arousal_scatter(
    metadata: pd.DataFrame,
    output_path: Path | str | None = None,
    title: str = "DEAM static valence–arousal distribution",
) -> plt.Figure:
    """Scatter plot of static valence vs arousal with median reference lines."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=metadata,
        x="valence",
        y="arousal",
        alpha=0.5,
        s=20,
        ax=ax,
    )
    if metadata["valence"].notna().any():
        ax.axvline(metadata["valence"].median(), color="gray", linestyle="--", linewidth=1)
    if metadata["arousal"].notna().any():
        ax.axhline(metadata["arousal"].median(), color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Valence")
    ax.set_ylabel("Arousal")
    ax.set_title(title)
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_valence_arousal_distributions(
    metadata: pd.DataFrame,
    output_path: Path | str | None = None,
    title: str = "DEAM static valence and arousal distributions",
) -> plt.Figure:
    """Histograms of static valence and arousal."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(metadata["valence"].dropna(), kde=True, ax=axes[0], color="steelblue")
    axes[0].set_title("Valence")
    sns.histplot(metadata["arousal"].dropna(), kde=True, ax=axes[1], color="indianred")
    axes[1].set_title("Arousal")
    fig.suptitle(title)
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_data_completeness(metadata: pd.DataFrame, output_path: Path | str | None = None) -> plt.Figure:
    """Bar chart of annotation/audio availability counts."""
    counts = pd.Series(
        {
            "static": int(metadata["has_static_annotation"].sum()),
            "dynamic": int(metadata["has_dynamic_annotation"].sum()),
            "audio": int(metadata["has_audio"].sum()),
            "complete": int(metadata["is_complete"].sum()),
        },
        name="count",
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    ax.set_ylabel("Number of tracks")
    ax.set_title("DEAM data completeness")
    ax.set_xticklabels(counts.index, rotation=0)
    fig.tight_layout()
    if output_path is not None:
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
