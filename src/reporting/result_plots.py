"""Final result comparison plots for thesis reporting."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.reporting.collect_results import collect_experiment_summaries
from src.utils.config import ensure_dir, get_project_root, load_configs, resolve_path

logger = logging.getLogger(__name__)


def _classification_test(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["eval_split"].eq("test") & df["accuracy"].notna()].copy()


def plot_final_result_figures(
    configs: dict[str, dict[str, Any]] | None = None,
    combined: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Create grouped bar plots for classification and regression metrics."""
    if configs is None:
        configs = load_configs()

    root = get_project_root()
    figures_dir = resolve_path(root, configs["paths"]["reports"]["final_results"])
    ensure_dir(figures_dir)

    if combined is None:
        combined, _ = collect_experiment_summaries(configs)

    saved: dict[str, Path] = {}
    clf_test = _classification_test(combined)

    if not clf_test.empty:
        for metric, filename in (
            ("macro_f1", "classification_macro_f1_comparison.png"),
            ("accuracy", "classification_accuracy_comparison.png"),
        ):
            plot_df = clf_test.copy()
            plot_df["label"] = plot_df["experiment_group"] + " / " + plot_df["model_name"].astype(str)
            plot_df = plot_df.sort_values(metric, ascending=False)

            fig, ax = plt.subplots(figsize=(12, 5))
            sns.barplot(data=plot_df, x="label", y=metric, hue="experiment_group", ax=ax, dodge=False)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right")
            ax.set_title(f"Test {metric.replace('_', ' ').title()} — classification models")
            ax.set_xlabel("Model")
            ax.set_ylabel(metric.replace("_", " ").title())
            fig.tight_layout()
            path = figures_dir / filename
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved[metric] = path

        grouped = (
            clf_test.groupby("experiment_group", as_index=False)[["macro_f1", "accuracy"]]
            .max()
            .sort_values("macro_f1", ascending=False)
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        grouped_melt = grouped.melt(id_vars="experiment_group", var_name="metric", value_name="score")
        sns.barplot(data=grouped_melt, x="experiment_group", y="score", hue="metric", ax=ax)
        ax.set_title("Best test scores per experiment group")
        fig.tight_layout()
        path = figures_dir / "classification_by_experiment_group.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved["by_experiment_group"] = path

    reg_test = combined.loc[combined["eval_split"].eq("test") & combined["mae"].notna()].copy()
    if not reg_test.empty:
        for metric, filename in (
            ("mae", "regression_mae_comparison.png"),
            ("rmse", "regression_rmse_comparison.png"),
        ):
            plot_df = reg_test.copy()
            plot_df["label"] = plot_df["model_name"].astype(str) + " (" + plot_df["target_type"].astype(str) + ")"
            fig, ax = plt.subplots(figsize=(12, 5))
            sns.barplot(data=plot_df, x="label", y=metric, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right")
            ax.set_title(f"Test {metric.upper()} — dynamic window regression")
            fig.tight_layout()
            path = figures_dir / filename
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved[f"regression_{metric}"] = path

    if not saved:
        logger.warning("No plots were generated — no metric data available.")

    return saved
