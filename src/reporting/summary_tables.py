"""Build thesis-ready summary tables from collected experiment results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.reporting.collect_results import STANDARD_COLUMNS, collect_experiment_summaries
from src.utils.config import ensure_dir, get_project_root, load_configs, resolve_path

logger = logging.getLogger(__name__)


def _classification_test(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["eval_split"].eq("test") & df["accuracy"].notna()
    return df.loc[mask].copy()


def build_best_model_per_task(combined: pd.DataFrame) -> pd.DataFrame:
    clf = _classification_test(combined)
    if clf.empty:
        return pd.DataFrame(columns=["task_type", "model_name", "macro_f1", "accuracy"])

    idx = clf.groupby("task_type")["macro_f1"].idxmax()
    best = clf.loc[idx, ["task_type", "model_name", "feature_type", "accuracy", "macro_f1", "weighted_f1"]]
    return best.sort_values("task_type").reset_index(drop=True)


def build_static_vs_dynamic_comparison(combined: pd.DataFrame) -> pd.DataFrame:
    groups = ("static", "dynamic_window")
    clf = _classification_test(combined)
    subset = clf[clf["experiment_group"].isin(groups)]
    if subset.empty:
        return pd.DataFrame()
    return (
        subset.groupby("experiment_group")[["accuracy", "macro_f1", "weighted_f1"]]
        .max()
        .reset_index()
    )


def build_classical_vs_neural_comparison(combined: pd.DataFrame) -> pd.DataFrame:
    clf = _classification_test(combined)
    if clf.empty:
        return pd.DataFrame()

    neural_groups = {"sequence", "spectrogram", "pretrained"}
    clf = clf.copy()
    clf["model_family"] = clf["experiment_group"].apply(
        lambda g: "neural" if g in neural_groups else "classical"
    )
    return (
        clf.groupby("model_family")[["accuracy", "macro_f1", "weighted_f1"]]
        .max()
        .reset_index()
    )


def build_feature_type_comparison(combined: pd.DataFrame) -> pd.DataFrame:
    clf = _classification_test(combined)
    if clf.empty:
        return pd.DataFrame()
    return (
        clf.groupby("feature_type")[["accuracy", "macro_f1", "weighted_f1"]]
        .max()
        .reset_index()
        .sort_values("macro_f1", ascending=False)
    )


def build_regression_summary(combined: pd.DataFrame) -> pd.DataFrame:
    reg = combined[combined["mae"].notna()].copy()
    if reg.empty:
        return pd.DataFrame()
    test = reg[reg["eval_split"].eq("test")]
    return test.sort_values(["target_type", "model_name"]).reset_index(drop=True)


def build_final_summary_tables(
    configs: dict[str, dict[str, Any]] | None = None,
    combined: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Build and save final thesis summary tables.

    Returns dict with combined summary, warnings, and paths to saved tables.
    """
    if configs is None:
        configs = load_configs()

    root = get_project_root()
    tables_dir = resolve_path(root, configs["paths"]["reports"]["tables"])
    ensure_dir(tables_dir)

    if combined is None:
        combined, warnings = collect_experiment_summaries(configs)
    else:
        warnings = []

    final_path = tables_dir / "final_experiment_summary.csv"
    combined.to_csv(final_path, index=False)
    logger.info("Saved final experiment summary to %s", final_path)

    outputs: dict[str, Any] = {
        "combined": combined,
        "warnings": warnings,
        "paths": {"final_experiment_summary": final_path},
    }

    table_builders = {
        "best_model_per_task": build_best_model_per_task,
        "static_vs_dynamic_comparison": build_static_vs_dynamic_comparison,
        "classical_vs_neural_comparison": build_classical_vs_neural_comparison,
        "feature_type_comparison": build_feature_type_comparison,
        "regression_results": build_regression_summary,
    }

    for name, builder in table_builders.items():
        table = builder(combined)
        path = tables_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        outputs["paths"][name] = path
        outputs[name] = table
        logger.info("Saved %s (%d rows)", path, len(table))

    return outputs
