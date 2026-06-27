"""Training and evaluation for dynamic window-level emotion models."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

from src.data.make_dynamic_dataset import load_dynamic_windows, standardize_dynamic_window_columns
from src.data.splits import verify_window_splits
from src.features.dynamic_window_features import load_dynamic_window_features
from src.models.dynamic_ml import build_dynamic_classification_models, build_dynamic_regression_models
from src.training.train_static import (
    compute_classification_metrics,
    plot_confusion_matrix,
)
from src.utils.config import ensure_dir, get_project_root, load_configs, resolve_path

logger = logging.getLogger(__name__)

NON_FEATURE_COLUMNS = frozenset(
    {
        "song_id",
        "track_id",
        "window_index",
        "split",
        "valence",
        "arousal",
        "dynamic_emotion_quadrant",
        "emotion_quadrant",
        "emotion_class",
        "dynamic_emotion_class",
        "valence_threshold",
        "arousal_threshold",
        "annotation_time_sec",
        "window_start_sec",
        "window_end_sec",
        "window_start",
        "window_end",
        "audio_path",
        "path",
        "file_path",
        "filename",
    }
)

LABEL_COLUMNS = (
    "song_id",
    "window_index",
    "split",
    "valence",
    "arousal",
    "dynamic_emotion_quadrant",
)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in NON_FEATURE_COLUMNS]
    if not feature_cols:
        raise ValueError(
            "No numeric feature columns found for training. "
            f"Numeric columns in data: {numeric_cols}."
        )
    return feature_cols


def prepare_dynamic_training_frame(
    features_df: pd.DataFrame,
    windows_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge window features with dynamic labels by song_id and window_index."""
    windows_df = standardize_dynamic_window_columns(windows_df)

    label_cols = [col for col in LABEL_COLUMNS if col in windows_df.columns]
    missing_labels = [col for col in LABEL_COLUMNS if col not in label_cols]
    if missing_labels:
        raise ValueError(
            f"Dynamic windows missing required label columns: {missing_labels}. "
            f"Available: {list(windows_df.columns)}"
        )

    features_df = features_df.copy()
    features_df["song_id"] = pd.to_numeric(features_df["song_id"], errors="coerce").astype(int)
    features_df["window_index"] = pd.to_numeric(features_df["window_index"], errors="coerce").astype(int)

    merged = features_df.merge(
        windows_df[label_cols],
        on=["song_id", "window_index"],
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlapping rows between dynamic window features and labels.")
    if len(merged) < len(features_df):
        logger.warning(
            "Dropped %d feature rows without matching labels.",
            len(features_df) - len(merged),
        )
    return merged


def validate_dynamic_training_frame(df: pd.DataFrame) -> None:
    """Validate splits and ensure no track-level leakage."""
    if df["split"].isna().any():
        missing = df.loc[df["split"].isna(), "song_id"].nunique()
        raise ValueError(f"Found windows with missing split values ({missing} tracks affected).")

    invalid_splits = set(df["split"].unique()) - {"train", "val", "test"}
    if invalid_splits:
        raise ValueError(f"Unexpected split values: {invalid_splits}")

    verify_window_splits(df, track_col="song_id")
    logger.info("Split validation passed (track-level, no leakage).")


def load_dynamic_training_data(
    configs: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Load and merge dynamic window features with labels."""
    if configs is None:
        configs = load_configs()

    features_df = load_dynamic_window_features(configs)
    windows_df = load_dynamic_windows(configs, attach_splits=True, save=False)
    data = prepare_dynamic_training_frame(features_df, windows_df)
    validate_dynamic_training_frame(data)
    return data


def compute_regression_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _split_frames(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = data[data["split"] == "train"]
    val_df = data[data["split"] == "val"]
    test_df = data[data["split"] == "test"]
    return train_df, val_df, test_df


def train_dynamic_window_classification(
    data: pd.DataFrame,
    configs: dict[str, dict[str, Any]],
    feature_cols: list[str],
    random_state: int,
    metrics_dir: Path,
    figures_dir: Path,
) -> pd.DataFrame:
    """Train window-level quadrant classifiers and save metrics."""
    models = build_dynamic_classification_models(configs, random_state=random_state)
    train_df, val_df, test_df = _split_frames(data)

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(train_df["dynamic_emotion_quadrant"].to_numpy())
    y_val_enc = label_encoder.transform(val_df["dynamic_emotion_quadrant"].to_numpy())
    y_test_enc = label_encoder.transform(test_df["dynamic_emotion_quadrant"].to_numpy())
    class_names = label_encoder.classes_.tolist()

    summary_rows: list[dict[str, Any]] = []

    for model_name, pipeline in models.items():
        X_train = train_df[feature_cols].to_numpy()
        X_val = val_df[feature_cols].to_numpy()
        X_test = test_df[feature_cols].to_numpy()

        pipeline.fit(X_train, y_train_enc)

        split_metrics: dict[str, Any] = {
            "model_name": model_name,
            "task": "classification",
            "target": "dynamic_emotion_quadrant",
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "n_features": len(feature_cols),
            "label_classes": class_names,
        }

        for split_name, X_split, y_enc in (
            ("val", X_val, y_val_enc),
            ("test", X_test, y_test_enc),
        ):
            y_pred_enc = pipeline.predict(X_split)
            y_true = label_encoder.inverse_transform(y_enc)
            y_pred = label_encoder.inverse_transform(y_pred_enc)
            metrics = compute_classification_metrics(y_true, y_pred, labels=class_names)

            if split_name == "test":
                cm = np.array(metrics["confusion_matrix"])
                plot_confusion_matrix(
                    cm,
                    labels=metrics["labels"],
                    title=f"{model_name} — dynamic window classification (test)",
                    output_path=figures_dir / f"dynamic_window_{model_name}_confusion_matrix.png",
                )

            split_metrics[split_name] = {
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "classification_report": metrics["classification_report"],
                "confusion_matrix": metrics["confusion_matrix"],
            }

            summary_rows.append(
                {
                    "model_name": model_name,
                    "task": "classification",
                    "target": "dynamic_emotion_quadrant",
                    "eval_split": split_name,
                    "train_size": len(train_df),
                    "val_size": len(val_df),
                    "test_size": len(test_df),
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                    "mae": None,
                    "rmse": None,
                    "r2": None,
                }
            )
            logger.info(
                "%s [classification %s] — accuracy=%.4f, macro_f1=%.4f",
                model_name,
                split_name,
                metrics["accuracy"],
                metrics["macro_f1"],
            )

        metrics_path = metrics_dir / f"dynamic_window_{model_name}_classification_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(split_metrics, handle, indent=2)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = metrics_dir / "dynamic_window_classification_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    return summary_df


def train_dynamic_window_regression(
    data: pd.DataFrame,
    configs: dict[str, dict[str, Any]],
    feature_cols: list[str],
    random_state: int,
    metrics_dir: Path,
) -> pd.DataFrame:
    """Train window-level valence/arousal regressors and save metrics."""
    models = build_dynamic_regression_models(configs, random_state=random_state)
    train_df, val_df, test_df = _split_frames(data)
    targets = ("valence", "arousal")

    summary_rows: list[dict[str, Any]] = []

    for model_name, pipeline in models.items():
        model_metrics: dict[str, Any] = {
            "model_name": model_name,
            "task": "regression",
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "n_features": len(feature_cols),
            "targets": {},
        }

        for target in targets:
            y_train = train_df[target].to_numpy()
            y_val = val_df[target].to_numpy()
            y_test = test_df[target].to_numpy()

            X_train = train_df[feature_cols].to_numpy()
            X_val = val_df[feature_cols].to_numpy()
            X_test = test_df[feature_cols].to_numpy()

            pipeline.fit(X_train, y_train)
            model_metrics["targets"][target] = {}

            for split_name, X_split, y_true in (
                ("val", X_val, y_val),
                ("test", X_test, y_test),
            ):
                y_pred = pipeline.predict(X_split)
                metrics = compute_regression_metrics(y_true, y_pred)
                model_metrics["targets"][target][split_name] = metrics

                summary_rows.append(
                    {
                        "model_name": model_name,
                        "task": "regression",
                        "target": target,
                        "eval_split": split_name,
                        "train_size": len(train_df),
                        "val_size": len(val_df),
                        "test_size": len(test_df),
                        "accuracy": None,
                        "macro_f1": None,
                        "weighted_f1": None,
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "r2": metrics["r2"],
                    }
                )
                logger.info(
                    "%s [regression %s %s] — MAE=%.4f, RMSE=%.4f, R2=%.4f",
                    model_name,
                    target,
                    split_name,
                    metrics["mae"],
                    metrics["rmse"],
                    metrics["r2"],
                )

        metrics_path = metrics_dir / f"dynamic_window_{model_name}_regression_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(model_metrics, handle, indent=2)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = metrics_dir / "dynamic_window_regression_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    return summary_df


def train_and_evaluate_dynamic_window_models(
    configs: dict[str, dict[str, Any]] | None = None,
    data: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Train dynamic window-level classification and regression models.

    Uses existing track-level splits from the dynamic windows table.
    """
    if configs is None:
        configs = load_configs()

    root = get_project_root()
    random_state = int(configs["training"]["general"]["random_state"])

    if data is None:
        data = load_dynamic_training_data(configs)

    feature_cols = _feature_columns(data)
    logger.info("Training dynamic window models with %d features.", len(feature_cols))

    metrics_dir = resolve_path(root, configs["paths"]["results"]["metrics"])
    figures_dir = resolve_path(root, configs["paths"]["reports"]["figures"])
    ensure_dir(metrics_dir)
    ensure_dir(figures_dir)

    classification_summary = train_dynamic_window_classification(
        data,
        configs,
        feature_cols,
        random_state,
        metrics_dir,
        figures_dir,
    )
    regression_summary = train_dynamic_window_regression(
        data,
        configs,
        feature_cols,
        random_state,
        metrics_dir,
    )

    return {
        "classification": classification_summary,
        "regression": regression_summary,
        "data": data,
    }
