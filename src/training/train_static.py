"""Training and evaluation for static (track-level) emotion classification."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder

from src.data.make_static_dataset import assign_emotion_quadrant
from src.data.splits import (
    assign_splits_to_dataframe,
    create_track_split_table,
    load_track_splits,
    save_track_splits,
)
from src.features.spectral_features import load_static_features
from src.models.classical_ml import build_static_models
from src.utils.config import ensure_dir, get_project_root, load_configs, resolve_path

logger = logging.getLogger(__name__)


NON_FEATURE_COLUMNS = frozenset(
    {
        "song_id",
        "track_id",
        "audio_path",
        "path",
        "file_path",
        "split",
        "valence",
        "arousal",
        "emotion_quadrant",
        "emotion_class",
    }
)


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns, excluding identifiers, labels, and metadata."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in NON_FEATURE_COLUMNS]
    if not feature_cols:
        raise ValueError(
            "No numeric feature columns found for training. "
            f"Numeric columns in data: {numeric_cols}. "
            f"Excluded non-feature columns: {sorted(NON_FEATURE_COLUMNS)}."
        )
    return feature_cols


def prepare_static_training_frame(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    configs: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Merge features with valence/arousal labels (without precomputed quadrants)."""
    if configs is None:
        configs = load_configs()

    label_cols = ["song_id", "valence", "arousal"]
    optional_cols = [c for c in ("audio_path",) if c in labels_df.columns]
    merged = features_df.merge(
        labels_df[label_cols + optional_cols],
        on="song_id",
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlapping tracks between features and labels.")
    return merged


def assign_train_threshold_labels(
    df: pd.DataFrame,
    split_col: str,
    configs: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """
    Assign emotion quadrants using thresholds fit on TRAIN tracks only.

    Prevents label leakage from validation/test sets into threshold estimation.
    """
    emotion_cfg = configs["features"]["emotion"]
    frame = df.copy()
    train_mask = frame[split_col] == "train"

    train_labels = assign_emotion_quadrant(
        frame.loc[train_mask, "valence"],
        frame.loc[train_mask, "arousal"],
        valence_threshold=emotion_cfg["valence_threshold"],
        arousal_threshold=emotion_cfg["arousal_threshold"],
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
    frame["emotion_class"] = all_labels["emotion_class"].values
    frame["valence_threshold"] = v_cut
    frame["arousal_threshold"] = a_cut
    return frame


def compute_classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Compute standard classification metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is not None:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
        display_labels = list(labels)
    else:
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, zero_division=0)
        display_labels = sorted(pd.unique(np.concatenate([y_true, y_pred])).tolist())

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "labels": display_labels,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    title: str,
    output_path: Path | str,
) -> None:
    """Save a confusion matrix heatmap."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def train_and_evaluate_static_models(
    configs: dict[str, dict[str, Any]] | None = None,
    features_df: pd.DataFrame | None = None,
    labels_df: pd.DataFrame | None = None,
    split_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Train classical ML baselines on static features and evaluate on val/test.

    Returns a summary DataFrame with metrics for each model.
    """
    if configs is None:
        configs = load_configs()

    root = get_project_root()
    random_state = int(configs["training"]["general"]["random_state"])

    if features_df is None:
        features_df = load_static_features(configs)

    if labels_df is None:
        labels_path = resolve_path(root, configs["paths"]["processed"]["static_labels"])
        if not labels_path.exists():
            raise FileNotFoundError(f"Static labels not found at {labels_path}")
        labels_df = pd.read_parquet(labels_path)

    data = prepare_static_training_frame(features_df, labels_df, configs)

    if split_df is None:
        try:
            split_df = load_track_splits(configs)
        except FileNotFoundError:
            split_df = create_track_split_table(
                labels_df,
                track_col="song_id",
                label_col="emotion_quadrant" if "emotion_quadrant" in labels_df.columns else None,
                configs=configs,
            )
            save_track_splits(split_df, configs)

    data = assign_splits_to_dataframe(data, split_df, track_col="song_id")
    data = assign_train_threshold_labels(data, split_col="split", configs=configs)

    feature_cols = _feature_columns(data)
    logger.info("Training with %d numeric feature columns.", len(feature_cols))
    models = build_static_models(configs, random_state=random_state)

    metrics_dir = resolve_path(root, configs["paths"]["results"]["metrics"])
    figures_dir = resolve_path(root, configs["paths"]["reports"]["figures"])
    ensure_dir(metrics_dir)
    ensure_dir(figures_dir)

    summary_rows: list[dict[str, Any]] = []

    train_df = data[data["split"] == "train"]
    val_df = data[data["split"] == "val"]
    test_df = data[data["split"] == "test"]

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(train_df["emotion_quadrant"].to_numpy())
    y_val_enc = label_encoder.transform(val_df["emotion_quadrant"].to_numpy())
    y_test_enc = label_encoder.transform(test_df["emotion_quadrant"].to_numpy())
    class_names = label_encoder.classes_.tolist()

    for model_name, pipeline in models.items():
        X_train = train_df[feature_cols].to_numpy()
        X_val = val_df[feature_cols].to_numpy()
        X_test = test_df[feature_cols].to_numpy()

        pipeline.fit(X_train, y_train_enc)

        for split_name, X_split, y_enc in (
            ("val", X_val, y_val_enc),
            ("test", X_test, y_test_enc),
        ):
            y_pred_enc = pipeline.predict(X_split)
            y_true = label_encoder.inverse_transform(y_enc)
            y_pred = label_encoder.inverse_transform(y_pred_enc)
            metrics = compute_classification_metrics(y_true, y_pred, labels=class_names)

            cm = np.array(metrics["confusion_matrix"])
            plot_confusion_matrix(
                cm,
                labels=metrics["labels"],
                title=f"{model_name} — {split_name}",
                output_path=figures_dir / f"static_{model_name}_{split_name}_confusion_matrix.png",
            )

            metrics_payload = {
                "model_name": model_name,
                "task_type": "static",
                "feature_type": "spectral_aggregated",
                "target_type": "emotion_quadrant",
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
                "comments": "Thresholds fit on train tracks only; splits by track_id.",
            }

            metrics_path = metrics_dir / f"static_{model_name}_{split_name}_metrics.json"
            with metrics_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        **metrics_payload,
                        "classification_report": metrics["classification_report"],
                        "confusion_matrix": metrics["confusion_matrix"],
                        "labels": metrics["labels"],
                        "label_classes": class_names,
                    },
                    handle,
                    indent=2,
                )

            summary_rows.append(metrics_payload)
            logger.info(
                "%s [%s] — accuracy=%.4f, macro_f1=%.4f",
                model_name,
                split_name,
                metrics["accuracy"],
                metrics["macro_f1"],
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = metrics_dir / "static_baselines_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    reports_tables = resolve_path(root, configs["paths"]["reports"]["tables"])
    ensure_dir(reports_tables)
    summary_df.to_csv(reports_tables / "static_baselines_summary.csv", index=False)

    return summary_df
