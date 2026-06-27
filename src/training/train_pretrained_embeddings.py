"""Train classifiers on pretrained audio embeddings."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

from src.data.splits import assert_no_track_leakage
from src.features.pretrained_audio_embeddings import (
    check_pretrained_dependencies,
    extract_all_pretrained_embeddings,
    extract_pretrained_embeddings_dataset,
    load_pretrained_embeddings_index,
    resolve_embedding_file_path,
    resolve_pretrained_model_configs,
)
from src.models.embedding_classifiers import build_embedding_classical_models
from src.training.train_static import compute_classification_metrics, plot_confusion_matrix
from src.utils.config import ensure_dir, get_project_root, load_configs, resolve_path

logger = logging.getLogger(__name__)


def _load_embedding_matrix(index_df: pd.DataFrame, configs: dict[str, dict[str, Any]]) -> tuple[np.ndarray, pd.DataFrame]:
    vectors: list[np.ndarray] = []
    for _, row in index_df.iterrows():
        path = resolve_embedding_file_path(row, configs)
        vectors.append(np.load(path))
    X = np.vstack(vectors)
    return X, index_df.reset_index(drop=True)


def validate_pretrained_splits(frame: pd.DataFrame) -> None:
    if frame["split"].isna().any():
        raise ValueError("Found tracks with missing split values.")
    splits_dict = {
        split: frame.loc[frame["split"] == split, "song_id"].astype(int).tolist()
        for split in ("train", "val", "test")
    }
    assert_no_track_leakage(splits_dict)


def _summary_row(
    model_cfg: dict[str, Any],
    clf_name: str,
    split_name: str,
    metrics: dict[str, Any],
    train_size: int,
    val_size: int,
    test_size: int,
    embedding_dim: int,
) -> dict[str, Any]:
    return {
        "model_name": str(model_cfg["model_name"]),
        "model_alias": str(model_cfg["alias"]),
        "backend": str(model_cfg["backend"]),
        "classifier": clf_name,
        "task_type": "pretrained_embedding",
        "feature_type": "pretrained_audio_embedding",
        "target_type": "emotion_quadrant",
        "eval_split": split_name,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "embedding_dim": embedding_dim,
        "sample_rate": int(model_cfg["sample_rate"]),
        "max_duration_sec": float(model_cfg["max_duration_sec"]),
    }


def train_pretrained_embedding_classifiers_for_model(
    model_cfg: dict[str, Any],
    index_df: pd.DataFrame,
    configs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Train sklearn classifiers for one pretrained model index."""
    validate_pretrained_splits(index_df)

    root = get_project_root()
    random_state = int(configs["training"]["general"]["random_state"])
    model_alias = str(model_cfg["alias"])

    X, frame = _load_embedding_matrix(index_df, configs)
    train_mask = frame["split"] == "train"
    val_mask = frame["split"] == "val"
    test_mask = frame["split"] == "test"

    embedding_dim = int(frame["embedding_dim"].iloc[0])
    logger.info(
        "Training classifiers | alias=%s | backend=%s | shape=%s | train=%d val=%d test=%d | embedding_dim=%d",
        model_alias,
        model_cfg["backend"],
        X.shape,
        int(train_mask.sum()),
        int(val_mask.sum()),
        int(test_mask.sum()),
        embedding_dim,
    )

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(frame.loc[train_mask, "emotion_quadrant"])
    y_val = label_encoder.transform(frame.loc[val_mask, "emotion_quadrant"])
    y_test = label_encoder.transform(frame.loc[test_mask, "emotion_quadrant"])
    class_names = label_encoder.classes_.tolist()

    models = build_embedding_classical_models(configs, random_state=random_state)
    metrics_dir = resolve_path(root, configs["paths"]["results"]["metrics"])
    figures_dir = resolve_path(root, configs["paths"]["reports"]["figures"])
    ensure_dir(metrics_dir)
    ensure_dir(figures_dir)

    summary_rows: list[dict[str, Any]] = []

    for clf_name, pipeline in tqdm(models.items(), desc=f"Classifiers [{model_alias}]"):
        logger.info("Training classifier alias=%s classifier=%s", model_alias, clf_name)
        pipeline.fit(X[train_mask], y_train)

        split_metrics: dict[str, Any] = {
            "model_name": model_cfg["model_name"],
            "model_alias": model_alias,
            "backend": model_cfg["backend"],
            "classifier": clf_name,
            "task_type": "pretrained_embedding",
            "feature_type": "pretrained_audio_embedding",
            "target_type": "emotion_quadrant",
            "train_size": int(train_mask.sum()),
            "val_size": int(val_mask.sum()),
            "test_size": int(test_mask.sum()),
            "embedding_dim": embedding_dim,
            "sample_rate": int(model_cfg["sample_rate"]),
            "max_duration_sec": float(model_cfg["max_duration_sec"]),
        }

        for split_name, mask, y_true_enc in (
            ("val", val_mask, y_val),
            ("test", test_mask, y_test),
        ):
            y_pred_enc = pipeline.predict(X[mask])
            y_true = label_encoder.inverse_transform(y_true_enc)
            y_pred = label_encoder.inverse_transform(y_pred_enc)
            metrics = compute_classification_metrics(y_true, y_pred, labels=class_names)

            if split_name == "test":
                plot_confusion_matrix(
                    np.array(metrics["confusion_matrix"]),
                    labels=metrics["labels"],
                    title=f"{model_alias} / {clf_name} — pretrained embedding (test)",
                    output_path=figures_dir / f"pretrained_{model_alias}_{clf_name}_confusion_matrix.png",
                )

            split_metrics[split_name] = {
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "classification_report": metrics["classification_report"],
                "confusion_matrix": metrics["confusion_matrix"],
            }

            summary_rows.append(
                _summary_row(
                    model_cfg,
                    clf_name,
                    split_name,
                    metrics,
                    int(train_mask.sum()),
                    int(val_mask.sum()),
                    int(test_mask.sum()),
                    embedding_dim,
                )
            )
            logger.info(
                "%s / %s [%s] — accuracy=%.4f, macro_f1=%.4f, weighted_f1=%.4f",
                model_alias,
                clf_name,
                split_name,
                metrics["accuracy"],
                metrics["macro_f1"],
                metrics["weighted_f1"],
            )

        metrics_path = metrics_dir / f"pretrained_{model_alias}_{clf_name}_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(split_metrics, handle, indent=2)

    return summary_rows


def train_all_pretrained_embedding_classifiers(
    configs: dict[str, dict[str, Any]] | None = None,
    extract_embeddings: bool = True,
    force_extract: bool = False,
) -> pd.DataFrame:
    """
    Extract embeddings (if requested) and train classifiers for all configured models.

    Returns an empty DataFrame if nothing could be trained.
    """
    if configs is None:
        configs = load_configs()

    status = check_pretrained_dependencies(configs)
    if not status["available"]:
        logger.warning("Pretrained embedding training skipped: %s", status["message"])
        return pd.DataFrame()

    model_configs = resolve_pretrained_model_configs(configs)
    if not model_configs:
        logger.warning("No pretrained models configured.")
        return pd.DataFrame()

    if extract_embeddings:
        from src.data.load_deam import build_metadata_table

        metadata = build_metadata_table(configs)
        extract_all_pretrained_embeddings(
            configs=configs,
            metadata=metadata,
            force=force_extract,
        )

    all_rows: list[dict[str, Any]] = []

    for model_cfg in model_configs:
        model_alias = str(model_cfg["alias"])
        try:
            index_df = load_pretrained_embeddings_index(configs, model_alias=model_alias)
        except FileNotFoundError as exc:
            logger.warning("Skipping alias=%s: %s", model_alias, exc)
            continue

        try:
            rows = train_pretrained_embedding_classifiers_for_model(model_cfg, index_df, configs)
            all_rows.extend(rows)
        except Exception as exc:
            logger.warning("Classifier training failed for alias=%s: %s", model_alias, exc)
            continue

    summary_df = pd.DataFrame(all_rows)
    root = get_project_root()
    metrics_dir = resolve_path(root, configs["paths"]["results"]["metrics"])
    ensure_dir(metrics_dir)
    summary_path = metrics_dir / "pretrained_audio_models_summary.csv"

    if summary_df.empty:
        logger.warning("No pretrained classifier results were produced.")
        return summary_df

    summary_df.to_csv(summary_path, index=False)
    logger.info(
        "Saved pretrained model summary to %s (%d rows, %d model(s))",
        summary_path,
        len(summary_df),
        summary_df["model_alias"].nunique(),
    )
    return summary_df


def train_pretrained_embedding_classifiers(
    configs: dict[str, dict[str, Any]] | None = None,
    index_df: pd.DataFrame | None = None,
    extract_embeddings: bool = True,
    force_extract: bool = False,
) -> pd.DataFrame | None:
    """
    Backward-compatible entry point.

    When ``index_df`` is provided, trains classifiers only for that model's alias.
    Otherwise delegates to ``train_all_pretrained_embedding_classifiers``.
    """
    if configs is None:
        configs = load_configs()

    if index_df is not None:
        model_alias = str(index_df["model_alias"].iloc[0]) if "model_alias" in index_df.columns else None
        if model_alias is None and "model_name" in index_df.columns:
            model_name = str(index_df["model_name"].iloc[0])
            model_cfg = next(
                (
                    cfg
                    for cfg in resolve_pretrained_model_configs(configs)
                    if cfg["model_name"] == model_name
                ),
                {
                    "alias": model_name.replace("/", "__"),
                    "model_name": model_name,
                    "backend": "hf_automodel",
                    "sample_rate": int(configs["features"]["pretrained"].get("sample_rate", 16000)),
                    "max_duration_sec": float(
                        configs["features"]["pretrained"].get("max_duration_sec", 30.0)
                    ),
                },
            )
        else:
            model_cfg = next(
                (cfg for cfg in resolve_pretrained_model_configs(configs) if cfg["alias"] == model_alias),
                None,
            )
            if model_cfg is None:
                logger.warning("Could not resolve model config for alias=%s", model_alias)
                return None

        rows = train_pretrained_embedding_classifiers_for_model(model_cfg, index_df, configs)
        summary_df = pd.DataFrame(rows)
        if summary_df.empty:
            return None

        root = get_project_root()
        metrics_dir = resolve_path(root, configs["paths"]["results"]["metrics"])
        ensure_dir(metrics_dir)
        summary_path = metrics_dir / "pretrained_audio_models_summary.csv"

        if summary_path.exists():
            existing = pd.read_csv(summary_path)
            if "model_alias" in existing.columns:
                existing = existing[existing["model_alias"] != model_cfg["alias"]]
            summary_df = pd.concat([existing, summary_df], ignore_index=True)

        summary_df.to_csv(summary_path, index=False)
        logger.info("Saved pretrained model summary to %s", summary_path)
        return summary_df

    summary_df = train_all_pretrained_embedding_classifiers(
        configs=configs,
        extract_embeddings=extract_embeddings,
        force_extract=force_extract,
    )
    if summary_df.empty:
        return None
    return summary_df
