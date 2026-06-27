"""Training and evaluation for sequence-based emotion classifiers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm

from src.data.load_deam import build_metadata_table
from src.data.splits import assign_splits_to_dataframe, assert_no_track_leakage, load_track_splits
from src.data.make_static_dataset import assign_emotion_quadrant
from src.features.mfcc_sequences import (
    _sequence_paths,
    extract_mfcc_sequences_dataset,
    load_mfcc_manifest,
)
from src.models.mlp import aggregate_sequence_features, build_mlp_model
from src.models.rnn import build_attention_model, build_gru_model, build_lstm_model
from src.models.transformer_encoder import build_transformer_model
from src.training.train_static import compute_classification_metrics, plot_confusion_matrix
from src.utils.config import ensure_dir, get_project_root, load_configs, resolve_path

logger = logging.getLogger(__name__)

SEQUENCE_MODEL_BUILDERS: dict[str, Callable[..., nn.Module]] = {
    "mlp": build_mlp_model,
    "lstm": build_lstm_model,
    "gru": build_gru_model,
    "attention": build_attention_model,
    "transformer": build_transformer_model,
}


def resolve_device(device_cfg: str = "auto") -> torch.device:
    if device_cfg == "cuda":
        if not torch.cuda.is_available():
            logger.warning("device=cuda requested but CUDA is unavailable; using CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    if device_cfg == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def log_training_device(device: torch.device) -> None:
    if device.type == "cuda":
        logger.info("Training device: cuda (%s)", torch.cuda.get_device_name(device))
    else:
        logger.warning(
            "Training device: cpu (set training.general.device to 'cuda' or install a CUDA build of PyTorch)."
        )


class TrackMFCCDataset(Dataset):
    """PyTorch dataset of per-track MFCC sequences with track-level labels."""

    def __init__(
        self,
        frame: pd.DataFrame,
        mfcc_dir: Path,
        max_sequence_length: int,
        label_encoder: LabelEncoder,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.mfcc_dir = mfcc_dir
        self.max_sequence_length = max_sequence_length
        self.label_encoder = label_encoder

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.frame.iloc[idx]
        song_id = int(row["song_id"])
        path = self.mfcc_dir / f"{song_id}.npy"
        sequence = np.load(path)

        if sequence.shape[0] > self.max_sequence_length:
            sequence = sequence[: self.max_sequence_length]

        length = int(sequence.shape[0])
        label = int(self.label_encoder.transform([row["emotion_quadrant"]])[0])

        return {
            "sequence": torch.from_numpy(sequence),
            "length": length,
            "label": label,
            "song_id": song_id,
        }


def collate_sequences(batch: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    feat_dim = batch[0]["sequence"].shape[1]

    sequences = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    for i, item in enumerate(batch):
        seq = item["sequence"]
        sequences[i, : item["length"]] = seq

    return {
        "sequences": sequences,
        "lengths": lengths,
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        "song_ids": [item["song_id"] for item in batch],
    }


def prepare_sequence_training_frame(
    manifest: pd.DataFrame,
    labels_df: pd.DataFrame,
    split_df: pd.DataFrame,
    configs: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Merge MFCC manifest with labels and track-level splits."""
    label_cols = ["song_id", "valence", "arousal"]
    merged = manifest.merge(labels_df[label_cols], on="song_id", how="inner")
    merged = assign_splits_to_dataframe(merged, split_df, track_col="song_id")

    train_mask = merged["split"] == "train"
    train_labels = assign_emotion_quadrant(
        merged.loc[train_mask, "valence"],
        merged.loc[train_mask, "arousal"],
        valence_threshold=configs["features"]["emotion"]["valence_threshold"],
        arousal_threshold=configs["features"]["emotion"]["arousal_threshold"],
    )
    v_cut = float(train_labels["valence_threshold"].iloc[0])
    a_cut = float(train_labels["arousal_threshold"].iloc[0])

    all_labels = assign_emotion_quadrant(
        merged["valence"],
        merged["arousal"],
        valence_threshold=v_cut,
        arousal_threshold=a_cut,
    )
    merged["emotion_quadrant"] = all_labels["emotion_quadrant"].values
    return merged


def validate_sequence_splits(frame: pd.DataFrame) -> None:
    if frame["split"].isna().any():
        raise ValueError("Found tracks with missing split values.")
    splits_dict = {
        split: frame.loc[frame["split"] == split, "song_id"].astype(int).tolist()
        for split in ("train", "val", "test")
    }
    assert_no_track_leakage(splits_dict)
    logger.info("Sequence split validation passed (track-level, no frame leakage).")


def _forward_batch(
    model: nn.Module,
    model_name: str,
    batch: dict[str, Any],
    n_mfcc: int,
) -> torch.Tensor:
    sequences = batch["sequences"]
    lengths = batch["lengths"]
    if model_name == "mlp":
        aggregated = aggregate_sequence_features(sequences, lengths)
        return model(aggregated)
    return model(sequences, lengths)


def _run_epoch(
    model: nn.Module,
    model_name: str,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    n_mfcc: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    losses: list[float] = []
    all_preds: list[int] = []
    all_labels: list[int] = []

    for batch in loader:
        sequences = batch["sequences"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)
        batch_t = {"sequences": sequences, "lengths": lengths, "labels": labels}

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = _forward_batch(model, model_name, batch_t, n_mfcc)
            loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        losses.append(float(loss.item()))
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    return float(np.mean(losses)), np.array(all_labels), np.array(all_preds)


def train_sequence_model(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    configs: dict[str, dict[str, Any]],
    label_encoder: LabelEncoder,
    device: torch.device,
    metrics_dir: Path,
    figures_dir: Path,
) -> list[dict[str, Any]]:
    """Train one sequence model and evaluate on val/test."""
    seq_cfg = configs["features"]["sequence"]
    train_cfg = configs["training"]["sequence"]
    n_mfcc = int(seq_cfg["n_mfcc"])
    max_seq_len = int(seq_cfg["max_sequence_length"])
    batch_size = int(train_cfg["batch_size"])
    epochs = int(train_cfg["epochs"])
    lr = float(train_cfg["learning_rate"])
    patience = int(train_cfg["early_stopping_patience"])

    mfcc_dir, _ = _sequence_paths(configs)
    num_classes = len(label_encoder.classes_)

    train_ds = TrackMFCCDataset(train_df, mfcc_dir, max_seq_len, label_encoder)
    val_ds = TrackMFCCDataset(val_df, mfcc_dir, max_seq_len, label_encoder)
    test_ds = TrackMFCCDataset(test_df, mfcc_dir, max_seq_len, label_encoder)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_sequences,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_sequences,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_sequences,
        pin_memory=pin_memory,
    )

    if model_name == "mlp":
        model = build_mlp_model(input_dim=n_mfcc * 2, num_classes=num_classes, configs=configs)
    else:
        model = SEQUENCE_MODEL_BUILDERS[model_name](input_dim=n_mfcc, num_classes=num_classes, configs=configs)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = -1.0
    best_state: dict[str, Any] | None = None
    epochs_without_improve = 0

    for epoch in range(epochs):
        train_loss, _, _ = _run_epoch(
            model, model_name, train_loader, criterion, optimizer, device, n_mfcc
        )
        _, y_val, y_val_pred = _run_epoch(
            model, model_name, val_loader, criterion, None, device, n_mfcc
        )
        y_val_true = label_encoder.inverse_transform(y_val)
        y_val_pred_labels = label_encoder.inverse_transform(y_val_pred)
        val_metrics = compute_classification_metrics(
            y_val_true, y_val_pred_labels, labels=label_encoder.classes_.tolist()
        )

        logger.info(
            "%s epoch %d/%d — train_loss=%.4f, val_macro_f1=%.4f",
            model_name,
            epoch + 1,
            epochs,
            train_loss,
            val_metrics["macro_f1"],
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                logger.info("%s early stopping at epoch %d.", model_name, epoch + 1)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    summary_rows: list[dict[str, Any]] = []
    class_names = label_encoder.classes_.tolist()

    for split_name, loader, split_df in (
        ("val", val_loader, val_df),
        ("test", test_loader, test_df),
    ):
        _, y_true_enc, y_pred_enc = _run_epoch(
            model, model_name, loader, criterion, None, device, n_mfcc
        )
        y_true = label_encoder.inverse_transform(y_true_enc)
        y_pred = label_encoder.inverse_transform(y_pred_enc)
        metrics = compute_classification_metrics(y_true, y_pred, labels=class_names)

        if split_name == "test":
            cm = np.array(metrics["confusion_matrix"])
            plot_confusion_matrix(
                cm,
                labels=metrics["labels"],
                title=f"{model_name} — sequence model (test)",
                output_path=figures_dir / f"sequence_{model_name}_confusion_matrix.png",
            )

        metrics_payload = {
            "model_name": model_name,
            "task_type": "sequence",
            "feature_type": "mfcc_sequence",
            "target_type": "emotion_quadrant",
            "eval_split": split_name,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "best_val_macro_f1": best_val_f1,
        }

        metrics_path = metrics_dir / f"sequence_{model_name}_{split_name}_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    **metrics_payload,
                    "classification_report": metrics["classification_report"],
                    "confusion_matrix": metrics["confusion_matrix"],
                    "labels": metrics["labels"],
                },
                handle,
                indent=2,
            )

        summary_rows.append(metrics_payload)

    return summary_rows


def train_and_evaluate_sequence_models(
    configs: dict[str, dict[str, Any]] | None = None,
    model_names: tuple[str, ...] | None = None,
    extract_features: bool = True,
    force_extract: bool = False,
) -> pd.DataFrame:
    """
    Extract MFCC sequences (if needed) and train sequence classifiers.

    Models: mlp, lstm, gru, attention, transformer.
    """
    if configs is None:
        configs = load_configs()

    root = get_project_root()
    random_state = int(configs["training"]["general"]["random_state"])
    device = resolve_device(str(configs["training"]["general"]["device"]))
    log_training_device(device)

    if model_names is None:
        model_names = ("mlp", "lstm", "gru", "attention", "transformer")

    metadata = build_metadata_table(configs)
    if extract_features:
        manifest = extract_mfcc_sequences_dataset(
            metadata, configs, complete_only=True, force=force_extract
        )
    else:
        manifest = load_mfcc_manifest(configs)

    labels_path = resolve_path(root, configs["paths"]["processed"]["static_labels"])
    if labels_path.suffix == ".parquet":
        labels_df = pd.read_parquet(labels_path)
    else:
        labels_df = pd.read_csv(labels_path)

    split_df = load_track_splits(configs)
    data = prepare_sequence_training_frame(manifest, labels_df, split_df, configs)
    validate_sequence_splits(data)

    train_df = data[data["split"] == "train"]
    val_df = data[data["split"] == "val"]
    test_df = data[data["split"] == "test"]

    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["emotion_quadrant"].to_numpy())

    metrics_dir = resolve_path(root, configs["paths"]["results"]["metrics"])
    figures_dir = resolve_path(root, configs["paths"]["reports"]["figures"])
    ensure_dir(metrics_dir)
    ensure_dir(figures_dir)

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    all_rows: list[dict[str, Any]] = []
    for model_name in tqdm(model_names, desc="Training sequence models"):
        logger.info("Training sequence model: %s on %s", model_name, device)
        rows = train_sequence_model(
            model_name,
            train_df,
            val_df,
            test_df,
            configs,
            label_encoder,
            device,
            metrics_dir,
            figures_dir,
        )
        all_rows.extend(rows)

    summary_df = pd.DataFrame(all_rows)
    summary_path = metrics_dir / "sequence_models_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved sequence model summary to %s", summary_path)
    return summary_df
