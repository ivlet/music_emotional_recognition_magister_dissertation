"""Training and evaluation for Mel-spectrogram CNN/CRNN models."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.data.load_deam import build_metadata_table
from src.data.splits import assert_no_track_leakage
from src.features.mel_spectrograms import (
    extract_mel_spectrograms_dataset,
    get_mel_spectrogram_dir,
    load_mel_spectrogram_index,
)
from src.models.cnn import build_cnn_model
from src.models.crnn import build_crnn_model
from src.training.train_sequence import log_training_device, resolve_device
from src.training.train_static import compute_classification_metrics, plot_confusion_matrix
from src.utils.config import ensure_dir, get_project_root, load_configs, resolve_path

logger = logging.getLogger(__name__)

MODEL_BUILDERS: dict[str, Callable[..., nn.Module]] = {
    "cnn": build_cnn_model,
    "crnn": build_crnn_model,
}


class MelSpectrogramDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        mel_dir: Path,
        label_encoder: LabelEncoder,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.mel_dir = mel_dir
        self.label_encoder = label_encoder

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.frame.iloc[idx]
        song_id = int(row["song_id"])
        mel = np.load(self.mel_dir / f"{song_id}.npy")
        label = int(self.label_encoder.transform([row["emotion_quadrant"]])[0])
        tensor = torch.from_numpy(mel).unsqueeze(0)
        return {"spectrogram": tensor, "labels": label, "song_id": song_id}


def validate_spectrogram_splits(frame: pd.DataFrame) -> None:
    if frame["split"].isna().any():
        raise ValueError("Found tracks with missing split values.")
    splits_dict = {
        split: frame.loc[frame["split"] == split, "song_id"].astype(int).tolist()
        for split in ("train", "val", "test")
    }
    assert_no_track_leakage(splits_dict)
    logger.info("Spectrogram split validation passed (track-level).")


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    losses: list[float] = []
    all_preds: list[int] = []
    all_labels: list[int] = []

    for batch in loader:
        x = batch["spectrogram"].to(device)
        labels = batch["labels"].to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        losses.append(float(loss.item()))
        all_preds.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    return float(np.mean(losses)), np.array(all_labels), np.array(all_preds)


def train_spectrogram_model(
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
    train_cfg = configs["training"]["spectrogram"]
    spec_cfg = configs["features"]["spectrogram"]
    n_mels = int(spec_cfg["n_mels"])
    batch_size = int(train_cfg["batch_size"])
    epochs = int(train_cfg["epochs"])
    lr = float(train_cfg["learning_rate"])
    patience = int(train_cfg["early_stopping_patience"])

    mel_dir = get_mel_spectrogram_dir(configs)
    num_classes = len(label_encoder.classes_)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        MelSpectrogramDataset(train_df, mel_dir, label_encoder),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        MelSpectrogramDataset(val_df, mel_dir, label_encoder),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        MelSpectrogramDataset(test_df, mel_dir, label_encoder),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    model = MODEL_BUILDERS[model_name](n_mels=n_mels, num_classes=num_classes, configs=configs)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_f1 = -1.0
    best_state: dict[str, Any] | None = None
    epochs_without_improve = 0

    for epoch in range(epochs):
        train_loss, _, _ = _run_epoch(model, train_loader, criterion, device, optimizer)
        _, y_val, y_val_pred = _run_epoch(model, val_loader, criterion, device, None)
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

    for split_name, loader in (("val", val_loader), ("test", test_loader)):
        _, y_true_enc, y_pred_enc = _run_epoch(model, loader, criterion, device, None)
        y_true = label_encoder.inverse_transform(y_true_enc)
        y_pred = label_encoder.inverse_transform(y_pred_enc)
        metrics = compute_classification_metrics(y_true, y_pred, labels=class_names)

        if split_name == "test":
            plot_confusion_matrix(
                np.array(metrics["confusion_matrix"]),
                labels=metrics["labels"],
                title=f"{model_name} — spectrogram model (test)",
                output_path=figures_dir / f"spectrogram_{model_name}_confusion_matrix.png",
            )

        payload = {
            "model_name": model_name,
            "task_type": "spectrogram",
            "feature_type": "mel_spectrogram",
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
        metrics_path = metrics_dir / f"spectrogram_{model_name}_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {**payload, "classification_report": metrics["classification_report"],
                 "confusion_matrix": metrics["confusion_matrix"], "labels": metrics["labels"]},
                handle,
                indent=2,
            )
        summary_rows.append(payload)

    return summary_rows


def train_and_evaluate_spectrogram_models(
    configs: dict[str, dict[str, Any]] | None = None,
    model_names: tuple[str, ...] | None = None,
    extract_features: bool = True,
    force_extract: bool = False,
) -> pd.DataFrame:
    """Extract Mel-spectrograms (if needed) and train CNN/CRNN classifiers."""
    if configs is None:
        configs = load_configs()

    root = get_project_root()
    random_state = int(configs["training"]["general"]["random_state"])
    device = resolve_device(str(configs["training"]["general"]["device"]))
    log_training_device(device)

    if model_names is None:
        model_names = ("cnn", "crnn")

    if extract_features:
        metadata = build_metadata_table(configs)
        index_df = extract_mel_spectrograms_dataset(metadata, configs, force=force_extract)
    else:
        index_df = load_mel_spectrogram_index(configs)

    validate_spectrogram_splits(index_df)

    train_df = index_df[index_df["split"] == "train"]
    val_df = index_df[index_df["split"] == "val"]
    test_df = index_df[index_df["split"] == "test"]

    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["emotion_quadrant"].to_numpy())

    metrics_dir = resolve_path(root, configs["paths"]["results"]["metrics"])
    figures_dir = resolve_path(root, configs["paths"]["reports"]["figures"])
    ensure_dir(metrics_dir)
    ensure_dir(figures_dir)

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    all_rows: list[dict[str, Any]] = []
    for model_name in tqdm(model_names, desc="Training spectrogram models"):
        logger.info("Training spectrogram model: %s on %s", model_name, device)
        rows = train_spectrogram_model(
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
    summary_path = metrics_dir / "spectrogram_models_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved spectrogram model summary to %s", summary_path)
    return summary_df
