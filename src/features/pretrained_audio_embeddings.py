"""Pretrained audio model embedding extraction (multi-model, dependency-aware)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.data.load_deam import build_metadata_table
from src.data.splits import load_track_splits
from src.features.mel_spectrograms import build_mel_spectrogram_index
from src.utils.config import ensure_dir, get_project_root, load_configs, resolve_path

logger = logging.getLogger(__name__)

INDEX_COLUMNS = [
    "song_id",
    "embedding_path",
    "embedding_dim",
    "split",
    "emotion_quadrant",
    "model_name",
    "model_alias",
    "backend",
    "sample_rate",
    "max_duration_sec",
]


def _embeddings_base_dir(configs: dict[str, dict[str, Any]]) -> Path:
    root = get_project_root()
    return resolve_path(root, configs["paths"]["features"]["pretrained_embeddings_dir"])


def _legacy_index_path(configs: dict[str, dict[str, Any]]) -> Path:
    root = get_project_root()
    return resolve_path(root, configs["paths"]["features"]["pretrained_embeddings_index"])


def _model_index_path(configs: dict[str, dict[str, Any]], model_alias: str) -> Path:
    return _embeddings_base_dir(configs) / model_alias / "index.csv"


def _model_embedding_dir(configs: dict[str, dict[str, Any]], model_cfg: dict[str, Any]) -> Path:
    base_dir = _embeddings_base_dir(configs)
    alias_dir = base_dir / str(model_cfg["alias"])
    if alias_dir.exists():
        return alias_dir

    legacy_dir = base_dir / str(model_cfg["model_name"]).replace("/", "__")
    if legacy_dir.exists():
        return legacy_dir

    return alias_dir


def resolve_pretrained_model_configs(
    configs: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Return normalized pretrained model configs.

    Supports the new ``features.pretrained.models`` list and the legacy single
    ``model_name`` block for backward compatibility.
    """
    if configs is None:
        configs = load_configs()

    pt_cfg = configs["features"]["pretrained"]
    models_cfg = pt_cfg.get("models")

    if models_cfg:
        normalized: list[dict[str, Any]] = []
        for entry in models_cfg:
            model_cfg = dict(entry)
            model_cfg["alias"] = str(model_cfg.get("alias", model_cfg["model_name"]).replace("/", "__"))
            model_cfg["model_name"] = str(model_cfg["model_name"])
            model_cfg["backend"] = str(model_cfg.get("backend", "hf_automodel"))
            model_cfg["sample_rate"] = int(model_cfg["sample_rate"])
            model_cfg["max_duration_sec"] = float(model_cfg["max_duration_sec"])
            model_cfg["pooling"] = str(model_cfg.get("pooling", "auto"))
            model_cfg["trust_remote_code"] = bool(model_cfg.get("trust_remote_code", False))
            normalized.append(model_cfg)
        return normalized

    if pt_cfg.get("model_name"):
        return [
            {
                "alias": str(pt_cfg.get("alias", pt_cfg["model_name"].replace("/", "__"))),
                "model_name": str(pt_cfg["model_name"]),
                "backend": str(pt_cfg.get("backend", "hf_automodel")),
                "sample_rate": int(pt_cfg["sample_rate"]),
                "max_duration_sec": float(pt_cfg["max_duration_sec"]),
                "pooling": str(pt_cfg.get("pooling", "auto")),
                "trust_remote_code": bool(pt_cfg.get("trust_remote_code", False)),
            }
        ]

    return []


def check_pretrained_dependencies(configs: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """
    Check whether optional pretrained-model dependencies are available.

    Returns a status dict with ``available``, ``provider``, ``models``, and ``message``.
    """
    if configs is None:
        configs = load_configs()

    pt_cfg = configs["features"]["pretrained"]
    provider = str(pt_cfg.get("provider", "huggingface")).lower()
    models = resolve_pretrained_model_configs(configs)

    status: dict[str, Any] = {
        "available": False,
        "provider": provider,
        "models": models,
        "model_name": models[0]["model_name"] if models else None,
        "message": "",
    }

    if not bool(pt_cfg.get("enabled", True)):
        status["message"] = "Pretrained embedding extraction is disabled in config."
        return status

    if not models:
        status["message"] = "No pretrained models configured in features.pretrained.models."
        return status

    try:
        import torch  # noqa: F401
    except ImportError:
        status["message"] = "PyTorch is not installed."
        return status

    if provider == "huggingface":
        try:
            import transformers  # noqa: F401
        except ImportError:
            status["message"] = (
                "Install transformers to use Hugging Face pretrained models: pip install transformers"
            )
            return status
        status["available"] = True
        status["message"] = f"Hugging Face transformers is available ({len(models)} model(s) configured)."
        return status

    status["message"] = f"Unsupported pretrained provider: {provider}"
    return status


def _hf_load_kwargs(
    allow_download: bool,
    hf_token: str | None,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if not allow_download:
        kwargs["local_files_only"] = True
    if hf_token:
        kwargs["token"] = hf_token
    else:
        kwargs["token"] = False
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    return kwargs


def _is_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error" in message


def _load_audio_segment(
    audio_path: Path,
    sample_rate: int,
    offset_sec: float = 0.0,
    duration_sec: float | None = None,
) -> np.ndarray:
    import librosa

    y, _ = librosa.load(
        audio_path,
        sr=sample_rate,
        mono=True,
        offset=offset_sec,
        duration=duration_sec,
    )
    if y.size == 0:
        raise ValueError(f"Audio segment is empty: {audio_path}")
    return y


def _pool_hf_automodel_outputs(outputs: Any) -> Any:
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        return outputs.pooler_output
    return outputs.last_hidden_state.mean(dim=1)


def _extract_hf_automodel_embedding(
    audio_path: Path,
    extractor: Any,
    model: Any,
    sample_rate: int,
    max_duration_sec: float,
    device: Any,
) -> np.ndarray:
    import torch

    y = _load_audio_segment(audio_path, sample_rate, duration_sec=max_duration_sec)
    inputs = extractor(y, sampling_rate=sample_rate, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = _pool_hf_automodel_outputs(outputs)

    return embedding.squeeze(0).cpu().numpy().astype(np.float32)


def _extract_mert_embedding(
    audio_path: Path,
    extractor: Any,
    model: Any,
    sample_rate: int,
    max_duration_sec: float,
    device: Any,
) -> np.ndarray:
    import torch

    def _forward(y: np.ndarray) -> np.ndarray:
        inputs = extractor(y, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            embedding = hidden.mean(dim=1)
        return embedding.squeeze(0).cpu().numpy().astype(np.float32)

    try:
        y = _load_audio_segment(audio_path, sample_rate, duration_sec=max_duration_sec)
        return _forward(y)
    except RuntimeError as exc:
        if not _is_oom_error(exc):
            raise
        if device.type == "cuda":
            torch.cuda.empty_cache()
        logger.warning(
            "MERT OOM on full track song_id context=%s; retrying with shorter duration.",
            audio_path.name,
        )

    for reduced_duration in (max_duration_sec / 2.0, 15.0, 10.0):
        if reduced_duration <= 0:
            continue
        try:
            y = _load_audio_segment(audio_path, sample_rate, duration_sec=reduced_duration)
            return _forward(y)
        except RuntimeError as exc:
            if not _is_oom_error(exc):
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()

    chunk_sec = 5.0
    chunk_embeddings: list[np.ndarray] = []
    offset = 0.0
    while offset < max_duration_sec:
        try:
            y = _load_audio_segment(
                audio_path,
                sample_rate,
                offset_sec=offset,
                duration_sec=chunk_sec,
            )
        except ValueError:
            break
        chunk_embeddings.append(_forward(y))
        offset += chunk_sec

    if not chunk_embeddings:
        raise RuntimeError(f"MERT embedding extraction failed for {audio_path}")

    return np.mean(np.vstack(chunk_embeddings), axis=0).astype(np.float32)


def _run_clap_processor(processor: Any, y: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Call ClapProcessor with transformers version-compatible kwargs."""
    try:
        return processor(audio=y, sampling_rate=sample_rate, return_tensors="pt")
    except (TypeError, ValueError) as exc:
        message = str(exc).lower()
        if "audios" in message or "audio" in message:
            return processor(audios=y, sampling_rate=sample_rate, return_tensors="pt")
        raise


def _tensor_from_clap_audio_output(output: Any) -> Any:
    """Extract a 2D audio embedding tensor from CLAP get_audio_features output."""
    if hasattr(output, "audio_embeds") and output.audio_embeds is not None:
        return output.audio_embeds
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    return output


def _extract_clap_embedding(
    audio_path: Path,
    processor: Any,
    model: Any,
    sample_rate: int,
    max_duration_sec: float,
    device: Any,
) -> np.ndarray:
    import torch

    y = _load_audio_segment(audio_path, sample_rate, duration_sec=max_duration_sec)
    inputs = _run_clap_processor(processor, y, sample_rate)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    model.eval()
    with torch.no_grad():
        output = model.get_audio_features(**inputs)
        embedding = _tensor_from_clap_audio_output(output)

    if not hasattr(embedding, "squeeze"):
        raise TypeError(
            f"Unexpected CLAP audio feature type: {type(output)!r}. "
            "Expected audio_embeds or pooler_output on model output."
        )

    return embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)


def _load_pretrained_backend(
    model_cfg: dict[str, Any],
    allow_download: bool,
    hf_token: str | None,
) -> tuple[str, Any, Any]:
    backend = str(model_cfg["backend"]).lower()
    model_name = str(model_cfg["model_name"])
    trust_remote_code = bool(model_cfg.get("trust_remote_code", False))
    kwargs = _hf_load_kwargs(allow_download, hf_token, trust_remote_code=trust_remote_code)

    if backend == "hf_automodel":
        from transformers import AutoFeatureExtractor, AutoModel

        extractor = AutoFeatureExtractor.from_pretrained(model_name, **kwargs)
        model = AutoModel.from_pretrained(model_name, **kwargs)
        return backend, extractor, model

    if backend == "mert_hf":
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, **kwargs)
        model = AutoModel.from_pretrained(model_name, **kwargs)
        return backend, extractor, model

    if backend == "clap_hf":
        from transformers import ClapModel, ClapProcessor

        processor = ClapProcessor.from_pretrained(model_name, **kwargs)
        model = ClapModel.from_pretrained(model_name, **kwargs)
        return backend, processor, model

    raise ValueError(f"Unsupported pretrained backend: {backend}")


def extract_track_pretrained_embedding(
    audio_path: str | Path,
    backend: str,
    extractor_or_processor: Any,
    model: Any,
    sample_rate: int,
    max_duration_sec: float,
    device: Any,
) -> np.ndarray:
    """Extract one track embedding using the configured backend."""
    audio_path = Path(audio_path)
    backend = backend.lower()

    if backend == "hf_automodel":
        return _extract_hf_automodel_embedding(
            audio_path,
            extractor_or_processor,
            model,
            sample_rate,
            max_duration_sec,
            device,
        )
    if backend == "mert_hf":
        return _extract_mert_embedding(
            audio_path,
            extractor_or_processor,
            model,
            sample_rate,
            max_duration_sec,
            device,
        )
    if backend == "clap_hf":
        return _extract_clap_embedding(
            audio_path,
            extractor_or_processor,
            model,
            sample_rate,
            max_duration_sec,
            device,
        )

    raise ValueError(f"Unsupported pretrained backend: {backend}")


def _load_legacy_index_for_model(
    configs: dict[str, dict[str, Any]],
    model_cfg: dict[str, Any],
    expected_tracks: int,
) -> pd.DataFrame | None:
    legacy_path = _legacy_index_path(configs)
    if not legacy_path.exists():
        return None

    legacy_df = pd.read_csv(legacy_path)
    if "model_name" not in legacy_df.columns:
        return None

    filtered = legacy_df[legacy_df["model_name"] == model_cfg["model_name"]].copy()
    if filtered.empty:
        return None
    if len(filtered) < expected_tracks:
        return None

    filtered["model_alias"] = model_cfg["alias"]
    filtered["backend"] = model_cfg["backend"]
    if "sample_rate" not in filtered.columns:
        filtered["sample_rate"] = model_cfg["sample_rate"]
    if "max_duration_sec" not in filtered.columns:
        filtered["max_duration_sec"] = model_cfg["max_duration_sec"]
    return filtered


def _index_is_complete(index_df: pd.DataFrame, expected_tracks: int, model_cfg: dict[str, Any]) -> bool:
    if len(index_df) < expected_tracks:
        return False
    if "model_name" in index_df.columns:
        names = index_df["model_name"].dropna().unique().tolist()
        if names and model_cfg["model_name"] not in names:
            return False
    return True


def extract_pretrained_embeddings_for_model(
    model_cfg: dict[str, Any],
    configs: dict[str, dict[str, Any]] | None = None,
    metadata: pd.DataFrame | None = None,
    complete_only: bool = True,
    force: bool = False,
) -> pd.DataFrame | None:
    """
    Extract and cache embeddings for one configured pretrained model.

    Saves ``{song_id}.npy`` files and ``index.csv`` under
    ``data/features/pretrained_embeddings/{model_alias}/``.
    """
    if configs is None:
        configs = load_configs()

    status = check_pretrained_dependencies(configs)
    if not status["available"]:
        logger.warning("Skipping %s: %s", model_cfg.get("alias"), status["message"])
        return None

    pt_cfg = configs["features"]["pretrained"]
    allow_download = bool(pt_cfg.get("allow_download", False))
    hf_token = pt_cfg.get("hf_token") or None
    cache_embeddings = bool(pt_cfg.get("cache_embeddings", True))

    if metadata is None:
        metadata = build_metadata_table(configs)

    model_alias = str(model_cfg["alias"])
    model_name = str(model_cfg["model_name"])
    backend = str(model_cfg["backend"])
    sample_rate = int(model_cfg["sample_rate"])
    max_duration_sec = float(model_cfg["max_duration_sec"])

    emb_dir = _model_embedding_dir(configs, model_cfg)
    index_path = _model_index_path(configs, model_alias)
    ensure_dir(emb_dir)

    root = get_project_root()
    labels_path = resolve_path(root, configs["paths"]["processed"]["static_labels"])
    labels_df = pd.read_parquet(labels_path) if labels_path.exists() else pd.read_csv(
        labels_path.with_suffix(".csv")
    )
    split_df = load_track_splits(configs)

    tracks = metadata.copy()
    if complete_only:
        tracks = tracks[tracks["is_complete"]].copy()
    expected_tracks = len(tracks)

    if cache_embeddings and not force:
        if index_path.exists():
            index_df = pd.read_csv(index_path)
            if _index_is_complete(index_df, expected_tracks, model_cfg):
                logger.info(
                    "Reusing cached embeddings for alias=%s (%d tracks) from %s",
                    model_alias,
                    len(index_df),
                    index_path,
                )
                return index_df

        legacy_df = _load_legacy_index_for_model(configs, model_cfg, expected_tracks)
        if legacy_df is not None:
            logger.info(
                "Reusing legacy embedding index for alias=%s (%d tracks) from %s",
                model_alias,
                len(legacy_df),
                _legacy_index_path(configs),
            )
            return legacy_df

    index_frame = build_mel_spectrogram_index(tracks, labels_df, split_df, configs)

    try:
        import torch
        from src.training.train_sequence import resolve_device

        loaded_backend, extractor_or_processor, model = _load_pretrained_backend(
            model_cfg,
            allow_download,
            hf_token,
        )
        device = resolve_device(str(pt_cfg.get("device", "auto")))
        model = model.to(device)
    except FileNotFoundError as exc:
        logger.warning("Could not load pretrained model alias=%s: %s", model_alias, exc)
        return None
    except Exception as exc:
        logger.warning(
            "Could not load pretrained model alias=%s backend=%s: %s",
            model_alias,
            backend,
            exc,
        )
        return None

    logger.info(
        "Extracting embeddings | alias=%s | model=%s | backend=%s | sample_rate=%d | max_duration=%.1fs | device=%s",
        model_alias,
        model_name,
        loaded_backend,
        sample_rate,
        max_duration_sec,
        device,
    )

    rows: list[dict[str, Any]] = []
    failed: list[int] = []
    first_error: str | None = None

    for _, track in tqdm(
        index_frame.iterrows(),
        total=len(index_frame),
        desc=f"Embeddings [{model_alias}]",
    ):
        song_id = int(track["song_id"])
        audio_path = track["audio_path"]
        if pd.isna(audio_path):
            failed.append(song_id)
            continue

        legacy_file = _embeddings_base_dir(configs) / model_name.replace("/", "__") / f"{song_id}.npy"
        target_file = emb_dir / f"{song_id}.npy"
        if cache_embeddings and not force and target_file.exists():
            embedding = np.load(target_file)
        elif cache_embeddings and not force and legacy_file.exists() and emb_dir == legacy_file.parent:
            embedding = np.load(legacy_file)
        else:
            try:
                embedding = extract_track_pretrained_embedding(
                    audio_path,
                    loaded_backend,
                    extractor_or_processor,
                    model,
                    sample_rate,
                    max_duration_sec,
                    device,
                )
            except Exception as exc:
                if first_error is None:
                    first_error = str(exc)
                if len(failed) < 3:
                    logger.warning(
                        "Embedding extraction failed for song_id=%s (%s): %s",
                        song_id,
                        model_alias,
                        exc,
                    )
                failed.append(song_id)
                continue

            np.save(target_file, embedding)

        rel_path = f"{model_alias}/{song_id}.npy"
        rows.append(
            {
                "song_id": song_id,
                "embedding_path": rel_path,
                "embedding_dim": int(embedding.shape[0]),
                "split": track["split"],
                "emotion_quadrant": track["emotion_quadrant"],
                "model_name": model_name,
                "model_alias": model_alias,
                "backend": loaded_backend,
                "sample_rate": sample_rate,
                "max_duration_sec": max_duration_sec,
            }
        )

    if not rows:
        logger.warning(
            "No embeddings extracted for alias=%s (%d failures). First error: %s",
            model_alias,
            len(failed),
            first_error or "unknown",
        )
        return None

    if len(failed) > 3:
        logger.warning(
            "Additional embedding failures for alias=%s: %d more tracks (first error: %s)",
            model_alias,
            len(failed) - 3,
            first_error,
        )

    index_df = pd.DataFrame(rows).sort_values("song_id").reset_index(drop=True)
    ensure_dir(index_path.parent)
    index_df.to_csv(index_path, index=False)

    embedding_dim = int(index_df["embedding_dim"].iloc[0])
    logger.info(
        "Saved embeddings for alias=%s | tracks=%d | skipped=%d | embedding_dim=%d | index=%s",
        model_alias,
        len(index_df),
        len(failed),
        embedding_dim,
        index_path,
    )
    return index_df


def extract_all_pretrained_embeddings(
    configs: dict[str, dict[str, Any]] | None = None,
    metadata: pd.DataFrame | None = None,
    complete_only: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """
    Extract embeddings for all configured pretrained models.

    Returns a combined index table. Models that fail to load are skipped with a warning.
    """
    if configs is None:
        configs = load_configs()

    if metadata is None:
        metadata = build_metadata_table(configs)

    model_configs = resolve_pretrained_model_configs(configs)
    combined_frames: list[pd.DataFrame] = []

    for model_cfg in model_configs:
        index_df = extract_pretrained_embeddings_for_model(
            model_cfg,
            configs=configs,
            metadata=metadata,
            complete_only=complete_only,
            force=force,
        )
        if index_df is not None:
            combined_frames.append(index_df)

    if not combined_frames:
        logger.warning("No pretrained embedding indexes were produced.")
        return pd.DataFrame(columns=INDEX_COLUMNS)

    combined = pd.concat(combined_frames, ignore_index=True)
    logger.info(
        "Combined pretrained embedding index: %d rows across %d model(s).",
        len(combined),
        combined["model_alias"].nunique(),
    )
    return combined


def extract_pretrained_embeddings_dataset(
    metadata: pd.DataFrame | None = None,
    configs: dict[str, dict[str, Any]] | None = None,
    complete_only: bool = True,
    force: bool = False,
) -> pd.DataFrame | None:
    """
    Backward-compatible wrapper around ``extract_all_pretrained_embeddings``.

    Returns ``None`` if dependencies are unavailable or no embeddings were extracted.
    """
    combined = extract_all_pretrained_embeddings(
        configs=configs,
        metadata=metadata,
        complete_only=complete_only,
        force=force,
    )
    if combined.empty:
        return None
    return combined


def load_pretrained_embeddings_index(
    configs: dict[str, dict[str, Any]] | None = None,
    model_alias: str | None = None,
) -> pd.DataFrame:
    """Load a per-model embedding index, or all models when ``model_alias`` is None."""
    if configs is None:
        configs = load_configs()

    if model_alias is not None:
        index_path = _model_index_path(configs, model_alias)
        if index_path.exists():
            return pd.read_csv(index_path)

        model_cfg = next(
            (cfg for cfg in resolve_pretrained_model_configs(configs) if cfg["alias"] == model_alias),
            None,
        )
        if model_cfg is None:
            raise FileNotFoundError(f"Unknown pretrained model alias: {model_alias}")

        legacy_df = _load_legacy_index_for_model(configs, model_cfg, expected_tracks=1)
        if legacy_df is not None:
            return legacy_df

        raise FileNotFoundError(
            f"Pretrained embeddings index not found for alias '{model_alias}' at {index_path}. "
            "Run extract_pretrained_embeddings_for_model() first."
        )

    model_configs = resolve_pretrained_model_configs(configs)
    frames: list[pd.DataFrame] = []
    missing: list[str] = []

    for model_cfg in model_configs:
        alias = str(model_cfg["alias"])
        try:
            frames.append(load_pretrained_embeddings_index(configs, model_alias=alias))
        except FileNotFoundError:
            missing.append(alias)

    if not frames:
        legacy_path = _legacy_index_path(configs)
        if legacy_path.exists():
            return pd.read_csv(legacy_path)
        raise FileNotFoundError(
            "No pretrained embedding indexes found. Run extract_all_pretrained_embeddings() first."
        )

    if missing:
        logger.warning("Missing pretrained embedding indexes for aliases: %s", ", ".join(missing))

    return pd.concat(frames, ignore_index=True)


def get_pretrained_embeddings_dir(
    configs: dict[str, dict[str, Any]] | None = None,
    model_alias: str | None = None,
) -> Path:
    if configs is None:
        configs = load_configs()

    base_dir = _embeddings_base_dir(configs)
    if model_alias is None:
        return base_dir

    model_cfg = next(
        (cfg for cfg in resolve_pretrained_model_configs(configs) if cfg["alias"] == model_alias),
        {"alias": model_alias, "model_name": model_alias},
    )
    return _model_embedding_dir(configs, model_cfg)


def resolve_embedding_file_path(
    row: pd.Series,
    configs: dict[str, dict[str, Any]] | None = None,
) -> Path:
    """Resolve an on-disk embedding path from an index row."""
    if configs is None:
        configs = load_configs()

    base_dir = _embeddings_base_dir(configs)
    embedding_path = Path(str(row["embedding_path"]))

    candidates = [
        base_dir / embedding_path,
        base_dir / str(row.get("model_alias", "")) / embedding_path.name,
        base_dir / str(row.get("model_name", "")).replace("/", "__") / embedding_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    song_id = int(row["song_id"])
    model_alias = str(row.get("model_alias", ""))
    if model_alias:
        alias_candidate = base_dir / model_alias / f"{song_id}.npy"
        if alias_candidate.exists():
            return alias_candidate

    raise FileNotFoundError(f"Embedding file not found for song_id={song_id}: {embedding_path}")
