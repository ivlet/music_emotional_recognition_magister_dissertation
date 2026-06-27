"""Classifiers trained on pretrained audio embeddings."""

from __future__ import annotations

import logging
from typing import Any

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _preprocess_steps() -> list[tuple[str, Any]]:
    return [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]


def build_embedding_classical_models(
    configs: dict[str, dict[str, Any]],
    random_state: int,
) -> dict[str, Pipeline]:
    """Sklearn classifiers for embedding vectors."""
    static_cfg = configs["models"]["static"]
    models: dict[str, Pipeline] = {}

    models["logistic_regression"] = Pipeline(
        [
            *_preprocess_steps(),
            (
                "clf",
                LogisticRegression(
                    C=float(static_cfg["logistic_regression"]["C"]),
                    max_iter=int(static_cfg["logistic_regression"]["max_iter"]),
                    random_state=random_state,
                    multi_class="auto",
                ),
            ),
        ]
    )

    rf_params: dict[str, Any] = {
        "n_estimators": int(static_cfg["random_forest"]["n_estimators"]),
        "random_state": random_state,
    }
    if static_cfg["random_forest"].get("max_depth") is not None:
        rf_params["max_depth"] = int(static_cfg["random_forest"]["max_depth"])
    models["random_forest"] = Pipeline([*_preprocess_steps(), ("clf", RandomForestClassifier(**rf_params))])

    models["gradient_boosting"] = Pipeline(
        [
            *_preprocess_steps(),
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=int(static_cfg["gradient_boosting"]["n_estimators"]),
                    learning_rate=float(static_cfg["gradient_boosting"]["learning_rate"]),
                    max_depth=int(static_cfg["gradient_boosting"]["max_depth"]),
                    random_state=random_state,
                ),
            ),
        ]
    )

    if XGBOOST_AVAILABLE:
        models["xgboost"] = Pipeline(
            [
                *_preprocess_steps(),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=int(static_cfg["xgboost"]["n_estimators"]),
                        learning_rate=float(static_cfg["xgboost"]["learning_rate"]),
                        max_depth=int(static_cfg["xgboost"]["max_depth"]),
                        random_state=random_state,
                        eval_metric="mlogloss",
                        verbosity=0,
                    ),
                ),
            ]
        )
    else:
        logger.info("XGBoost not installed — skipping xgboost embedding classifier.")

    return models


if TORCH_AVAILABLE:

    class EmbeddingMLP(nn.Module):
        def __init__(self, input_dim: int, num_classes: int, hidden_dims: list[int], dropout: float) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            in_dim = input_dim
            for hidden_dim in hidden_dims:
                layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, num_classes))
            self.network = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)


def build_embedding_mlp(
    input_dim: int,
    num_classes: int,
    configs: dict[str, dict[str, Any]],
) -> Any:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for the embedding MLP classifier.")
    cfg = configs["models"]["sequence"]["mlp"]
    return EmbeddingMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=[int(d) for d in cfg["hidden_dims"]],
        dropout=float(cfg["dropout"]),
    )
