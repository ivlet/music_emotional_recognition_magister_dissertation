"""Classical machine learning models for static emotion classification."""

from __future__ import annotations

import logging
from typing import Any

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def _static_preprocess_steps() -> list[tuple[str, Any]]:
    return [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]


def build_static_models(
    configs: dict[str, dict[str, Any]],
    random_state: int,
) -> dict[str, Pipeline]:
    """Build sklearn pipelines (scaler + classifier) for static baselines."""
    cfg = configs["models"]["static"]
    models: dict[str, Pipeline] = {}

    models["logistic_regression"] = Pipeline(
        [
            *_static_preprocess_steps(),
            (
                "clf",
                LogisticRegression(
                    C=float(cfg["logistic_regression"]["C"]),
                    max_iter=int(cfg["logistic_regression"]["max_iter"]),
                    random_state=random_state,
                    multi_class="auto",
                ),
            ),
        ]
    )

    models["knn"] = Pipeline(
        [
            *_static_preprocess_steps(),
            (
                "clf",
                KNeighborsClassifier(
                    n_neighbors=int(cfg["knn"]["n_neighbors"]),
                    weights=str(cfg["knn"]["weights"]),
                ),
            ),
        ]
    )

    models["svm"] = Pipeline(
        [
            *_static_preprocess_steps(),
            (
                "clf",
                SVC(
                    C=float(cfg["svm"]["C"]),
                    kernel=str(cfg["svm"]["kernel"]),
                    gamma=str(cfg["svm"]["gamma"]),
                    random_state=random_state,
                ),
            ),
        ]
    )

    rf_params: dict[str, Any] = {
        "n_estimators": int(cfg["random_forest"]["n_estimators"]),
        "random_state": random_state,
    }
    if cfg["random_forest"].get("max_depth") is not None:
        rf_params["max_depth"] = int(cfg["random_forest"]["max_depth"])
    models["random_forest"] = Pipeline(
        [
            *_static_preprocess_steps(),
            ("clf", RandomForestClassifier(**rf_params)),
        ]
    )

    models["gradient_boosting"] = Pipeline(
        [
            *_static_preprocess_steps(),
            (
                "clf",
                GradientBoostingClassifier(
                    n_estimators=int(cfg["gradient_boosting"]["n_estimators"]),
                    learning_rate=float(cfg["gradient_boosting"]["learning_rate"]),
                    max_depth=int(cfg["gradient_boosting"]["max_depth"]),
                    random_state=random_state,
                ),
            ),
        ]
    )

    if XGBOOST_AVAILABLE:
        models["xgboost"] = Pipeline(
            [
                *_static_preprocess_steps(),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=int(cfg["xgboost"]["n_estimators"]),
                        learning_rate=float(cfg["xgboost"]["learning_rate"]),
                        max_depth=int(cfg["xgboost"]["max_depth"]),
                        random_state=random_state,
                        eval_metric="mlogloss",
                        verbosity=0,
                    ),
                ),
            ]
        )
    else:
        logger.info("XGBoost not installed — skipping xgboost model.")

    mlp_cfg = cfg["mlp"]
    models["mlp"] = Pipeline(
        [
            *_static_preprocess_steps(),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=tuple(int(s) for s in mlp_cfg["hidden_layer_sizes"]),
                    activation=str(mlp_cfg["activation"]),
                    solver=str(mlp_cfg["solver"]),
                    alpha=float(mlp_cfg["alpha"]),
                    learning_rate_init=float(mlp_cfg["learning_rate_init"]),
                    max_iter=int(mlp_cfg["max_iter"]),
                    early_stopping=bool(mlp_cfg["early_stopping"]),
                    validation_fraction=float(mlp_cfg["validation_fraction"]),
                    n_iter_no_change=int(mlp_cfg["n_iter_no_change"]),
                    random_state=random_state,
                ),
            ),
        ]
    )

    return models
