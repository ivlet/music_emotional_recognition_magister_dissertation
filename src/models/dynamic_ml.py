"""Classical ML models for dynamic window-level emotion prediction."""

from __future__ import annotations

import logging
from typing import Any

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def _preprocess_steps(with_scaler: bool = True) -> list[tuple[str, Any]]:
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if with_scaler:
        steps.append(("scaler", StandardScaler()))
    return steps


def build_dynamic_classification_models(
    configs: dict[str, dict[str, Any]],
    random_state: int,
) -> dict[str, Pipeline]:
    """Build sklearn pipelines for dynamic window emotion quadrant classification."""
    cfg = configs["models"]["dynamic_window"]
    models: dict[str, Pipeline] = {}

    models["logistic_regression"] = Pipeline(
        [
            *_preprocess_steps(with_scaler=True),
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

    models["svm"] = Pipeline(
        [
            *_preprocess_steps(with_scaler=True),
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
            *_preprocess_steps(with_scaler=True),
            ("clf", RandomForestClassifier(**rf_params)),
        ]
    )

    models["gradient_boosting"] = Pipeline(
        [
            *_preprocess_steps(with_scaler=True),
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
                *_preprocess_steps(with_scaler=True),
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
        logger.info("XGBoost not installed — skipping xgboost classifier.")

    return models


def build_dynamic_regression_models(
    configs: dict[str, dict[str, Any]],
    random_state: int,
) -> dict[str, Pipeline]:
    """Build sklearn pipelines for dynamic window valence/arousal regression."""
    cfg = configs["models"]["dynamic_window"]
    models: dict[str, Pipeline] = {}

    models["ridge"] = Pipeline(
        [
            *_preprocess_steps(with_scaler=True),
            ("reg", Ridge(alpha=float(cfg["ridge"]["alpha"]))),
        ]
    )

    rf_params: dict[str, Any] = {
        "n_estimators": int(cfg["random_forest_regressor"]["n_estimators"]),
        "random_state": random_state,
    }
    if cfg["random_forest_regressor"].get("max_depth") is not None:
        rf_params["max_depth"] = int(cfg["random_forest_regressor"]["max_depth"])
    models["random_forest_regressor"] = Pipeline(
        [
            *_preprocess_steps(with_scaler=False),
            ("reg", RandomForestRegressor(**rf_params)),
        ]
    )

    models["gradient_boosting_regressor"] = Pipeline(
        [
            *_preprocess_steps(with_scaler=False),
            (
                "reg",
                GradientBoostingRegressor(
                    n_estimators=int(cfg["gradient_boosting_regressor"]["n_estimators"]),
                    learning_rate=float(cfg["gradient_boosting_regressor"]["learning_rate"]),
                    max_depth=int(cfg["gradient_boosting_regressor"]["max_depth"]),
                    random_state=random_state,
                ),
            ),
        ]
    )

    return models
