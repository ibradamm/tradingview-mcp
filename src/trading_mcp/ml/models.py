"""Model registry. Add a model by registering a factory; nothing else needs to change.

Example (optional dependency)::

    from xgboost import XGBClassifier
    register_model("xgboost", "classification", lambda: XGBClassifier(n_estimators=300, max_depth=3))
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

Factory = Callable[[], Any]
_REGISTRY: dict[tuple[str, str], Factory] = {}


def register_model(name: str, task: str, factory: Factory) -> None:
    _REGISTRY[(name, task)] = factory


def available_models() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name, task in _REGISTRY:
        out.setdefault(name, []).append(task)
    return out


def create_model(name: str, task: str) -> Pipeline:
    """Return an unfitted pipeline (scaler fitted on training data only, inside the pipeline)."""
    try:
        estimator = _REGISTRY[(name, task)]()
    except KeyError as exc:
        raise ValueError(f"Model '{name}' is not available for {task}. Available: {available_models()}") from exc
    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


# Conservative defaults: shallow trees and strong regularisation limit overfitting on noisy returns.
# Sizes are also chosen so a full walk-forward fits in a few seconds on a small (1-2 vCPU) server.
_RF = {"n_estimators": 100, "max_depth": 4, "min_samples_leaf": 50, "max_samples": 0.5, "n_jobs": -1,
       "random_state": 42}
_GB = {"max_iter": 150, "max_depth": 3, "learning_rate": 0.05, "min_samples_leaf": 50, "l2_regularization": 1.0,
       "early_stopping": False, "random_state": 42}

register_model("logistic_regression", "classification", lambda: LogisticRegression(C=0.1, max_iter=2000))
register_model("random_forest", "classification", lambda: RandomForestClassifier(**_RF))
register_model("gradient_boosting", "classification", lambda: HistGradientBoostingClassifier(**_GB))
register_model("logistic_regression", "regression", lambda: Ridge(alpha=10.0))  # linear counterpart
register_model("ridge", "regression", lambda: Ridge(alpha=10.0))
register_model("random_forest", "regression", lambda: RandomForestRegressor(**_RF))
register_model("gradient_boosting", "regression", lambda: HistGradientBoostingRegressor(**_GB))
