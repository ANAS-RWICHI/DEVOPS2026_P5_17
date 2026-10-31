"""Model factory. The naive baseline is the reference every model must beat."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import HORIZON


class NaiveLag(RegressorMixin, BaseEstimator):
    """Predict the sales observed one horizon ago (same weekday last week)."""

    def __init__(self, column: str = f"lag_{HORIZON}"):
        self.column = column

    def fit(self, X, y=None):
        if self.column not in X.columns:
            raise ValueError(f"NaiveLag needs the column {self.column!r}")
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return np.asarray(X[self.column], dtype=float)


def build_model(cfg: dict):
    """Build the estimator selected by `train.model` in params.yaml."""
    name, seed = cfg["model"], cfg["seed"]
    if name == "naive":
        return NaiveLag()
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(**cfg["ridge"]))
    if name == "random_forest":
        return RandomForestRegressor(**cfg["random_forest"], random_state=seed, n_jobs=-1)
    if name == "gbr":
        return HistGradientBoostingRegressor(**cfg["gbr"], random_state=seed)
    raise ValueError(f"unknown model {name!r}")
