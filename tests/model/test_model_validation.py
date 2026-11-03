"""Validation of the trained model. Run after `dvc repro`."""

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import METRICS_JSON, MODEL_PKL, TEST_CSV, load_params, read_json
from src.features import feature_columns

pytestmark = pytest.mark.skipif(
    not (MODEL_PKL.exists() and TEST_CSV.exists() and METRICS_JSON.exists()),
    reason="no trained model: run `dvc repro` first",
)

PARAMS = load_params()
COLUMNS = feature_columns(PARAMS["features"]["lags"], PARAMS["features"]["rolling_windows"])


@pytest.fixture(scope="module")
def model():
    return joblib.load(MODEL_PKL)


@pytest.fixture(scope="module")
def test_set():
    return pd.read_csv(TEST_CSV)


def test_rmse_is_under_the_promotion_threshold():
    assert read_json(METRICS_JSON)["rmse"] <= PARAMS["promote"]["max_rmse"]


def test_model_is_not_worse_than_the_naive_baseline():
    metrics = read_json(METRICS_JSON)
    assert metrics["rmse"] <= metrics["naive_rmse"]


def test_output_format(model, test_set):
    pred = model.predict(test_set[COLUMNS])
    assert pred.shape == (len(test_set),)
    assert np.isfinite(pred).all()


def test_edge_case_no_recent_sales(model, test_set):
    """A product that sold nothing recently must give a finite, small forecast."""
    row = test_set[COLUMNS].head(1).copy()
    row[[c for c in COLUMNS if c.startswith(("lag_", "rmean_"))]] = 0.0
    pred = float(model.predict(row)[0])
    assert np.isfinite(pred)
    assert pred < test_set["sales"].mean()

