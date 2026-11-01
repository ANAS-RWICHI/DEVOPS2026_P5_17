"""Stage `evaluate`: score the model on the held-out period."""

from __future__ import annotations

import joblib
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.config import (
    METRICS_JSON,
    MODEL_PKL,
    PLOT_PNG,
    RUN_JSON,
    TARGET,
    TEST_CSV,
    load_params,
    read_json,
    tracking_uri,
    write_json,
)
from src.features import HORIZON, feature_columns

matplotlib.use("Agg")  # no display in Docker or CI


def compute_metrics(y_true, y_pred, y_naive) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    return {
        "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "wape": round(float(np.abs(y_true - y_pred).sum() / y_true.sum()), 4),
        "naive_rmse": round(float(root_mean_squared_error(y_true, y_naive)), 4),
        "n_test": int(len(y_true)),
    }


def plot_daily_totals(test: pd.DataFrame, y_pred: np.ndarray) -> None:
    daily = test.assign(pred=y_pred).groupby("date")[[TARGET, "pred"]].sum()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(pd.to_datetime(daily.index), daily[TARGET], label="actual")
    ax.plot(pd.to_datetime(daily.index), daily["pred"], label="predicted")
    ax.set_title("Total daily sales on the test period")
    ax.set_ylabel("units")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    PLOT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PNG, dpi=110)
    plt.close(fig)


def main() -> None:
    feat = load_params()["features"]
    columns = feature_columns(feat["lags"], feat["rolling_windows"])

    test = pd.read_csv(TEST_CSV)
    model = joblib.load(MODEL_PKL)
    y_pred = np.clip(model.predict(test[columns]), 0, None)

    metrics = compute_metrics(test[TARGET], y_pred, test[f"lag_{HORIZON}"])
    write_json(METRICS_JSON, metrics)
    plot_daily_totals(test, y_pred)

    mlflow.set_tracking_uri(tracking_uri())
    with mlflow.start_run(run_id=read_json(RUN_JSON)["run_id"]):
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items() if k != "n_test"})
        mlflow.log_artifact(str(PLOT_PNG))
    print(f"evaluate: {metrics}")


if __name__ == "__main__":
    main()
