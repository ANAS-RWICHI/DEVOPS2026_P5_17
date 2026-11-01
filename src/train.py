"""Stage `train`: fit the model and log the run to MLflow."""

from __future__ import annotations

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import root_mean_squared_error

from src.config import (
    MODEL_PKL,
    ROOT,
    RUN_JSON,
    TARGET,
    TRAIN_CSV,
    file_md5,
    git_commit,
    load_params,
    tracking_uri,
    write_json,
)
from src.features import feature_columns
from src.models import build_model


def main() -> None:
    params = load_params()
    cfg, feat, ml = params["train"], params["features"], params["mlflow"]
    columns = feature_columns(feat["lags"], feat["rolling_windows"])

    train = pd.read_csv(TRAIN_CSV)
    X, y = train[columns], train[TARGET]

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(ml["experiment"])

    with mlflow.start_run(run_name=cfg["model"]) as run:
        mlflow.set_tags({"git_commit": git_commit(), "train_data_md5": file_md5(TRAIN_CSV)})
        mlflow.log_params({"model": cfg["model"], "seed": cfg["seed"]})
        mlflow.log_params({f"{cfg['model']}.{k}": v for k, v in cfg.get(cfg["model"], {}).items()})
        mlflow.log_params({f"features.{k}": v for k, v in feat.items()})
        mlflow.log_artifact(str(ROOT / "params.yaml"))

        model = build_model(cfg).fit(X, y)
        train_rmse = float(root_mean_squared_error(y, np.clip(model.predict(X), 0, None)))
        mlflow.log_metric("train_rmse", train_rmse)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=ml["model_name"],
            input_example=X.head(2),
        )

    versions = MlflowClient().search_model_versions(f"run_id='{run.info.run_id}'")
    version = versions[0].version

    MODEL_PKL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PKL)
    write_json(
        RUN_JSON,
        {"run_id": run.info.run_id, "model_name": ml["model_name"], "model_version": version},
    )
    print(f"train: {cfg['model']} train_rmse={train_rmse:.3f} run={run.info.run_id} v{version}")


if __name__ == "__main__":
    main()
