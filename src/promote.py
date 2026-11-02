"""Stage `promote`: move the `production` alias to the candidate if it earns it.

Rule (params.yaml > promote):
  1. test RMSE must be <= max_rmse;
  2. if a production model exists, the candidate must beat its test RMSE
     by more than min_improvement.
Rolling back is a single alias move to an older version in the registry.
"""

from __future__ import annotations

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.config import (
    METRICS_JSON,
    PRODUCTION_ALIAS,
    PROMOTION_JSON,
    RUN_JSON,
    load_params,
    read_json,
    tracking_uri,
    write_json,
)


def decide(
    new_rmse: float, prod_rmse: float | None, max_rmse: float, min_improvement: float
) -> tuple[bool, str]:
    if new_rmse > max_rmse:
        return False, f"rmse {new_rmse:.4f} is above the threshold {max_rmse}"
    if prod_rmse is None:
        return True, "no production model yet and the candidate is under the threshold"
    if new_rmse < prod_rmse - min_improvement:
        return True, f"rmse {new_rmse:.4f} beats production {prod_rmse:.4f}"
    return False, f"rmse {new_rmse:.4f} does not beat production {prod_rmse:.4f}"


def production_rmse(client: MlflowClient, model_name: str) -> tuple[str | None, float | None]:
    try:
        current = client.get_model_version_by_alias(model_name, PRODUCTION_ALIAS)
    except MlflowException:
        return None, None
    return current.version, client.get_run(current.run_id).data.metrics.get("test_rmse")


def main() -> None:
    rule = load_params()["promote"]
    run = read_json(RUN_JSON)
    new_rmse = read_json(METRICS_JSON)["rmse"]

    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    prod_version, prod_rmse = production_rmse(client, run["model_name"])

    promoted, reason = decide(new_rmse, prod_rmse, rule["max_rmse"], rule["min_improvement"])
    if promoted:
        client.set_registered_model_alias(
            run["model_name"], PRODUCTION_ALIAS, run["model_version"]
        )
    client.set_model_version_tag(
        run["model_name"], run["model_version"], "promotion", "promoted" if promoted else "rejected"
    )

    write_json(
        PROMOTION_JSON,
        {
            "promoted": promoted,
            "reason": reason,
            "candidate_version": run["model_version"],
            "candidate_rmse": new_rmse,
            "previous_production_version": prod_version,
            "previous_production_rmse": prod_rmse,
        },
    )
    print(f"promote: v{run['model_version']} promoted={promoted} ({reason})")


if __name__ == "__main__":
    main()
