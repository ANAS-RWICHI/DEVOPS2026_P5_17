"""Shared paths, parameters and MLflow settings for every pipeline stage."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

RAW_CSV = ROOT / "data" / "raw" / "train.csv"  # Kaggle file, tracked by DVC
CLEAN_CSV = ROOT / "data" / "processed" / "sales_clean.csv"
TRAIN_CSV = ROOT / "data" / "features" / "train.csv"
TEST_CSV = ROOT / "data" / "features" / "test.csv"
MODEL_PKL = ROOT / "models" / "model.pkl"
RUN_JSON = ROOT / "models" / "run.json"
METRICS_JSON = ROOT / "metrics" / "metrics.json"
PLOT_PNG = ROOT / "metrics" / "pred_vs_actual.png"
PROMOTION_JSON = ROOT / "metrics" / "promotion.json"

PRODUCTION_ALIAS = "production"
TARGET = "sales"


def load_params(path: Path = ROOT / "params.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": identical bytes (and DVC hashes) on Windows and Linux
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
