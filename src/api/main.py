"""Demand forecasting API: serves the model holding the `production` alias."""

from __future__ import annotations

import datetime as dt
import logging
import os
from contextlib import asynccontextmanager

import joblib
import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import MODEL_PKL, PRODUCTION_ALIAS, load_params, tracking_uri
from src.features import add_calendar, feature_columns, features_from_history, history_length

log = logging.getLogger("api")

PARAMS = load_params()
LAGS = PARAMS["features"]["lags"]
WINDOWS = PARAMS["features"]["rolling_windows"]
COLUMNS = feature_columns(LAGS, WINDOWS)
MODEL_NAME = os.getenv("MODEL_NAME", PARAMS["mlflow"]["model_name"])



def load_model():
    """Return (model, source). Registry first, local pipeline output as a fallback."""
    uri = f"models:/{MODEL_NAME}@{PRODUCTION_ALIAS}"
    try:
        mlflow.set_tracking_uri(tracking_uri())
        return mlflow.sklearn.load_model(uri), uri
    except Exception as exc:
        log.warning("could not load %s from MLflow: %s", uri, exc)
    if MODEL_PKL.exists():
        return joblib.load(MODEL_PKL), f"file:{MODEL_PKL.name}"
    return None, "none"


def refresh(app: FastAPI) -> None:
    app.state.model, app.state.model_source = load_model()


@asynccontextmanager
async def lifespan(app: FastAPI):
    refresh(app)
    yield


app = FastAPI(
    title="Demand forecast API",
    description="Forecasts the units sold for a store and an item, 7 days ahead.",
    version="0.1.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    store: int = Field(ge=1, examples=[1])
    item: int = Field(ge=1, examples=[3])
    date: dt.date = Field(description="Day to forecast", examples=["2018-01-15"])
    history: list[float] = Field(
        description=(
            f"Daily sales of this store and item, oldest first, ending 7 days before `date`. "
            f"At least {history_length(LAGS, WINDOWS)} values."
        ),
        examples=[[30.0] * history_length(LAGS, WINDOWS)],
    )


class PredictResponse(BaseModel):
    prediction: float
    model_source: str


@app.get("/health", summary="Liveness probe")
def health():
    return {"status": "ok"}


@app.get("/ready", summary="Readiness probe: is a model loaded?")
def ready():
    if app.state.model is None:
        raise HTTPException(503, "no model loaded")
    return {"status": "ready", "model_source": app.state.model_source}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if app.state.model is None:
        raise HTTPException(503, "no model loaded")
    if any(not np.isfinite(v) or v < 0 for v in req.history):
        raise HTTPException(422, "history values must be finite and >= 0")
    try:
        lag_features = features_from_history(req.history, LAGS, WINDOWS)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    row = pd.DataFrame(
        [
            {
                "store": req.store,
                "item": req.item,
                "date": req.date.isoformat(),
                **lag_features,
            }
        ]
    )
    row = add_calendar(row)
    prediction = max(0.0, float(app.state.model.predict(row[COLUMNS])[0]))
    return PredictResponse(prediction=round(prediction, 2), model_source=app.state.model_source)
