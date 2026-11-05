import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import main


class FakeModel:
    def __init__(self, value: float):
        self.value = value

    def predict(self, X):
        assert list(X.columns) == main.COLUMNS
        return np.full(len(X), self.value)


def make_client(monkeypatch, model):
    monkeypatch.setattr(main, "load_model", lambda: (model, "test" if model else "none"))
    return TestClient(main.app)


PAYLOAD = {
    "store": 1,
    "item": 3,
    "date": "2026-01-15",
    "history": [30.0] * 28,
}


def test_health_and_ready(monkeypatch):
    with make_client(monkeypatch, FakeModel(42.0)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/ready").status_code == 200


def test_predict_returns_a_number(monkeypatch):
    with make_client(monkeypatch, FakeModel(42.0)) as client:
        response = client.post("/predict", json=PAYLOAD)
    assert response.status_code == 200
    assert response.json() == {"prediction": 42.0, "model_source": "test"}


def test_negative_predictions_are_clipped_to_zero(monkeypatch):
    with make_client(monkeypatch, FakeModel(-5.0)) as client:
        assert client.post("/predict", json=PAYLOAD).json()["prediction"] == 0.0


@pytest.mark.parametrize(
    "change",
    [
        {"history": [30.0] * 5},  # too short
        {"history": [-1.0] * 28},  # negative sales
        {"item": -1},
        {"store": 0},
        {"date": "not-a-date"},
    ],
)
def test_invalid_input_is_rejected(monkeypatch, change):
    with make_client(monkeypatch, FakeModel(42.0)) as client:
        assert client.post("/predict", json={**PAYLOAD, **change}).status_code == 422


def test_without_model_the_api_is_alive_but_not_ready(monkeypatch):
    with make_client(monkeypatch, None) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/predict", json=PAYLOAD).status_code == 503


def test_metrics_endpoint_exposes_prometheus_counters(monkeypatch):
    with make_client(monkeypatch, FakeModel(42.0)) as client:
        client.post("/predict", json=PAYLOAD)
        body = client.get("/metrics").text
    assert 'api_requests_total{path="/predict",status="200"}' in body
    assert "model_prediction_units_count" in body
    assert "model_loaded 1.0" in body
