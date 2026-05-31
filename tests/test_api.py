"""Integration tests for the FastAPI prediction endpoints (mock predictor)."""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from pos_classifier.api.app import create_app
from pos_classifier.config import TrainingConfig
from pos_classifier.serving.predictor import Predictor
from pos_classifier.serving.schema import PredictResponse


def _make_mock_predictor(category: str = "Beverages", confidence: float = 0.95) -> MagicMock:
    pred = MagicMock(spec=Predictor)
    pred.model_version = "test-v1"
    pred.confidence_threshold = 0.70
    pred.predict.return_value = [
        PredictResponse(
            product_description="test item",
            category=category,
            confidence=confidence,
            flagged_for_human_review=False,
            model_version="test-v1",
            predicted_at=datetime.now(timezone.utc),
        )
    ]

    def mock_record_feedback(description: str, corrected: str, original: str | None) -> None:
        from pos_classifier.config import LABEL_MAP
        if corrected not in LABEL_MAP:
            raise ValueError(f"Invalid category '{corrected}'. Valid: {list(LABEL_MAP.keys())}")

    pred.record_feedback.side_effect = mock_record_feedback
    return pred


@pytest.fixture()
def client():
    app = create_app(TrainingConfig())
    with TestClient(app) as c:
        with patch.object(Predictor, "_instance", _make_mock_predictor()):
            yield c


class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


class TestPredict:
    def test_valid_request(self, client):
        resp = client.post("/predict", json={"product_description": "Sparkling Water"})
        assert resp.status_code == 200
        data = resp.json()
        assert "category" in data
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0
        assert "flagged_for_human_review" in data

    def test_empty_description_rejected(self, client):
        resp = client.post("/predict", json={"product_description": ""})
        assert resp.status_code == 422

    def test_missing_field_rejected(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422


class TestBatchPredict:
    def test_valid_batch(self, client):
        with patch.object(
            Predictor._instance,  # type: ignore[union-attr]
            "predict",
            return_value=[
                PredictResponse(
                    product_description=f"item{i}",
                    category="Beverages",
                    confidence=0.9,
                    flagged_for_human_review=False,
                    model_version="test-v1",
                    predicted_at=datetime.now(timezone.utc),
                )
                for i in range(3)
            ],
        ):
            resp = client.post(
                "/predict/batch",
                json={"items": [{"product_description": f"item{i}"} for i in range(3)]},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
        assert len(data["results"]) == 3

    def test_empty_batch_rejected(self, client):
        resp = client.post("/predict/batch", json={"items": []})
        assert resp.status_code == 422


class TestFeedback:
    def test_valid_feedback(self, client):
        resp = client.post(
            "/feedback",
            json={
                "product_description": "Sparkling Water",
                "corrected_category": "Beverages",
            },
        )
        assert resp.status_code == 204

    def test_unknown_category_rejected(self, client):
        resp = client.post(
            "/feedback",
            json={
                "product_description": "Sparkling Water",
                "corrected_category": "INVALID_CATEGORY",
            },
        )
        assert resp.status_code == 422


class TestMetrics:
    def test_prometheus_endpoint(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert b"pos_classifier" in resp.content


class TestContract:
    def test_openapi_contract_export(self, client):
        resp = client.get("/contract")
        assert resp.status_code == 200
        schema = resp.json()
        assert schema["openapi"] == "3.1.0"
        assert "paths" in schema
        assert "/predict" in schema["paths"]
        assert "/feedback" in schema["paths"]
        assert schema["info"]["title"] == "POS Product Classifier"
