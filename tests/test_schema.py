"""Tests for Pydantic request/response schema validation."""

import pytest
from pydantic import ValidationError

from pos_classifier.serving.schema import (
    BatchPredictRequest,
    FeedbackRequest,
    PredictRequest,
)


class TestPredictRequest:
    def test_valid(self):
        req = PredictRequest(product_description="Chocolate Sandwich Cookies")
        assert req.product_description == "Chocolate Sandwich Cookies"

    def test_empty_string_rejected(self):
        with pytest.raises(ValidationError):
            PredictRequest(product_description="")

    def test_too_long_rejected(self):
        with pytest.raises(ValidationError):
            PredictRequest(product_description="x" * 501)

    def test_max_length_accepted(self):
        req = PredictRequest(product_description="x" * 500)
        assert len(req.product_description) == 500


class TestBatchPredictRequest:
    def test_valid_batch(self):
        req = BatchPredictRequest(
            items=[
                PredictRequest(product_description="Product A"),
                PredictRequest(product_description="Product B"),
            ]
        )
        assert len(req.items) == 2

    def test_empty_batch_rejected(self):
        with pytest.raises(ValidationError):
            BatchPredictRequest(items=[])

    def test_over_512_rejected(self):
        with pytest.raises(ValidationError):
            BatchPredictRequest(
                items=[PredictRequest(product_description=f"P{i}") for i in range(513)]
            )


class TestFeedbackRequest:
    def test_valid(self):
        req = FeedbackRequest(
            product_description="Chocolate Sandwich Cookies",
            corrected_category="Dry Goods & Pantry Staples",
        )
        assert req.corrected_category == "Dry Goods & Pantry Staples"

    def test_empty_description_rejected(self):
        with pytest.raises(ValidationError):
            FeedbackRequest(product_description="", corrected_category="Beverages")
