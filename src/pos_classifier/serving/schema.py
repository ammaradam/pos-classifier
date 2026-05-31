"""Pydantic request/response models and schema validation for the serving API."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    product_description: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Raw POS product description text.",
        examples=["Chocolate Sandwich Cookies"],
    )


class PredictResponse(BaseModel):
    product_description: str
    category: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    flagged_for_human_review: bool
    model_version: str
    predicted_at: datetime


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest] = Field(..., min_length=1, max_length=512)


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    total: int


class FeedbackRequest(BaseModel):
    product_description: str = Field(..., min_length=1, max_length=500)
    corrected_category: str = Field(
        ...,
        description="Human-verified category label.",
    )
    original_prediction: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_loaded: bool
