"""Pydantic schemas for API requests/responses."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class FraudFeatures(BaseModel):
    amount: float = Field(..., gt=0)
    transaction_hour: int = Field(..., ge=0, le=23)
    merchant_category: str
    foreign_transaction: int = Field(..., ge=0, le=1)
    location_mismatch: int = Field(..., ge=0, le=1)
    device_trust_score: float = Field(..., ge=0, le=100)
    velocity_last_24h: int = Field(..., ge=0)
    cardholder_age: int = Field(..., ge=18, le=120)


class PredictResponse(BaseModel):
    fraud_probability: float
    is_fraud: int
    threshold_used: float
    model: str


class BatchPredictResponse(BaseModel):
    count: int
    predictions: List[PredictResponse]

