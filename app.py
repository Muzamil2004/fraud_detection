"""FastAPI app for fraud detection inference."""
# pyright: reportMissingImports=false
# pylint: disable=import-error

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, List

import joblib
import numpy as np
import pandas as pd
try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import FileResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
except ModuleNotFoundError as exc:
    if __name__ == "__main__":
        raise SystemExit(
            "Missing API dependencies in current interpreter.\n"
            "Use project env:\n"
            "  .\\venv\\Scripts\\uvicorn.exe app:app --reload\n"
            "or install packages in this interpreter:\n"
            "  python -m pip install -r requirements.txt"
        ) from exc
    raise

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model.pkl"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Predicts fraud probability and decision with tuned threshold.",
    version="1.1.0",
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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


def _load_artifact() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise RuntimeError(
            "model.pkl not found. Train first: `python -m src.train --data data/credit_card_fraud_10k.csv --output model.pkl`"
        )
    artifact = joblib.load(MODEL_PATH)
    if "model" not in artifact:
        raise RuntimeError("Invalid model artifact: missing `model` key.")
    return artifact


@lru_cache(maxsize=1)
def _get_runtime() -> dict[str, Any]:
    artifact = _load_artifact()
    return {
        "artifact": artifact,
        "model": artifact["model"],
        "threshold": float(artifact.get("threshold", 0.5)),
        "model_name": str(artifact.get("model_name", "unknown")),
    }


def _predict_probabilities(payloads: List[FraudFeatures]) -> np.ndarray:
    runtime = _get_runtime()
    x = pd.DataFrame([p.model_dump() for p in payloads])
    return runtime["model"].predict_proba(x)[:, 1]


@app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict[str, Any]:
    runtime = _get_runtime()
    return {
        "status": "ok",
        "model": runtime["model_name"],
        "threshold": runtime["threshold"],
        "model_path": str(MODEL_PATH),
    }


@app.get("/ui", include_in_schema=False)
def ui() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(index_path)


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    runtime = _get_runtime()
    artifact = runtime["artifact"]
    return {
        "model_name": runtime["model_name"],
        "threshold": runtime["threshold"],
        "target_col": artifact.get("target_col"),
        "available_metrics": artifact.get("metrics_at_0_5", {}),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: FraudFeatures, custom_threshold: float | None = Query(default=None, ge=0, le=1)) -> PredictResponse:
    try:
        runtime = _get_runtime()
        prob = float(_predict_probabilities([payload])[0])
        t = runtime["threshold"] if custom_threshold is None else custom_threshold
        pred = int(prob >= t)
        return PredictResponse(
            fraud_probability=round(prob, 6),
            is_fraud=pred,
            threshold_used=round(float(t), 6),
            model=runtime["model_name"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(
    payloads: List[FraudFeatures],
    custom_threshold: float | None = Query(default=None, ge=0, le=1),
) -> BatchPredictResponse:
    if not payloads:
        raise HTTPException(status_code=400, detail="Payload list cannot be empty.")

    try:
        runtime = _get_runtime()
        probs = _predict_probabilities(payloads)
        t = runtime["threshold"] if custom_threshold is None else custom_threshold

        preds = [
            PredictResponse(
                fraud_probability=round(float(prob), 6),
                is_fraud=int(prob >= t),
                threshold_used=round(float(t), 6),
                model=runtime["model_name"],
            )
            for prob in probs
        ]
        return BatchPredictResponse(count=len(preds), predictions=preds)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}") from exc


@app.post("/reload-model")
def reload_model() -> dict[str, Any]:
    _get_runtime.cache_clear()
    runtime = _get_runtime()
    return {
        "status": "reloaded",
        "model": runtime["model_name"],
        "threshold": runtime["threshold"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
