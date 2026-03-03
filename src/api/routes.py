"""API routes for fraud detection inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse

from src.api.runtime import get_runtime_loader, predict_probabilities
from src.api.schemas import BatchPredictResponse, FraudFeatures, PredictResponse


def build_router(model_path: Path, frontend_dir: Path) -> APIRouter:
    router = APIRouter()
    runtime_loader = get_runtime_loader(model_path)

    @router.get("/")
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @router.get("/health")
    def health() -> dict[str, Any]:
        runtime = runtime_loader()
        return {
            "status": "ok",
            "model": runtime["model_name"],
            "threshold": runtime["threshold"],
            "model_path": str(model_path),
        }

    @router.get("/ui", include_in_schema=False)
    def ui() -> FileResponse:
        index_path = frontend_dir / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Frontend not found.")
        return FileResponse(index_path)

    @router.get("/model-info")
    def model_info() -> dict[str, Any]:
        runtime = runtime_loader()
        artifact = runtime["artifact"]
        return {
            "model_name": runtime["model_name"],
            "threshold": runtime["threshold"],
            "target_col": artifact.get("target_col"),
            "available_metrics": artifact.get("metrics_at_0_5", {}),
        }

    @router.post("/predict", response_model=PredictResponse)
    def predict(
        payload: FraudFeatures,
        custom_threshold: float | None = Query(default=None, ge=0, le=1),
    ) -> PredictResponse:
        try:
            runtime = runtime_loader()
            prob = float(predict_probabilities(runtime=runtime, payloads=[payload])[0])
            threshold = runtime["threshold"] if custom_threshold is None else custom_threshold
            return PredictResponse(
                fraud_probability=round(prob, 6),
                is_fraud=int(prob >= threshold),
                threshold_used=round(float(threshold), 6),
                model=runtime["model_name"],
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    @router.post("/predict/batch", response_model=BatchPredictResponse)
    def predict_batch(
        payloads: list[FraudFeatures],
        custom_threshold: float | None = Query(default=None, ge=0, le=1),
    ) -> BatchPredictResponse:
        if not payloads:
            raise HTTPException(status_code=400, detail="Payload list cannot be empty.")
        try:
            runtime = runtime_loader()
            probs = predict_probabilities(runtime=runtime, payloads=payloads)
            threshold = runtime["threshold"] if custom_threshold is None else custom_threshold
            predictions = [
                PredictResponse(
                    fraud_probability=round(float(prob), 6),
                    is_fraud=int(prob >= threshold),
                    threshold_used=round(float(threshold), 6),
                    model=runtime["model_name"],
                )
                for prob in probs
            ]
            return BatchPredictResponse(count=len(predictions), predictions=predictions)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}") from exc

    @router.post("/reload-model")
    def reload_model() -> dict[str, Any]:
        runtime_loader.cache_clear()
        runtime = runtime_loader()
        return {
            "status": "reloaded",
            "model": runtime["model_name"],
            "threshold": runtime["threshold"],
        }

    return router

