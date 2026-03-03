"""Runtime model loading and prediction utilities."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.api.schemas import FraudFeatures


def load_artifact(model_path: Path) -> dict[str, Any]:
    if not model_path.exists():
        raise RuntimeError(
            "model.pkl not found. Train first: `python -m src.train --data data/credit_card_fraud_10k.csv --output model.pkl`"
        )
    artifact = joblib.load(model_path)
    if "model" not in artifact:
        raise RuntimeError("Invalid model artifact: missing `model` key.")
    return artifact


def build_runtime(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact": artifact,
        "model": artifact["model"],
        "threshold": float(artifact.get("threshold", 0.5)),
        "model_name": str(artifact.get("model_name", "unknown")),
    }


def get_runtime_loader(model_path: Path):
    resolved_path = model_path.resolve()

    @lru_cache(maxsize=1)
    def _cached() -> dict[str, Any]:
        artifact = load_artifact(resolved_path)
        return build_runtime(artifact)

    return _cached


def predict_probabilities(runtime: dict[str, Any], payloads: list[FraudFeatures]) -> np.ndarray:
    x = pd.DataFrame([p.model_dump() for p in payloads])
    return runtime["model"].predict_proba(x)[:, 1]
