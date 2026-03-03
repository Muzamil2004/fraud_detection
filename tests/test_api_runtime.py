from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from src.api.runtime import get_runtime_loader, load_artifact, predict_probabilities
from src.api.schemas import FraudFeatures


class DummyModel:
    def predict_proba(self, x):
        probs = np.full((len(x), 2), 0.0)
        probs[:, 0] = 0.2
        probs[:, 1] = 0.8
        return probs


def _payload() -> FraudFeatures:
    return FraudFeatures(
        amount=120.5,
        transaction_hour=13,
        merchant_category="Travel",
        foreign_transaction=1,
        location_mismatch=0,
        device_trust_score=75,
        velocity_last_24h=3,
        cardholder_age=35,
    )


def test_runtime_loader_and_prediction(tmp_path: Path) -> None:
    model_path = tmp_path / "model.pkl"
    artifact = {"model": DummyModel(), "model_name": "dummy", "threshold": 0.4}
    joblib.dump(artifact, model_path)

    loaded = load_artifact(model_path)
    assert loaded["model_name"] == "dummy"

    runtime_loader = get_runtime_loader(model_path)
    runtime = runtime_loader()
    probs = predict_probabilities(runtime=runtime, payloads=[_payload()])
    assert probs.shape == (1,)
    assert float(probs[0]) == 0.8

    runtime_loader.cache_clear()
    runtime_reloaded = runtime_loader()
    assert runtime_reloaded["threshold"] == 0.4

