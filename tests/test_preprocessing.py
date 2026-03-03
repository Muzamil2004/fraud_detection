from __future__ import annotations

import pandas as pd

from src.preprocessing import BehavioralFeatureEngineer, build_preprocessor


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "amount": [50.0, 750.0, 120.0],
            "transaction_hour": [2, 14, 23],
            "merchant_category": ["Travel", "Electronics", "Food"],
            "foreign_transaction": [1, 0, 1],
            "location_mismatch": [0, 1, 1],
            "device_trust_score": [20, 90, 40],
            "velocity_last_24h": [2, 12, 5],
            "cardholder_age": [45, 38, 29],
        }
    )


def test_behavioral_feature_engineering_adds_expected_columns() -> None:
    frame = _sample_frame()
    engineered = BehavioralFeatureEngineer().fit_transform(frame)

    expected_cols = {
        "transaction_hour_sin",
        "transaction_hour_cos",
        "is_night_transaction",
        "amount_log1p",
        "is_high_amount",
        "velocity_last_1h_proxy",
        "velocity_last_7d_proxy",
        "amount_per_velocity",
        "location_device_risk",
        "foreign_device_risk",
    }
    assert expected_cols.issubset(set(engineered.columns))
    assert engineered[list(expected_cols)].isna().sum().sum() == 0


def test_preprocessor_builds_with_engineered_frame() -> None:
    frame = BehavioralFeatureEngineer().fit_transform(_sample_frame())
    preprocessor, num_cols, cat_cols = build_preprocessor(frame)

    transformed = preprocessor.fit_transform(frame)
    assert transformed.shape[0] == len(frame)
    assert "merchant_category" in cat_cols
    assert "amount_log1p" in num_cols

