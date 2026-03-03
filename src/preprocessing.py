"""Data loading and preprocessing utilities for fraud detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


class BehavioralFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create temporal and behavioral features from base transaction fields."""

    def __init__(self) -> None:
        self._high_amount_threshold: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "BehavioralFeatureEngineer":
        if "amount" in X.columns:
            self._high_amount_threshold = float(X["amount"].quantile(0.95))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()

        if "transaction_hour" in out.columns:
            hour = out["transaction_hour"].astype(float)
            out["transaction_hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
            out["transaction_hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
            out["is_night_transaction"] = ((hour <= 5) | (hour >= 22)).astype(int)

        if "amount" in out.columns:
            amount = out["amount"].astype(float)
            out["amount_log1p"] = np.log1p(np.clip(amount, a_min=0.0, a_max=None))
            out["is_high_amount"] = (amount >= self._high_amount_threshold).astype(int)
        else:
            amount = pd.Series(0.0, index=out.index)

        if "velocity_last_24h" in out.columns:
            velocity_24h = out["velocity_last_24h"].astype(float)
            out["velocity_last_1h_proxy"] = velocity_24h / 24.0
            out["velocity_last_7d_proxy"] = velocity_24h * 7.0
            out["amount_per_velocity"] = amount / (velocity_24h + 1.0)
        else:
            out["velocity_last_1h_proxy"] = 0.0
            out["velocity_last_7d_proxy"] = 0.0
            out["amount_per_velocity"] = amount

        if {"device_trust_score", "location_mismatch"}.issubset(out.columns):
            trust = out["device_trust_score"].astype(float) / 100.0
            mismatch = out["location_mismatch"].astype(float)
            out["location_device_risk"] = mismatch * (1.0 - trust)

        if {"device_trust_score", "foreign_transaction"}.issubset(out.columns):
            trust = out["device_trust_score"].astype(float) / 100.0
            foreign = out["foreign_transaction"].astype(float)
            out["foreign_device_risk"] = foreign * (1.0 - trust)

        return out


def load_data(path: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    return pd.read_csv(path)


def split_data(
    df: pd.DataFrame,
    target_col: str = "is_fraud",
    drop_cols: List[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplit:
    """Split dataset with stratification and optional column drops."""
    drop_cols = drop_cols or []
    X = df.drop(columns=[target_col] + drop_cols)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    return DatasetSplit(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Create preprocessing pipeline with scaling and one-hot encoding."""
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_pipe = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipe = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ]
    )

    return preprocessor, num_cols, cat_cols
