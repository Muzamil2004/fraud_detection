"""Train fraud detection models and persist the best one."""

from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from src.preprocessing import build_preprocessor, load_data, split_data
except ModuleNotFoundError:
    from preprocessing import build_preprocessor, load_data, split_data


def _build_models(scale_pos_weight: float, random_state: int) -> Dict[str, object]:
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1200,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
        )
    except Exception:
        pass

    return models


def evaluate_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }


def train(
    data_path: str,
    output_model_path: str,
    target_col: str = "is_fraud",
    random_state: int = 42,
    use_smote: bool = True,
) -> Tuple[str, Dict[str, Dict[str, float]]]:
    df = load_data(data_path)
    split = split_data(
        df,
        target_col=target_col,
        drop_cols=["transaction_id"] if "transaction_id" in df.columns else [],
        test_size=0.2,
        random_state=random_state,
    )

    preprocessor, _, _ = build_preprocessor(split.X_train)

    pos = int(split.y_train.sum())
    neg = int(len(split.y_train) - pos)
    scale_pos_weight = (neg / max(pos, 1))

    models = _build_models(scale_pos_weight=scale_pos_weight, random_state=random_state)

    all_metrics: Dict[str, Dict[str, float]] = {}
    best_name = ""
    best_pipeline = None
    best_score = -1.0

    for name, clf in models.items():
        steps = [("preprocessor", preprocessor)]
        if use_smote:
            steps.append(("smote", SMOTE(random_state=random_state)))
        steps.append(("model", clf))

        pipeline = ImbPipeline(steps=steps)
        pipeline.fit(split.X_train, split.y_train)

        y_prob = pipeline.predict_proba(split.X_test)[:, 1]
        metrics = evaluate_at_threshold(split.y_test.to_numpy(), y_prob, threshold=0.5)
        all_metrics[name] = metrics

        if metrics["pr_auc"] > best_score:
            best_score = metrics["pr_auc"]
            best_name = name
            best_pipeline = pipeline

    artifact = {
        "model": best_pipeline,
        "model_name": best_name,
        "threshold": 0.5,
        "target_col": target_col,
        "metrics_at_0_5": all_metrics,
    }

    os.makedirs(os.path.dirname(output_model_path) or ".", exist_ok=True)
    joblib.dump(artifact, output_model_path)
    return best_name, all_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fraud detection models")
    parser.add_argument("--data", default="data/credit_card_fraud_10k.csv")
    parser.add_argument("--output", default="model.pkl")
    parser.add_argument("--target", default="is_fraud")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-smote", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    best_name, all_metrics = train(
        data_path=args.data,
        output_model_path=args.output,
        target_col=args.target,
        random_state=args.random_state,
        use_smote=not args.no_smote,
    )

    print(f"Best model: {best_name}")
    for name, metrics in all_metrics.items():
        print(f"\\n{name}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
