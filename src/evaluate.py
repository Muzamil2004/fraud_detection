"""Evaluate trained fraud model at configurable threshold."""

from __future__ import annotations

import argparse

import joblib
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from src.preprocessing import load_data, split_data
except ModuleNotFoundError:
    from preprocessing import load_data, split_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fraud model")
    parser.add_argument("--model", default="model.pkl")
    parser.add_argument("--data", default="data/credit_card_fraud_10k.csv")
    parser.add_argument("--target", default="is_fraud")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    artifact = joblib.load(args.model)
    model = artifact["model"]
    threshold = args.threshold if args.threshold is not None else artifact.get("threshold", 0.5)

    df = load_data(args.data)
    split = split_data(
        df,
        target_col=args.target,
        drop_cols=["transaction_id"] if "transaction_id" in df.columns else [],
        test_size=0.2,
        random_state=args.random_state,
    )

    y_prob = model.predict_proba(split.X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print(f"Model: {artifact.get('model_name', 'unknown')}")
    print(f"Threshold: {threshold:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(split.y_test, y_pred))
    print("\\nClassification Report:")
    print(classification_report(split.y_test, y_pred, digits=4, zero_division=0))

    print("Summary Metrics:")
    print(f"Precision: {precision_score(split.y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(split.y_test, y_pred, zero_division=0):.4f}")
    print(f"F1:        {f1_score(split.y_test, y_pred, zero_division=0):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(split.y_test, y_prob):.4f}")
    print(f"PR-AUC:    {average_precision_score(split.y_test, y_prob):.4f}")


if __name__ == "__main__":
    main()
