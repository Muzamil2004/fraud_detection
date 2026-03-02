"""Tune probability threshold based on F1 or custom costs."""

from __future__ import annotations

import argparse
import numpy as np
import joblib
from sklearn.metrics import f1_score

try:
    from src.preprocessing import load_data, split_data
except ModuleNotFoundError:
    from preprocessing import load_data, split_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Threshold tuning")
    parser.add_argument("--model", default="model.pkl")
    parser.add_argument("--data", default="data/credit_card_fraud_10k.csv")
    parser.add_argument("--target", default="is_fraud")
    parser.add_argument("--mode", choices=["f1", "cost"], default="f1")
    parser.add_argument("--fp-cost", type=float, default=1.0)
    parser.add_argument("--fn-cost", type=float, default=10.0)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _cost(y_true: np.ndarray, y_pred: np.ndarray, fp_cost: float, fn_cost: float) -> float:
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return (fp * fp_cost) + (fn * fn_cost)


def main() -> None:
    args = parse_args()

    artifact = joblib.load(args.model)
    model = artifact["model"]

    df = load_data(args.data)
    split = split_data(
        df,
        target_col=args.target,
        drop_cols=["transaction_id"] if "transaction_id" in df.columns else [],
        test_size=0.2,
        random_state=args.random_state,
    )

    y_true = split.y_test.to_numpy()
    y_prob = model.predict_proba(split.X_test)[:, 1]

    thresholds = np.arange(args.step, 1.0, args.step)
    best_threshold = 0.5

    if args.mode == "f1":
        best_score = -1.0
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = float(t)
        print(f"Best threshold (F1): {best_threshold:.4f}")
        print(f"Best F1: {best_score:.4f}")
    else:
        best_cost = float("inf")
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            c = _cost(y_true, y_pred, args.fp_cost, args.fn_cost)
            if c < best_cost:
                best_cost = c
                best_threshold = float(t)
        print(f"Best threshold (Cost): {best_threshold:.4f}")
        print(f"Minimum cost: {best_cost:.2f}")

    artifact["threshold"] = best_threshold
    joblib.dump(artifact, args.model)
    print(f"Saved updated threshold to: {args.model}")


if __name__ == "__main__":
    main()
