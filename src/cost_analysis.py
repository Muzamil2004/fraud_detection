"""Cost analysis across thresholds for fraud detection."""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score

try:
    from src.preprocessing import load_data, split_data
except ModuleNotFoundError:
    from preprocessing import load_data, split_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cost analysis")
    parser.add_argument("--model", default="model.pkl")
    parser.add_argument("--data", default="data/credit_card_fraud_10k.csv")
    parser.add_argument("--target", default="is_fraud")
    parser.add_argument("--fp-cost", type=float, default=1.0)
    parser.add_argument("--fn-cost", type=float, default=10.0)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--output", default="data/cost_analysis.csv")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


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

    rows = []
    for t in np.arange(args.step, 1.0, args.step):
        y_pred = (y_prob >= t).astype(int)
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        cost = (fp * args.fp_cost) + (fn * args.fn_cost)
        rows.append(
            {
                "threshold": round(float(t), 4),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "fp": fp,
                "fn": fn,
                "total_cost": cost,
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(args.output, index=False)

    best_row = result.loc[result["total_cost"].idxmin()]
    print(f"Saved cost analysis: {args.output}")
    print(
        f"Best threshold={best_row['threshold']:.4f}, "
        f"cost={best_row['total_cost']:.2f}, "
        f"precision={best_row['precision']:.4f}, "
        f"recall={best_row['recall']:.4f}"
    )


if __name__ == "__main__":
    main()
