"""Train fraud detection models and persist the best one."""

from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier

try:
    from src.preprocessing import BehavioralFeatureEngineer, build_preprocessor, load_data, split_data
except ModuleNotFoundError:
    from preprocessing import BehavioralFeatureEngineer, build_preprocessor, load_data, split_data


def _build_model_spaces(scale_pos_weight: float, random_state: int) -> Dict[str, Dict[str, object]]:
    log_reg = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        class_weight="balanced",
        random_state=random_state,
    )
    random_forest = RandomForestClassifier(
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    hist_gb = HistGradientBoostingClassifier(random_state=random_state)
    mlp = MLPClassifier(
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=random_state,
    )
    soft_voting = VotingClassifier(
        estimators=[
            (
                "lr",
                LogisticRegression(
                    max_iter=1500,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
            ("hgb", HistGradientBoostingClassifier(random_state=random_state)),
        ],
        voting="soft",
    )

    spaces: Dict[str, Dict[str, object]] = {
        "logistic_regression": {
            "estimator": log_reg,
            "param_distributions": {
                "model__C": [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
                "model__penalty": ["l1", "l2"],
            },
        },
        "random_forest": {
            "estimator": random_forest,
            "param_distributions": {
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [None, 8, 12, 16],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", None],
            },
        },
        "hist_gradient_boosting": {
            "estimator": hist_gb,
            "param_distributions": {
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__max_leaf_nodes": [15, 31, 63],
                "model__max_depth": [None, 4, 8, 12],
                "model__min_samples_leaf": [20, 50, 100],
                "model__l2_regularization": [0.0, 0.01, 0.1, 1.0],
            },
        },
        "mlp_classifier": {
            "estimator": mlp,
            "param_distributions": {
                "model__hidden_layer_sizes": [(64, 32), (96, 48), (128, 64), (64, 64, 32)],
                "model__activation": ["relu", "tanh"],
                "model__alpha": [1e-5, 1e-4, 1e-3, 1e-2],
                "model__learning_rate_init": [1e-4, 5e-4, 1e-3, 2e-3],
            },
        },
        "soft_voting_ensemble": {
            "estimator": soft_voting,
            "param_distributions": {
                "model__weights": [(1, 1, 1), (2, 1, 2), (1, 2, 2), (3, 1, 1)],
            },
        },
    }

    try:
        from xgboost import XGBClassifier

        spaces["xgboost"] = {
            "estimator": XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=random_state,
                n_jobs=-1,
            ),
            "param_distributions": {
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.7, 0.85, 1.0],
                "model__reg_lambda": [1.0, 3.0, 5.0],
            },
        }
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier

        spaces["lightgbm"] = {
            "estimator": LGBMClassifier(
                objective="binary",
                class_weight="balanced",
                random_state=random_state,
                verbose=-1,
            ),
            "param_distributions": {
                "model__n_estimators": [200, 300, 500],
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__num_leaves": [31, 63, 127],
                "model__max_depth": [-1, 6, 10],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.7, 0.85, 1.0],
            },
        }
    except Exception:
        pass

    try:
        from catboost import CatBoostClassifier

        spaces["catboost"] = {
            "estimator": CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="AUC",
                auto_class_weights="Balanced",
                random_state=random_state,
                verbose=False,
            ),
            "param_distributions": {
                "model__iterations": [200, 400, 600],
                "model__depth": [4, 6, 8],
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__l2_leaf_reg": [1, 3, 5, 7, 9],
            },
        }
    except Exception:
        pass

    return spaces


def _build_search(
    pipeline: ImbPipeline,
    param_distributions: Dict[str, object],
    cv_folds: int,
    random_state: int,
    search_iter: int,
) -> RandomizedSearchCV:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    return RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=search_iter,
        scoring="average_precision",
        n_jobs=-1,
        cv=cv,
        refit=True,
        random_state=random_state,
        verbose=0,
    )


def _build_pipeline(
    preprocessor: object,
    clf: object,
    random_state: int,
    use_smote: bool,
) -> ImbPipeline:
    steps = [("feature_engineering", BehavioralFeatureEngineer()), ("preprocessor", preprocessor)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))
    steps.append(("model", clf))
    return ImbPipeline(steps=steps)


def _safe_float(value: float | np.floating) -> float:
    return float(value)


def _to_serializable(params: Dict[str, object]) -> Dict[str, object]:
    serializable: Dict[str, object] = {}
    for key, value in params.items():
        if isinstance(value, np.generic):
            serializable[key] = value.item()
        else:
            serializable[key] = value
    return serializable


def evaluate_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "precision": _safe_float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": _safe_float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": _safe_float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_float(roc_auc_score(y_true, y_prob)),
        "pr_auc": _safe_float(average_precision_score(y_true, y_prob)),
    }


def train(
    data_path: str,
    output_model_path: str,
    target_col: str = "is_fraud",
    random_state: int = 42,
    use_smote: bool = True,
    cv_folds: int = 5,
    search_iter: int = 12,
) -> Tuple[str, Dict[str, Dict[str, float]]]:
    df = load_data(data_path)
    split = split_data(
        df,
        target_col=target_col,
        drop_cols=["transaction_id"] if "transaction_id" in df.columns else [],
        test_size=0.2,
        random_state=random_state,
    )

    feature_engineer = BehavioralFeatureEngineer()
    X_train_engineered = feature_engineer.fit_transform(split.X_train)
    preprocessor, _, _ = build_preprocessor(X_train_engineered)

    pos = int(split.y_train.sum())
    neg = int(len(split.y_train) - pos)
    scale_pos_weight = neg / max(pos, 1)

    model_spaces = _build_model_spaces(scale_pos_weight=scale_pos_weight, random_state=random_state)

    all_metrics: Dict[str, Dict[str, float]] = {}
    benchmark: Dict[str, Dict[str, object]] = {}
    best_name = ""
    best_pipeline = None
    best_score = -1.0

    for name, spec in model_spaces.items():
        pipeline = _build_pipeline(
            preprocessor=preprocessor,
            clf=spec["estimator"],
            random_state=random_state,
            use_smote=use_smote,
        )
        search = _build_search(
            pipeline=pipeline,
            param_distributions=spec["param_distributions"],
            cv_folds=cv_folds,
            random_state=random_state,
            search_iter=search_iter,
        )

        search.fit(split.X_train, split.y_train)

        tuned_model = search.best_estimator_
        y_prob = tuned_model.predict_proba(split.X_test)[:, 1]
        metrics = evaluate_at_threshold(split.y_test.to_numpy(), y_prob, threshold=0.5)
        all_metrics[name] = metrics
        benchmark[name] = {
            "cv_pr_auc": _safe_float(search.best_score_),
            "best_params": _to_serializable(search.best_params_),
            "test_metrics_at_0_5": metrics,
        }

        if metrics["pr_auc"] > best_score:
            best_score = metrics["pr_auc"]
            best_name = name
            best_pipeline = tuned_model

    artifact = {
        "model": best_pipeline,
        "model_name": best_name,
        "threshold": 0.5,
        "target_col": target_col,
        "metrics_at_0_5": all_metrics,
        "benchmark": benchmark,
        "search": {
            "strategy": "RandomizedSearchCV",
            "cv_folds": cv_folds,
            "search_iter": search_iter,
            "scoring": "average_precision",
            "stratified": True,
            "use_smote": use_smote,
        },
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
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--search-iter", type=int, default=12)
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
        cv_folds=args.cv_folds,
        search_iter=args.search_iter,
    )

    print(f"Best model: {best_name}")
    for name, metrics in all_metrics.items():
        print(f"\n{name}")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
