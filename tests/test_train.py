from __future__ import annotations

import numpy as np

from src.train import _build_model_spaces, evaluate_at_threshold


def test_build_model_spaces_contains_core_candidates() -> None:
    spaces = _build_model_spaces(scale_pos_weight=10.0, random_state=42)
    required = {
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
        "mlp_classifier",
        "soft_voting_ensemble",
    }
    assert required.issubset(set(spaces.keys()))


def test_evaluate_at_threshold_returns_valid_metric_range() -> None:
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.4, 0.8, 0.7, 0.2])
    metrics = evaluate_at_threshold(y_true=y_true, y_prob=y_prob, threshold=0.5)

    for metric_name in ("precision", "recall", "f1", "roc_auc", "pr_auc"):
        assert 0.0 <= metrics[metric_name] <= 1.0

