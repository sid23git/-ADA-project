"""Unit tests for the deterministic halves of the cleaning and ML agents."""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from agents.cleaning_agent import CleaningStrategy, apply_cleaning_strategy
from agents.ml_agent import (
    decide_problem_type,
    prepare_features,
    select_best_model,
    train_and_evaluate,
)

# ── Cleaning ────────────────────────────────────────────────────────────────

def test_cleaning_never_drops_or_imputes_the_target():
    df = pd.DataFrame({"x": [1.0, None, 3.0, 4.0], "y": [0, 1, None, 1]})
    strategy = {"columns_to_drop": ["y"], "missing_value_strategies": {"y": "mean", "x": "mean"}}

    out = apply_cleaning_strategy(df, strategy, target_col="y")

    assert "y" in out.columns
    # The row with a missing label is dropped, not given an invented label.
    assert len(out) == 3
    assert out["y"].isnull().sum() == 0


def test_numeric_strategy_on_text_column_falls_back_to_mode():
    df = pd.DataFrame({"id": [1, 2, 3, 4], "city": ["a", "a", None, "b"]})
    out = apply_cleaning_strategy(df, {"missing_value_strategies": {"city": "mean"}})
    assert out["city"].tolist() == ["a", "a", "a", "b"]


def test_unknown_cleaning_strategy_is_rejected_by_the_schema():
    with pytest.raises(ValidationError):
        CleaningStrategy.model_validate(
            {"missing_value_strategies": {"age": "interpolate"}, "columns_to_drop": []}
        )


# ── ML ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("values, expected", [
    (["yes", "no", "yes"], "classification"),
    ([0, 1, 1, 0], "classification"),
    ([1, 2, 3, 2, 1], "classification"),
    ([0.5, 1.7, 2.2, 3.9], "regression"),
    (list(range(100)), "regression"),
])
def test_problem_type_is_decided_from_the_data(values, expected):
    assert decide_problem_type(pd.Series(values))["problem_type"] == expected


def test_best_model_is_highest_cv_score_with_ties_to_the_simpler_model():
    results = {
        "Logistic Regression": {"cv_f1_mean": 0.80},
        "Random Forest": {"cv_f1_mean": 0.80},
        "XGBoost": {"cv_f1_mean": 0.79},
    }
    assert select_best_model(results, "classification") == "Logistic Regression"

    results["XGBoost"]["cv_f1_mean"] = 0.81
    assert select_best_model(results, "classification") == "XGBoost"


def test_non_zero_indexed_class_labels_train_with_xgboost():
    # XGBoost rejects labels {1, 2, 3}; prepare_features must re-encode them.
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "a": rng.normal(size=120),
        "b": rng.choice(["p", "q"], size=120),
        "label": np.repeat([1, 2, 3], 40),
    })
    X, y, _ = prepare_features(df, "label", "classification")
    assert sorted(set(y)) == [0, 1, 2]

    results = train_and_evaluate(X, y, "classification")
    assert set(results) == {"Logistic Regression", "Random Forest", "XGBoost"}
    assert results["XGBoost"]["roc_auc"] is None     # only reported for binary targets


def test_regression_reports_cross_validated_scores():
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    df = pd.DataFrame({"x": x, "noise": rng.normal(size=200), "y": 3 * x + rng.normal(size=200)})
    X, y, _ = prepare_features(df, "y", "regression")

    results = train_and_evaluate(X, y, "regression")

    assert results["Linear Regression"]["cv_r2_mean"] > 0.8
    assert select_best_model(results, "regression") in results


def test_leftover_missing_values_do_not_break_linear_models():
    df = pd.DataFrame({"a": [1.0, None] * 30, "b": list(range(60)), "y": [0, 1, 1] * 20})
    X, y, _ = prepare_features(df, "y", "classification")
    assert X.isnull().sum().sum() == 0
