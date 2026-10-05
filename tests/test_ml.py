"""Model training, selection and leakage detection (ada/ml)."""

import numpy as np
import pandas as pd
import pytest

from ada.ml.explain import explain
from ada.ml.modeling import (
    decide_problem_type,
    prepare_features,
    run_modeling,
    select_best_model,
    train_and_evaluate,
)


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
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=120), "b": rng.choice(["p", "q"], size=120),
                       "label": np.repeat([1, 2, 3], 40)})
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


def test_leftover_missing_values_do_not_break_linear_models():
    df = pd.DataFrame({"a": [1.0, None] * 30, "b": list(range(60)), "y": [0, 1, 1] * 20})
    X, _, _ = prepare_features(df, "y", "classification")
    assert X.isnull().sum().sum() == 0


def test_leaked_column_is_flagged_and_can_be_excluded(titanic):
    df = titanic.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"])
    df["refund_issued"] = 1 - df["Survived"]          # recorded after the outcome

    leaky = run_modeling(df, "Survived")
    assert [s["feature"] for s in leaky["leakage_suspects"]] == ["refund_issued"]
    assert any("leakage" in w for w in leaky["warnings"])

    clean = run_modeling(df, "Survived", exclude=("refund_issued",))
    assert clean["leakage_suspects"] == []
    assert 0.7 < clean["models"][clean["best_model"]]["cv_f1_mean"] < 0.95
    assert clean["majority_class_baseline_accuracy"] == pytest.approx(0.6162, abs=1e-3)


def test_shap_explanation_reports_direction(titanic, tmp_path):
    df = titanic.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"])
    out = explain(df, "Survived", "Random Forest", str(tmp_path / "shap.png"))
    top = {f["feature"]: f for f in out["top_features"]}
    assert "Sex" in list(top)[:2]
    # Higher Pclass (third class) pushes predicted survival down.
    assert top["Pclass"]["value_shap_spearman"] < 0
    assert (tmp_path / "shap.png").exists()
