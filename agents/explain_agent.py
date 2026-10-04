import json
import os
import warnings

import numpy as np
import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

import matplotlib
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pydantic import BaseModel, ConfigDict, Field

from agents.ml_agent import build_model, prepare_features, split_data
from utils.llm import call_llm_json

load_dotenv()


# The estimator class each best_model_name is expected to produce. This is the
# single source of truth for both "can we explain this model?" and the
# post-construction mismatch guard — keep it in sync with the model names
# ml_agent.train_and_evaluate reports.
EXPECTED_MODEL_CLASSES = {
    "Random Forest": (RandomForestClassifier, RandomForestRegressor),
    "XGBoost": (xgb.XGBClassifier, xgb.XGBRegressor),
    "Logistic Regression": (LogisticRegression,),
    "Linear Regression": (LinearRegression,),
}


def assert_model_matches_best(model, best_model_name: str):
    """Fail loudly if the estimator we're about to explain isn't the one ML
    reported as best. Silently substituting a different model produces SHAP
    values that describe something other than what the report claims."""
    expected = EXPECTED_MODEL_CLASSES.get(best_model_name)
    if expected is None:
        raise ValueError(
            f"Cannot explain unknown best model '{best_model_name}'. "
            f"Known models: {sorted(EXPECTED_MODEL_CLASSES)}."
        )
    if not isinstance(model, expected):
        raise ValueError(
            f"Explanation model mismatch: best_model is '{best_model_name}' "
            f"but the retrained estimator is {type(model).__name__}. Refusing "
            f"to explain a model the report doesn't claim as best."
        )


def train_best_model(df: pd.DataFrame, target_col: str,
                     problem_type: str, best_model_name: str):
    """
    Retrain the model ml_agent reported as best, using ml_agent's own feature
    preparation, model factory and train/test split. Sharing that code (rather
    than mirroring it here) is what guarantees the SHAP values describe the model
    the report talks about.
    """
    if best_model_name not in EXPECTED_MODEL_CLASSES:
        raise ValueError(
            f"Cannot explain unknown best model '{best_model_name}'. "
            f"Known models: {sorted(EXPECTED_MODEL_CLASSES)}."
        )

    X, y, _ = prepare_features(df, target_col, problem_type)
    X_train, X_test, y_train, y_test = split_data(X, y, problem_type)

    model = build_model(best_model_name, problem_type)
    model.fit(X_train, y_train)

    # Linear models come wrapped in a scaling pipeline. SHAP explains the final
    # estimator, so hand it the scaled features — as DataFrames, so column names
    # survive into the plot.
    if isinstance(model, Pipeline):
        preprocess = model[:-1]
        X_train = pd.DataFrame(preprocess.transform(X_train),
                               columns=X.columns, index=X_train.index)
        X_test = pd.DataFrame(preprocess.transform(X_test),
                              columns=X.columns, index=X_test.index)
        model = model[-1]

    # Catches the case where problem_type and best_model_name disagree — e.g.
    # 'Logistic Regression' on a regression problem.
    assert_model_matches_best(model, best_model_name)
    return model, X_train, X_test, X.columns.tolist()


def compute_shap_values(model, X_train, X_test):
    print("Computing SHAP values...")
    if isinstance(model, (LogisticRegression, LinearRegression)):
        explainer = shap.LinearExplainer(model, X_train)
    else:
        explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Older SHAP versions return one array per class; stack them into the same
    # (samples, features, classes) layout newer versions use.
    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=-1)

    return explainer, shap_values


OUTPUT_DIR = "outputs"
SHAP_PLOT_PATH = os.path.join(OUTPUT_DIR, "shap_summary.png")


def _as_2d(shap_values) -> np.ndarray:
    """
    Collapse SHAP output to (samples, features). Classifiers return one slice
    per class; averaging |SHAP| across classes gives a ranking that does not
    privilege any one class (for a binary model both slices are mirror images).
    """
    arr = np.array(shap_values)
    if arr.ndim == 3:
        arr = np.abs(arr).mean(axis=2)
    return arr


def get_feature_importance_summary(shap_values, feature_names: list) -> dict:
    mean_shap = np.abs(_as_2d(shap_values)).mean(axis=0).flatten()

    importance_dict = {
        feature: round(float(importance), 4)
        for feature, importance in zip(feature_names, mean_shap)
    }

    return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


def save_shap_plot(shap_values, X_test, feature_names: list) -> str:
    arr = np.array(shap_values)
    # The beeswarm needs signed values, so plot the positive class rather than
    # the unsigned class-average used for the ranking.
    if arr.ndim == 3:
        arr = arr[:, :, -1]

    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(arr, X_test_df, show=False, plot_size=None)
    plt.tight_layout()
    plt.savefig(SHAP_PLOT_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"SHAP plot saved to {SHAP_PLOT_PATH}")
    return SHAP_PLOT_PATH


class FeatureEffect(BaseModel):
    model_config = ConfigDict(extra="ignore")

    feature: str
    impact: str
    explanation: str


class ShapInterpretation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    plain_english_summary: str
    top_3_features: list[FeatureEffect] = Field(max_length=5)
    surprising_findings: str
    business_insight: str


def interpret_shap_with_ai(importance_dict: dict,
                            problem_type: str,
                            target_col: str) -> dict:
    prompt = f"""
You are an expert data scientist explaining model predictions.

TARGET VARIABLE: {target_col}
PROBLEM TYPE: {problem_type}

FEATURE IMPORTANCE (mean |SHAP| value):
{json.dumps(importance_dict, indent=2)}

Respond ONLY with JSON:
{{
    "plain_english_summary": "2-3 sentence explanation",
    "top_3_features": [
        {{
            "feature": "feature name",
            "impact": "positive or negative",
            "explanation": "plain English explanation"
        }}
    ],
    "surprising_findings": "anything unexpected",
    "business_insight": "practical actionable insight"
}}
"""
    return call_llm_json(
        prompt=prompt,
        system="You are an AI explainability expert. Respond with valid JSON only.",
        model_cls=ShapInterpretation,
        max_tokens=1024,
    ).model_dump()


def run_explain_agent(df: pd.DataFrame, target_col: str,
                      problem_type: str, best_model_name: str) -> dict:
    print("\nExplanation Agent starting...")
    print(f"Retraining {best_model_name} for explanation...")

    model, X_train, X_test, feature_names = train_best_model(
        df, target_col, problem_type, best_model_name
    )

    explainer, shap_values = compute_shap_values(model, X_train, X_test)
    importance_dict = get_feature_importance_summary(shap_values, feature_names)

    print("\nFeature importance (SHAP):")
    for feat, val in list(importance_dict.items())[:5]:
        print(f"  {feat}: {val}")

    plot_path = save_shap_plot(shap_values, X_test, feature_names)

    print("\nAsking AI to interpret SHAP results...")
    interpretation = interpret_shap_with_ai(importance_dict, problem_type, target_col)

    print("\n--- Explanation Summary ---")
    print(f"Summary: {interpretation['plain_english_summary']}")

    return {
        "explained_model": best_model_name,
        "feature_importance": importance_dict,
        "interpretation": interpretation,
        "shap_plot_path": plot_path
    }
