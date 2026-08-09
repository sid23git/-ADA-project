import pandas as pd
import numpy as np
from dotenv import load_dotenv
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.llm import call_llm

load_dotenv()


def train_best_model(df: pd.DataFrame, target_col: str,
                     problem_type: str, best_model_name: str):
    df = df.copy()
    y = df[target_col]
    X = df.drop(columns=[target_col])

    cols_to_drop = [col for col in X.columns
                    if X[col].dtype == "object" and X[col].nunique() > 50]
    X = X.drop(columns=cols_to_drop)

    le = LabelEncoder()
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = le.fit_transform(X[col].astype(str))

    if y.dtype == "object":
        y = le.fit_transform(y.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if best_model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42) \
            if problem_type == "classification" \
            else RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0) \
            if problem_type == "classification" \
            else xgb.XGBRegressor(random_state=42, verbosity=0)

    model.fit(X_train, y_train)
    return model, X_train, X_test, X.columns.tolist()


def compute_shap_values(model, X_train, X_test):
    print("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return explainer, shap_values


def get_feature_importance_summary(shap_values, feature_names: list) -> dict:
    if isinstance(shap_values, list):
        shap_array = np.array(shap_values[1])
    else:
        shap_array = np.array(shap_values)

    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 0]

    mean_shap = np.abs(shap_array).mean(axis=0).flatten()

    importance_dict = {
        feature: round(float(importance), 4)
        for feature, importance in zip(feature_names, mean_shap)
    }

    return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


def save_shap_plot(shap_values, X_test, feature_names: list):
    if isinstance(shap_values, list):
        shap_array = np.array(shap_values[1])
    else:
        shap_array = np.array(shap_values)

    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 0]

    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_array, X_test_df, show=False, plot_size=None)
    plt.tight_layout()
    plt.savefig("outputs/shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("SHAP plot saved to outputs/shap_summary.png")


def interpret_shap_with_ai(importance_dict: dict,
                            problem_type: str,
                            target_col: str) -> dict:
    prompt = f"""
You are an expert data scientist explaining model predictions.

TARGET VARIABLE: {target_col}
PROBLEM TYPE: {problem_type}

FEATURE IMPORTANCE (SHAP values):
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
    raw = call_llm(
        prompt=prompt,
        system="You are an AI explainability expert. Respond with valid JSON only.",
        max_tokens=1024
    )
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


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

    save_shap_plot(shap_values, X_test, feature_names)

    print("\nAsking AI to interpret SHAP results...")
    interpretation = interpret_shap_with_ai(importance_dict, problem_type, target_col)

    print(f"\n--- Explanation Summary ---")
    print(f"Summary: {interpretation['plain_english_summary']}")

    return {
        "feature_importance": importance_dict,
        "interpretation": interpretation,
        "shap_plot_path": "outputs/shap_summary.png"
    }
