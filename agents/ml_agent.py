"""
Model training and selection.

Same division of labour as hypothesis validation: everything that has a correct
answer (problem type, metrics, which model scored best) is computed in code, and
the LLM is only asked to interpret numbers it did not choose.
"""

import json
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils.llm import call_llm_json

warnings.filterwarnings('ignore')
load_dotenv()

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
MAX_CLASSES = 20            # integer targets with more levels than this are treated as regression
MAX_FEATURE_CARDINALITY = 50

MODEL_NAMES = {
    "classification": ["Logistic Regression", "Random Forest", "XGBoost"],
    "regression": ["Linear Regression", "Random Forest", "XGBoost"],
}

# Model selection uses the cross-validated score, not the single hold-out split:
# one 80/20 split on a few hundred rows is too noisy to rank models by.
# F1 rather than accuracy, because accuracy rewards predicting the majority class.
SELECTION_METRIC = {
    "classification": "cv_f1_mean",
    "regression": "cv_r2_mean",
}


# ── Problem framing ─────────────────────────────────────────────────────────

def decide_problem_type(y: pd.Series) -> dict:
    """Classification vs regression from the target's dtype and cardinality."""
    n_unique = int(y.nunique(dropna=True))

    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return {"problem_type": "classification",
                "reasoning": f"target is non-numeric ({y.dtype}) with {n_unique} classes"}

    values = y.dropna().to_numpy(dtype=float)
    integer_valued = bool(np.all(np.mod(values, 1) == 0))
    if integer_valued and n_unique <= MAX_CLASSES:
        return {"problem_type": "classification",
                "reasoning": f"target is integer-valued with {n_unique} distinct values "
                             f"(<= {MAX_CLASSES})"}

    return {"problem_type": "regression",
            "reasoning": f"target is numeric with {n_unique} distinct values"}


def prepare_features(df: pd.DataFrame, target_col: str, problem_type: str):
    """
    Encode features for tree and linear models alike. Shared with the explain
    agent so the model it explains sees exactly the features this one trained on.
    """
    df = df.copy()
    y = df[target_col]
    X = df.drop(columns=[target_col])

    cols_to_drop = [col for col in X.columns
                    if X[col].dtype == "object" and X[col].nunique() > MAX_FEATURE_CARDINALITY]
    X = X.drop(columns=cols_to_drop)
    if cols_to_drop:
        print(f"Dropped high-cardinality columns: {cols_to_drop}")

    for col in X.select_dtypes(include=["object", "category", "bool"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Safety net for anything the cleaning plan left missing — scikit-learn's
    # linear models refuse NaN outright.
    for col in X.columns[X.isnull().any()]:
        X[col] = X[col].fillna(X[col].median())

    # XGBoost requires class labels 0..k-1, so {1, 2, 3} targets must be encoded too.
    if problem_type == "classification":
        y = LabelEncoder().fit_transform(y.astype(str))
    else:
        y = y.to_numpy(dtype=float)

    return X, y, cols_to_drop


def split_data(X, y, problem_type: str):
    """The one train/test split both training and explanation use."""
    stratify = None
    if problem_type == "classification" and np.bincount(y).min() >= 2:
        stratify = y
    return train_test_split(X, y, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE, stratify=stratify)


def build_model(name: str, problem_type: str):
    """
    Construct a fresh estimator. Linear models are wrapped in a scaling pipeline,
    so cross-validation fits the scaler inside each fold instead of leaking
    statistics from the validation fold.
    """
    if problem_type == "classification":
        models = {
            "Logistic Regression": lambda: make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=1000)),
            "Random Forest": lambda: RandomForestClassifier(
                n_estimators=100, random_state=RANDOM_STATE),
            "XGBoost": lambda: xgb.XGBClassifier(
                random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0),
        }
    else:
        models = {
            "Linear Regression": lambda: make_pipeline(StandardScaler(), LinearRegression()),
            "Random Forest": lambda: RandomForestRegressor(
                n_estimators=100, random_state=RANDOM_STATE),
            "XGBoost": lambda: xgb.XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
        }
    if name not in models:
        raise ValueError(f"Unknown {problem_type} model '{name}'. Known: {sorted(models)}")
    return models[name]()


def _cv_splitter(y, problem_type: str):
    if problem_type == "classification":
        folds = max(2, min(CV_FOLDS, int(np.bincount(y).min())))
        return StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    return KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


# ── Training ────────────────────────────────────────────────────────────────

def train_and_evaluate(X, y, problem_type: str) -> dict:
    X_train, X_test, y_train, y_test = split_data(X, y, problem_type)
    cv = _cv_splitter(y, problem_type)
    binary = problem_type == "classification" and len(np.unique(y)) == 2

    results = {}
    for name in MODEL_NAMES[problem_type]:
        model = build_model(name, problem_type)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if problem_type == "classification":
            cv_scores = cross_val_score(build_model(name, problem_type), X, y,
                                        cv=cv, scoring="f1_weighted")
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred, average="weighted"),
                "roc_auc": (roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
                            if binary else None),
                "cv_f1_mean": cv_scores.mean(),
                "cv_f1_std": cv_scores.std(),
            }
            print(f"{name}: F1={metrics['f1']:.4f}, CV F1={metrics['cv_f1_mean']:.4f}")
        else:
            cv_scores = cross_val_score(build_model(name, problem_type), X, y,
                                        cv=cv, scoring="r2")
            metrics = {
                "r2_score": r2_score(y_test, y_pred),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "mae": mean_absolute_error(y_test, y_pred),
                "cv_r2_mean": cv_scores.mean(),
                "cv_r2_std": cv_scores.std(),
            }
            print(f"{name}: R2={metrics['r2_score']:.4f}, CV R2={metrics['cv_r2_mean']:.4f}")

        results[name] = {k: (round(float(v), 4) if v is not None else None)
                         for k, v in metrics.items()}

    return results


def select_best_model(results: dict, problem_type: str) -> str:
    """Highest cross-validated score wins; ties go to the simpler (earlier) model."""
    metric = SELECTION_METRIC[problem_type]
    order = MODEL_NAMES[problem_type]
    return max(results, key=lambda name: (results[name][metric], -order.index(name)))


# ── Interpretation (LLM narrates; it does not choose) ───────────────────────

class ModelInterpretation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    reasoning: str
    performance_summary: str
    concerns: str
    recommendation: str


def interpret_results(results: dict, problem_type: str, best_model: str) -> dict:
    metric = SELECTION_METRIC[problem_type]
    prompt = f"""
You are an expert data scientist reviewing ML model results.

PROBLEM TYPE: {problem_type}
MODEL RESULTS: {json.dumps(results, indent=2)}

The best model was selected in code by highest {metric}: {best_model}.
That choice is final. Explain why the numbers favour it, and flag anything
concerning (overfitting gaps between hold-out and CV, high variance, weak scores).

Respond ONLY with JSON:
{{
    "reasoning": "why {best_model} is best, citing the metrics",
    "performance_summary": "one sentence on overall performance",
    "concerns": "any concerns about the results",
    "recommendation": "what to try next to improve"
}}
"""
    interpretation = call_llm_json(
        prompt=prompt,
        system="You are an ML expert. Respond with valid JSON only.",
        model_cls=ModelInterpretation,
        max_tokens=768,
    ).model_dump()

    # Set after the LLM call so nothing in its response can override the choice.
    interpretation["best_model"] = best_model
    interpretation["selection_metric"] = metric
    return interpretation


def run_ml_agent(df: pd.DataFrame, target_col: str = None) -> dict:
    print("\nML Agent starting...")

    if target_col is None:
        target_col = df.columns[-1]
        print(f"No target column specified. Using last column: '{target_col}'")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' is not in the cleaned dataset")

    problem_info = decide_problem_type(df[target_col])
    problem_type = problem_info["problem_type"]
    print(f"Problem type: {problem_type} ({problem_info['reasoning']})")

    print("\nPreparing features...")
    X, y, dropped = prepare_features(df, target_col, problem_type)
    print(f"Features shape: {X.shape}")

    print(f"\nTraining models for {problem_type}...")
    results = train_and_evaluate(X, y, problem_type)
    best_model = select_best_model(results, problem_type)

    print("\nAsking AI to interpret results...")
    interpretation = interpret_results(results, problem_type, best_model)

    print("\n--- ML Summary ---")
    print(f"Best model:  {interpretation['best_model']}")
    print(f"Reasoning:   {interpretation['reasoning']}")
    print(f"Performance: {interpretation['performance_summary']}")
    print(f"Concerns:    {interpretation['concerns']}")
    print(f"Next steps:  {interpretation['recommendation']}")

    return {
        "problem_type": problem_type,
        "problem_type_reasoning": problem_info["reasoning"],
        "target_column": target_col,
        "dropped_features": dropped,
        "model_results": results,
        "interpretation": interpretation,
    }
