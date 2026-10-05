"""
Model training, selection and leakage detection — pure computation, no LLM.

Everything that has a correct answer (problem type, metrics, which model scored
best, whether a feature leaks the target) is decided here; agents call this
through the train_models tool and reason about the numbers it returns.
"""

import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
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

warnings.filterwarnings('ignore')

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

        results[name] = {k: (round(float(v), 4) if v is not None else None)
                         for k, v in metrics.items()}

    return results


def select_best_model(results: dict, problem_type: str) -> str:
    """Highest cross-validated score wins; ties go to the simpler (earlier) model."""
    metric = SELECTION_METRIC[problem_type]
    order = MODEL_NAMES[problem_type]
    return max(results, key=lambda name: (results[name][metric], -order.index(name)))


# ── Leakage detection ───────────────────────────────────────────────────────

LEAKAGE_AUC = 0.95        # one feature alone separating the classes this well is suspicious
LEAKAGE_CORR = 0.95
SUSPICIOUS_CV_SCORE = 0.99


def detect_leakage(X: pd.DataFrame, y, problem_type: str) -> list[dict]:
    """
    Flag features that predict the target almost perfectly on their own.

    In real data that is almost never a genuine signal: it is a column recorded
    after the outcome (a refund flag for churn, a discharge code for
    readmission) or a re-encoding of the target. A model trained on it scores
    beautifully and is useless in production.
    """
    suspects = []
    y = np.asarray(y)
    binary = problem_type == "classification" and len(np.unique(y)) == 2
    for col in X.columns:
        x = X[col].to_numpy(dtype=float)
        if np.nanstd(x) == 0:
            continue
        if binary:
            auc = roc_auc_score(y, x)
            score = max(auc, 1 - auc)
            if score >= LEAKAGE_AUC:
                suspects.append({"feature": col, "metric": "univariate_auc", "value": round(score, 4)})
        elif problem_type == "regression":
            rho = abs(stats.spearmanr(x, y).statistic)
            if np.isfinite(rho) and rho >= LEAKAGE_CORR:
                suspects.append({"feature": col, "metric": "abs_spearman", "value": round(float(rho), 4)})
    return suspects


def run_modeling(df: pd.DataFrame, target_col: str, exclude: tuple = ()) -> dict:
    """Frame the problem, train every candidate, select the best, check for leakage."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' is not in the dataset")
    data = df.drop(columns=[c for c in exclude if c in df.columns and c != target_col])
    data = data.dropna(subset=[target_col])

    problem = decide_problem_type(data[target_col])
    problem_type = problem["problem_type"]
    X, y, dropped = prepare_features(data, target_col, problem_type)
    results = train_and_evaluate(X, y, problem_type)
    best = select_best_model(results, problem_type)
    metric = SELECTION_METRIC[problem_type]

    warnings_ = []
    leaks = detect_leakage(X, y, problem_type)
    if leaks:
        warnings_.append(
            f"possible target leakage: {[s['feature'] for s in leaks]} predict the target "
            f"almost perfectly on their own — check whether they are recorded after the outcome"
        )
    if results[best][metric] >= SUSPICIOUS_CV_SCORE:
        warnings_.append(f"{metric}={results[best][metric]} is implausibly high for real data")

    baseline = None
    if problem_type == "classification":
        counts = np.bincount(y)
        baseline = round(float(counts.max() / counts.sum()), 4)

    return {
        "target": target_col,
        "problem_type": problem_type,
        "problem_type_reasoning": problem["reasoning"],
        "n_rows": int(len(X)),
        "features": list(X.columns),
        "dropped_high_cardinality": dropped,
        "excluded": [c for c in exclude if c in df.columns],
        "models": results,
        "best_model": best,
        "selection_metric": metric,
        "majority_class_baseline_accuracy": baseline,
        "leakage_suspects": leaks,
        "warnings": warnings_,
    }
