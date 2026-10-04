"""SHAP explanations of the model train_models selected — pure computation."""

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

import matplotlib
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ada.ml.modeling import build_model, decide_problem_type, prepare_features, split_data

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


def save_shap_plot(shap_values, X_test, feature_names: list, path: str) -> str:
    arr = np.array(shap_values)
    # The beeswarm needs signed values, so plot the positive class rather than
    # the unsigned class-average used for the ranking.
    if arr.ndim == 3:
        arr = arr[:, :, -1]

    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(arr, X_test_df, show=False, plot_size=None)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def _direction(shap_2d: np.ndarray, X_test: pd.DataFrame, feature_names: list) -> dict:
    """
    Sign of the relationship between a feature's value and its SHAP value:
    +1 means higher values push the prediction up. Spearman, so it is robust to
    the feature's scale and to label-encoded categoricals' arbitrary order.
    """
    out = {}
    X = np.asarray(X_test, dtype=float)
    for i, name in enumerate(feature_names):
        x, s = X[:, i], shap_2d[:, i]
        if np.std(x) == 0 or np.std(s) == 0:
            out[name] = 0.0
            continue
        from scipy import stats
        out[name] = round(float(stats.spearmanr(x, s).statistic), 3)
    return out


def explain(df: pd.DataFrame, target_col: str, model_name: str, plot_path: str,
            exclude: tuple = (), top_k: int = 10) -> dict:
    data = df.drop(columns=[c for c in exclude if c in df.columns and c != target_col])
    data = data.dropna(subset=[target_col])
    problem_type = decide_problem_type(data[target_col])["problem_type"]

    model, X_train, X_test, feature_names = train_best_model(
        data, target_col, problem_type, model_name)
    _, shap_values = compute_shap_values(model, X_train, X_test)
    importance = get_feature_importance_summary(shap_values, feature_names)

    signed = np.array(shap_values)
    if signed.ndim == 3:
        signed = signed[:, :, -1]
    direction = _direction(signed, X_test, feature_names)
    save_shap_plot(shap_values, X_test, feature_names, plot_path)

    return {
        "target": target_col,
        "model": model_name,
        "problem_type": problem_type,
        "top_features": [
            {"feature": f, "mean_abs_shap": v, "value_shap_spearman": direction.get(f)}
            for f, v in list(importance.items())[:top_k]
        ],
        "plot_path": plot_path,
        "note": "value_shap_spearman > 0: higher feature values push the prediction "
                "towards the positive class / higher target. Label-encoded categoricals "
                "have arbitrary order, so their sign is not interpretable.",
    }
