"""
Every tool an agent can call. Each is deterministic computation over the
dataset; none of them calls a model.
"""

import json
import os
from typing import Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ada.ledger import Effect, Finding, FindingKind
from ada.ml.explain import explain
from ada.ml.modeling import run_modeling
from ada.stats.hypothesis_schema import TestSpec
from ada.stats.stat_tests import run_test
from ada.tools import sandbox
from ada.tools.registry import Tool, ToolContext

SENTINELS = (-999, -99, -9, -1, 99, 999, 9999, 99999)


def _jsonable(value):
    return json.loads(json.dumps(value, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def _require(ctx: ToolContext, *columns: str) -> None:
    missing = [c for c in columns if c not in ctx.df.columns]
    if missing:
        raise ValueError(f"unknown column(s) {missing}; available: {list(ctx.df.columns)}")


def _as_numeric_outcome(series: pd.Series) -> pd.Series:
    """Numeric as-is; a two-level non-numeric column becomes 0/1 (sorted, last level = 1)."""
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return series.astype(float)
    levels = sorted(series.dropna().unique(), key=str)
    if len(levels) != 2:
        raise ValueError(f"'{series.name}' is neither numeric nor binary ({len(levels)} levels)")
    return (series == levels[1]).astype(float).where(series.notna())


# ── describe_column ─────────────────────────────────────────────────────────

class ColumnArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    column: str


def describe_column(ctx: ToolContext, a: ColumnArgs) -> dict:
    _require(ctx, a.column)
    s = ctx.df[a.column]
    out = {"column": a.column, "dtype": str(s.dtype), "rows": int(len(s)),
           "missing": int(s.isna().sum()), "missing_pct": round(float(s.isna().mean()) * 100, 2),
           "n_unique": int(s.nunique())}
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        d = s.describe()
        out["summary"] = {k: round(float(d[k]), 4) for k in ("mean", "std", "min", "25%", "50%", "75%", "max")}
        out["skew"] = round(float(s.skew()), 3)
        out["negative_count"] = int((s < 0).sum())
        sentinel_hits = {str(v): int((s == v).sum()) for v in SENTINELS if (s == v).sum() > 0}
        if sentinel_hits:
            out["possible_sentinel_values"] = sentinel_hits
    else:
        counts = s.value_counts(dropna=True).head(10)
        out["top_values"] = [{"value": str(k), "count": int(v), "share": round(float(v / s.notna().sum()), 4)}
                             for k, v in counts.items()]
    return out


# ── group_summary ───────────────────────────────────────────────────────────

class GroupArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    group_by: str = Field(description="Categorical column to group by")
    metric: str = Field(description="Numeric or binary column to summarise per group")


def group_summary(ctx: ToolContext, a: GroupArgs) -> dict:
    _require(ctx, a.group_by, a.metric)
    metric = _as_numeric_outcome(ctx.df[a.metric])
    frame = pd.DataFrame({"g": ctx.df[a.group_by].astype(str), "m": metric}).dropna()
    groups = frame.groupby("g")["m"].agg(["count", "mean", "median"]).sort_values("mean", ascending=False)
    if len(groups) > 25:
        raise ValueError(f"'{a.group_by}' has {len(groups)} groups — too many to summarise")
    binary = metric.dropna().isin([0, 1]).all()
    return {
        "group_by": a.group_by, "metric": a.metric,
        "metric_is_binary_rate": bool(binary),
        "overall_mean": round(float(frame["m"].mean()), 4),
        "groups": [{"group": g, "n": int(r["count"]), "mean": round(float(r["mean"]), 4),
                    "median": round(float(r["median"]), 4)} for g, r in groups.iterrows()],
        "note": "Descriptive only — use run_hypothesis_test before claiming a difference is real.",
    }


# ── correlations ────────────────────────────────────────────────────────────

class CorrelationArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    column: str = Field(description="Numeric or binary column to correlate everything else against")
    top_k: int = Field(default=10, ge=1, le=30)


def correlations(ctx: ToolContext, a: CorrelationArgs) -> dict:
    _require(ctx, a.column)
    y = _as_numeric_outcome(ctx.df[a.column])
    rows = []
    for col in ctx.df.columns:
        if col == a.column:
            continue
        s = ctx.df[col]
        frame = pd.DataFrame({"x": s, "y": y}).dropna()
        if len(frame) < 10:
            continue
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            if frame["x"].std() == 0:
                continue
            rho = frame["x"].corr(frame["y"], method="spearman")
            rows.append({"column": col, "measure": "spearman_rho", "value": round(float(rho), 4)})
        elif s.nunique() <= 25:
            # Correlation ratio (eta): share of y's variance explained by the categories.
            groups = frame.groupby(frame["x"].astype(str))["y"]
            ss_between = float((groups.count() * (groups.mean() - frame["y"].mean()) ** 2).sum())
            ss_total = float(((frame["y"] - frame["y"].mean()) ** 2).sum())
            if ss_total > 0:
                rows.append({"column": col, "measure": "eta", "value": round((ss_between / ss_total) ** 0.5, 4)})
    rows.sort(key=lambda r: abs(r["value"]), reverse=True)
    return {"against": a.column, "results": rows[:a.top_k],
            "note": "Association, not causation. eta is unsigned; spearman_rho is signed."}


# ── check_data_quality ──────────────────────────────────────────────────────

class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


def check_data_quality(ctx: ToolContext, a: NoArgs) -> dict:
    df = ctx.df
    issues = []
    dupes = int(df.duplicated().sum())
    if dupes:
        issues.append({"type": "duplicate_rows", "count": dupes, "pct": round(dupes / len(df) * 100, 2)})
    for col in df.columns:
        s = df[col]
        miss = float(s.isna().mean())
        if miss > 0:
            issues.append({"type": "missing_values", "column": col, "pct": round(miss * 100, 2)})
        if s.nunique(dropna=True) <= 1:
            issues.append({"type": "constant_column", "column": col})
            continue
        if s.nunique() == len(s) and len(s) > 50:
            issues.append({"type": "id_like_column", "column": col,
                           "detail": "unique on every row — not a feature"})
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            clean = s.dropna()
            for v in SENTINELS:
                hits = int((clean == v).sum())
                if hits and hits / len(clean) >= 0.005:
                    rest = clean[clean != v]
                    if len(rest) and (v < rest.quantile(0.01) or v > rest.quantile(0.99)):
                        issues.append({"type": "sentinel_value", "column": col, "value": v,
                                       "count": hits, "detail": "placeholder value far outside the "
                                                                "column's normal range"})
            negatives = int((clean < 0).sum())
            if 0 < negatives <= 0.05 * len(clean):
                issues.append({"type": "suspicious_negatives", "column": col, "count": negatives})
            z = (clean - clean.mean()) / clean.std() if clean.std() else clean * 0
            extreme = int((z.abs() > 5).sum())
            if extreme:
                issues.append({"type": "extreme_outliers", "column": col, "count": extreme,
                               "detail": "|z| > 5"})
        elif s.dtype == object:
            parsed = pd.to_numeric(s.dropna(), errors="coerce")
            share = float(parsed.notna().mean()) if len(parsed) else 0.0
            if 0.9 <= share < 1.0:
                issues.append({"type": "numeric_stored_as_text", "column": col,
                               "unparseable_examples": s.dropna()[parsed.isna()].astype(str).head(5).tolist()})
    return {"rows": int(len(df)), "columns": int(df.shape[1]), "issue_count": len(issues), "issues": issues}


# ── run_hypothesis_test ─────────────────────────────────────────────────────

def run_hypothesis_test(ctx: ToolContext, spec: TestSpec) -> dict:
    record = _jsonable(run_test(spec, ctx.df))
    if "verdict_hint" in record:
        return {"status": "not_testable", "verdict": record["verdict_hint"],
                "detail": record.get("detail"), "n_used": record.get("n_used")}
    return {"status": "tested", "p_value": record["p_value"], "record": record}


def _after_test(ctx: ToolContext, eid: str) -> dict:
    ctx.ledger.correct_family()
    ev = ctx.ledger.evidence(eid)
    r = ev.result
    if r.get("status") != "tested":
        return r
    rec = r["record"]
    return {
        "status": "tested", "test": rec.get("test_used"), "comparison": rec.get("comparison"),
        "statistic": rec.get("statistic"), "p_value": r["p_value"],
        "p_adjusted": r.get("p_adjusted"), "family_size": r.get("family_size"),
        "effect": {"name": rec.get("effect_name"), "value": rec.get("effect_value"),
                   "ci_95": rec.get("effect_ci")},
        "group_means": rec.get("group_means"), "group_proportions": rec.get("group_proportions"),
        "risk_difference": rec.get("risk_difference"),
        "n_used": rec.get("n_used"), "assumptions": rec.get("assumption_note"),
        "verdict": r.get("verdict"), "evidence": r.get("evidence"),
        "note": "p_adjusted is corrected across EVERY test the team has run so far and may "
                "rise as more tests are run. Only CONFIRMED verdicts support a statistical claim.",
    }


# ── train_models / explain_model ────────────────────────────────────────────

class TrainArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: Optional[str] = Field(default=None, description="Defaults to the investigation target")
    exclude_columns: list[str] = Field(default_factory=list,
                                       description="Columns to leave out, e.g. IDs or leaked columns")


def _target(ctx: ToolContext, target: Optional[str]) -> str:
    target = target or ctx.target
    if not target:
        raise ValueError("no target column given and the investigation has no target")
    _require(ctx, target)
    return target


def train_models(ctx: ToolContext, a: TrainArgs) -> dict:
    target = _target(ctx, a.target)
    result = run_modeling(ctx.df, target, exclude=tuple(a.exclude_columns))
    ctx.cache[("model", target, tuple(sorted(a.exclude_columns)))] = result
    return _jsonable(result)


def explain_model(ctx: ToolContext, a: TrainArgs) -> dict:
    target = _target(ctx, a.target)
    key = ("model", target, tuple(sorted(a.exclude_columns)))
    trained = ctx.cache.get(key) or run_modeling(ctx.df, target, exclude=tuple(a.exclude_columns))
    plot = os.path.join(ctx.run_dir, f"shap_{target}.png")
    out = explain(ctx.df, target, trained["best_model"], plot, exclude=tuple(a.exclude_columns))
    metric = trained["selection_metric"]
    out["cv_score"] = {metric: trained["models"][trained["best_model"]][metric]}
    return _jsonable(out)


# ── run_python ──────────────────────────────────────────────────────────────

class PythonArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str = Field(description="pandas code over `df` (pd, np, stats preloaded; no imports or "
                                  "file access). The last expression is printed.")


def run_python(ctx: ToolContext, a: PythonArgs) -> dict:
    result = sandbox.run_python(a.code, ctx.df)
    if not result["ok"]:
        raise RuntimeError(result["error"])
    return {"code": a.code, "stdout": result["stdout"]}


# ── submit_findings (terminal) ──────────────────────────────────────────────

class FindingInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str = Field(description="One specific sentence with the key numbers, copied from evidence")
    kind: FindingKind
    columns: list[str] = Field(description="Dataset columns this finding is about")
    evidence_ids: list[str] = Field(min_length=1, description="IDs like E3 that support every number in the claim")
    confidence: Literal["high", "medium", "low"]
    implication: str = Field(description="What a decision-maker should do or watch, one sentence")
    effect: Effect = Field(
        default="not_applicable",
        description="'present' if the claim says these columns are associated with / drive the target, "
                    "'absent' if it says they are not or only negligibly, 'not_applicable' for "
                    "data-quality or other findings")


class SubmitArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    findings: list[FindingInput] = Field(min_length=0, max_length=6)
    summary: str = Field(description="Two sentences on what you investigated and what you found")


def submit_findings(ctx: ToolContext, a: SubmitArgs) -> dict:
    unknown = sorted({eid for f in a.findings for eid in f.evidence_ids
                      if ctx.ledger.evidence(eid) is None})
    if unknown:
        raise ValueError(f"evidence ids {unknown} do not exist — cite only IDs returned by your tools")
    for f in a.findings:
        ctx.submitted.append(Finding(agent=ctx.agent, **f.model_dump()))
    ctx.cache["summary"] = a.summary
    return {"accepted_for_review": len(a.findings)}


# ── registry ────────────────────────────────────────────────────────────────

TOOLS = {t.name: t for t in [
    Tool("describe_column", "Summary statistics, missingness, top values and possible sentinel "
         "placeholder values for one column.", ColumnArgs, describe_column),
    Tool("group_summary", "Mean/median (or rate, for a binary metric) of a metric within each group "
         "of a categorical column. Descriptive only.", GroupArgs, group_summary),
    Tool("correlations", "Rank every other column by its association with one numeric or binary "
         "column (Spearman for numeric, correlation ratio eta for categorical).",
         CorrelationArgs, correlations),
    Tool("check_data_quality", "Scan the whole dataset for duplicates, missing values, sentinel "
         "placeholders (e.g. -999), suspicious negatives, extreme outliers, ID-like and constant "
         "columns, and numbers stored as text.", NoArgs, check_data_quality),
    Tool("run_hypothesis_test",
         "Run a real statistical test for one hypothesis. The test (Welch t, Mann-Whitney, "
         "chi-square, Fisher, Pearson, Spearman, ANOVA, Kruskal-Wallis) is chosen from assumption "
         "checks. Returns effect size with 95% CI and a verdict computed in code after correcting "
         "for every test the team has run. claim_type rules: group_difference_numeric and "
         "proportion_difference need group_var + outcome_var (+ focus_level for a directional "
         "claim); correlation_numeric needs var_a + var_b (both numeric); association_categorical "
         "needs var_a + var_b and direction 'any'.",
         TestSpec, run_hypothesis_test, after_record=_after_test),
    Tool("train_models", "Train and cross-validate Logistic/Linear Regression, Random Forest and "
         "XGBoost on the target; selects the best by CV score, reports baseline and checks every "
         "feature for target leakage.", TrainArgs, train_models),
    Tool("explain_model", "SHAP explanation of the best model: top features by mean |SHAP| and the "
         "direction of each feature's effect.", TrainArgs, explain_model),
    Tool("run_python", "Run sandboxed pandas code over the dataset `df` for analysis no other tool "
         "covers. No imports, files or network.", PythonArgs, run_python),
    Tool("submit_findings", "Submit your findings and finish. Every number in a claim must appear in "
         "a cited evidence item. A statistical claim must cite a run_hypothesis_test result whose "
         "verdict is CONFIRMED.", SubmitArgs, submit_findings, record_evidence=False),
]}

SPECIALIST_TOOLS = {
    "data_quality": ["check_data_quality", "describe_column", "group_summary", "run_python",
                     "submit_findings"],
    "statistician": ["describe_column", "group_summary", "correlations", "run_hypothesis_test",
                     "run_python", "submit_findings"],
    "ml_engineer": ["train_models", "explain_model", "correlations", "describe_column",
                    "submit_findings"],
}


def profile_dataset(df: pd.DataFrame) -> dict:
    """Compact column profile given to every agent up front."""
    cols = []
    for col in df.columns:
        s = df[col]
        entry = {"name": str(col), "dtype": str(s.dtype), "missing_pct": round(float(s.isna().mean()) * 100, 1),
                 "n_unique": int(s.nunique())}
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            entry["range"] = [_jsonable(s.min()), _jsonable(s.max())]
        if s.nunique() <= 12:
            entry["levels"] = [str(v) for v in sorted(s.dropna().unique(), key=str)]
        else:
            entry["examples"] = [str(v) for v in s.dropna().head(3)]
        cols.append(entry)
    return {"rows": int(len(df)), "columns": cols}


__all__ = ["TOOLS", "SPECIALIST_TOOLS", "profile_dataset", "np"]
