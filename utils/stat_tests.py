"""
Statistical testing for hypothesis validation.

INVARIANT: this module must never import utils.llm. Every verdict ADA reports is
decided here, in code, from a computed test statistic — that guarantee is only
real if no model can reach into this file. tests/test_stat_tests.py asserts it.

Pipeline: resolve -> test -> correct across the family -> decide verdict.
"""

import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from utils.hypothesis_schema import (
    ClaimType,
    Direction,
    NotTestableReason,
    TestSpec,
    Verdict,
)

ALPHA = 0.05

# Conventional small-effect floors (Cohen family). These are heuristics, not
# laws — a "CONFIRMED" that clears them is still only as good as the convention.
PRACTICAL_THRESHOLDS = {
    "hedges_g": 0.2,
    "rank_biserial": 0.1,
    "eta_squared": 0.01,
    "epsilon_squared": 0.01,
    "cramers_v": 0.1,
    "pearson_r": 0.1,
    "spearman_r": 0.1,
}

MIN_GROUP_N = 5           # below this a group mean carries no information
MIN_TOTAL_N = 20
MIN_CORRELATION_N = 10
MAX_GROUP_LEVELS = 20     # above this a "group comparison" is really an ID column
MAX_NUMERIC_AS_CATEGORICAL = 10
PARAMETRIC_N = 30         # CLT threshold: above it, skip the normality test


# ── Column resolution ───────────────────────────────────────────────────────

def _match_column(name: str, columns) -> Optional[str]:
    """
    Exact match, then an unambiguous case/whitespace-insensitive match.

    Deliberately no fuzzy matching: silently mapping a hallucinated 'icome' onto
    'income' would produce an audit trail claiming a hypothesis about 'icome'
    was confirmed by a test that never touched it.
    """
    if name in columns:
        return name
    target = str(name).strip().lower()
    hits = [c for c in columns if str(c).strip().lower() == target]
    return hits[0] if len(hits) == 1 else None


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _usable_as_categorical(series: pd.Series) -> tuple[bool, Optional[str]]:
    """Returns (ok, coercion_note)."""
    if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
        return True, None
    # Numeric columns like Pclass or SibSp are categorical in intent.
    if series.nunique(dropna=True) <= MAX_NUMERIC_AS_CATEGORICAL:
        return True, "treated numeric column as categorical (low cardinality)"
    return False, None


def resolve(spec: TestSpec, df: pd.DataFrame) -> tuple[Optional[dict], Optional[dict]]:
    """
    Validate that a spec can actually run against this dataframe.

    Returns (resolved, None) or (None, {reason, detail}).
    """
    def fail(reason: NotTestableReason, detail: str):
        return None, {"reason": reason.value, "detail": detail}

    resolved_names = {}
    for role, name in (
        ("group_var", spec.group_var),
        ("outcome_var", spec.outcome_var),
        ("var_a", spec.var_a),
        ("var_b", spec.var_b),
    ):
        if name is None:
            continue
        match = _match_column(name, df.columns)
        if match is None:
            return fail(
                NotTestableReason.UNKNOWN_COLUMN,
                f"column '{name}' is not in the dataset",
            )
        resolved_names[role] = match

    coercions: list[str] = []

    if spec.claim_type in (ClaimType.GROUP_DIFFERENCE_NUMERIC, ClaimType.PROPORTION_DIFFERENCE):
        group_col = resolved_names["group_var"]
        outcome_col = resolved_names["outcome_var"]

        ok, note = _usable_as_categorical(df[group_col])
        if not ok:
            return fail(
                NotTestableReason.TOO_MANY_LEVELS,
                f"'{group_col}' has {df[group_col].nunique()} distinct values — "
                f"too many to compare as groups",
            )
        if note:
            coercions.append(f"{group_col}: {note}")

        levels = df[group_col].dropna().unique()
        if len(levels) > MAX_GROUP_LEVELS:
            return fail(
                NotTestableReason.TOO_MANY_LEVELS,
                f"'{group_col}' has {len(levels)} groups (limit {MAX_GROUP_LEVELS})",
            )

        if spec.claim_type == ClaimType.GROUP_DIFFERENCE_NUMERIC:
            if not _is_numeric(df[outcome_col]):
                return fail(
                    NotTestableReason.WRONG_DTYPE,
                    f"'{outcome_col}' is not numeric, so group means are undefined",
                )
        else:
            if df[outcome_col].dropna().nunique() != 2:
                return fail(
                    NotTestableReason.NOT_BINARY,
                    f"'{outcome_col}' has {df[outcome_col].dropna().nunique()} distinct "
                    f"values — a proportion claim needs exactly 2",
                )

        if spec.focus_level is not None:
            as_str = {str(v) for v in levels}
            if str(spec.focus_level) not in as_str:
                return fail(
                    NotTestableReason.UNKNOWN_FOCUS_LEVEL,
                    f"'{spec.focus_level}' is not a value of '{group_col}' "
                    f"(values: {sorted(as_str)[:8]})",
                )

    elif spec.claim_type == ClaimType.CORRELATION_NUMERIC:
        for role in ("var_a", "var_b"):
            col = resolved_names[role]
            if not _is_numeric(df[col]):
                return fail(
                    NotTestableReason.WRONG_DTYPE,
                    f"'{col}' is not numeric, so a correlation is undefined",
                )

    elif spec.claim_type == ClaimType.ASSOCIATION_CATEGORICAL:
        for role in ("var_a", "var_b"):
            col = resolved_names[role]
            ok, note = _usable_as_categorical(df[col])
            if not ok:
                return fail(
                    NotTestableReason.TOO_MANY_LEVELS,
                    f"'{col}' has {df[col].nunique()} distinct values — too many for "
                    f"a contingency table",
                )
            if note:
                coercions.append(f"{col}: {note}")

    return {"columns": resolved_names, "coercions": coercions}, None


# ── Test execution ──────────────────────────────────────────────────────────

def _insufficient(detail: str, n_used: int, n_total: int) -> dict:
    return {
        "verdict_hint": Verdict.INSUFFICIENT_DATA,
        "detail": detail,
        "n_used": int(n_used),
        "n_total": int(n_total),
    }


def _split_focus(series_group: pd.Series, focus_level) -> tuple[pd.Series, str, str]:
    """Boolean mask selecting the focus level, plus display labels."""
    mask = series_group.astype(str) == str(focus_level)
    return mask, str(focus_level), "all other groups"


def _run_group_difference(spec, df, resolved) -> dict:
    group_col = resolved["columns"]["group_var"]
    outcome_col = resolved["columns"]["outcome_var"]
    n_total = len(df)

    sub = df[[group_col, outcome_col]].dropna()
    sub = sub[np.isfinite(pd.to_numeric(sub[outcome_col], errors="coerce"))]
    n_used = len(sub)
    if n_used < MIN_TOTAL_N:
        return _insufficient(f"only {n_used} complete rows (need {MIN_TOTAL_N})", n_used, n_total)

    levels = sub[group_col].unique()
    if len(levels) < 2:
        return _insufficient("fewer than 2 groups present after dropping missing values",
                             n_used, n_total)

    directional = spec.direction in (Direction.HIGHER_IN_FOCUS, Direction.LOWER_IN_FOCUS)
    outcome = pd.to_numeric(sub[outcome_col], errors="coerce")

    # An omnibus F/H test says "the groups differ somewhere" — it cannot tell us
    # WHICH group is higher, so a directional claim is always reduced to
    # focus-versus-rest rather than run as an omnibus.
    if directional or len(levels) == 2:
        if directional:
            mask, label_a, label_b = _split_focus(sub[group_col], spec.focus_level)
        else:
            ordered = sorted(levels, key=str)
            mask = sub[group_col] == ordered[0]
            label_a, label_b = str(ordered[0]), str(ordered[1])
        a = outcome[mask].to_numpy(dtype=float)
        b = outcome[~mask].to_numpy(dtype=float)

        if min(a.size, b.size) < MIN_GROUP_N:
            return _insufficient(
                f"smallest group has {min(a.size, b.size)} rows (need {MIN_GROUP_N})",
                n_used, n_total,
            )

        from utils import effect_sizes as es

        if min(a.size, b.size) >= PARAMETRIC_N:
            parametric, assumption = True, f"both groups n>={PARAMETRIC_N}, CLT applies"
        else:
            pa = stats.shapiro(a).pvalue if 3 <= a.size <= 5000 else 0.0
            pb = stats.shapiro(b).pvalue if 3 <= b.size <= 5000 else 0.0
            parametric = pa > 0.05 and pb > 0.05
            assumption = (
                f"Shapiro-Wilk normality p={pa:.3f}/{pb:.3f} — "
                f"{'parametric' if parametric else 'nonparametric'} test selected"
            )

        if parametric:
            # Always Welch: strictly safer than pooled-variance Student's t and
            # costs nothing when variances happen to be equal.
            result = stats.ttest_ind(a, b, equal_var=False)
            effect, ci = es.hedges_g(a, b)
            record = {
                "test_used": "Welch's t-test",
                "statistic": float(result.statistic),
                "p_value": float(result.pvalue),
                "effect_name": "hedges_g",
            }
        else:
            result = stats.mannwhitneyu(a, b, alternative="two-sided")
            effect, ci = es.rank_biserial(a, b)
            record = {
                "test_used": "Mann-Whitney U",
                "statistic": float(result.statistic),
                "p_value": float(result.pvalue),
                "effect_name": "rank_biserial",
            }

        record.update({
            "effect_value": effect,
            "effect_ci": ci,
            "observed_direction": _sign(effect),
            "comparison": f"{label_a} vs {label_b}",
            "group_means": {label_a: float(a.mean()), label_b: float(b.mean())},
            "assumption_note": assumption,
            "n_used": n_used,
            "n_total": n_total,
        })
        return record

    # Non-directional claim across 3+ groups: omnibus, no direction check.
    groups = [outcome[sub[group_col] == lv].to_numpy(dtype=float) for lv in levels]
    if min(g.size for g in groups) < MIN_GROUP_N:
        return _insufficient(
            f"smallest of {len(groups)} groups has {min(g.size for g in groups)} rows",
            n_used, n_total,
        )

    from utils import effect_sizes as es

    normal = all(
        (stats.shapiro(g).pvalue > 0.05) if 3 <= g.size <= 5000 else False
        for g in groups
    )
    equal_var = stats.levene(*groups, center="median").pvalue > 0.05

    if normal and equal_var:
        result = stats.f_oneway(*groups)
        effect, ci = es.eta_squared(groups)
        record = {
            "test_used": "One-way ANOVA",
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "effect_name": "eta_squared",
            "effect_value": effect,
            "effect_ci": ci,
        }
    else:
        result = stats.kruskal(*groups)
        record = {
            "test_used": "Kruskal-Wallis",
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "effect_name": "epsilon_squared",
            "effect_value": es.epsilon_squared(float(result.statistic), n_used),
            "effect_ci": None,
        }

    record.update({
        "observed_direction": 0,
        "comparison": f"{len(levels)} groups of {group_col}",
        "assumption_note": (
            f"normality {'ok' if normal else 'violated'}, "
            f"equal variance {'ok' if equal_var else 'violated'}"
        ),
        "n_used": n_used,
        "n_total": n_total,
    })
    return record


def _run_proportion_difference(spec, df, resolved) -> dict:
    group_col = resolved["columns"]["group_var"]
    outcome_col = resolved["columns"]["outcome_var"]
    n_total = len(df)

    sub = df[[group_col, outcome_col]].dropna()
    n_used = len(sub)
    if n_used < MIN_TOTAL_N:
        return _insufficient(f"only {n_used} complete rows (need {MIN_TOTAL_N})", n_used, n_total)

    outcome_levels = sorted(sub[outcome_col].unique(), key=str)
    if len(outcome_levels) != 2:
        return _insufficient("outcome is not binary after dropping missing values",
                             n_used, n_total)
    success = outcome_levels[-1]

    if spec.focus_level is not None:
        mask, label_a, label_b = _split_focus(sub[group_col], spec.focus_level)
    else:
        group_levels = sorted(sub[group_col].unique(), key=str)
        if len(group_levels) != 2:
            return _insufficient(
                f"'{group_col}' has {len(group_levels)} groups; a non-directional "
                f"proportion claim needs exactly 2",
                n_used, n_total,
            )
        mask = sub[group_col] == group_levels[0]
        label_a, label_b = str(group_levels[0]), str(group_levels[1])

    is_success = sub[outcome_col] == success
    table = np.array([
        [int((mask & is_success).sum()), int((mask & ~is_success).sum())],
        [int((~mask & is_success).sum()), int((~mask & ~is_success).sum())],
    ])
    if table.sum(axis=1).min() < MIN_GROUP_N:
        return _insufficient(
            f"smallest group has {int(table.sum(axis=1).min())} rows (need {MIN_GROUP_N})",
            n_used, n_total,
        )

    from utils import effect_sizes as es

    expected = stats.contingency.expected_freq(table)
    if expected.min() < 5:
        result = stats.fisher_exact(table, alternative="two-sided")
        test_used, statistic = "Fisher's exact test", float(result.statistic)
        p_value = float(result.pvalue)
        assumption = f"expected cell count {expected.min():.1f} < 5 — exact test used"
    else:
        result = stats.chi2_contingency(table, correction=True)
        test_used, statistic = "Chi-square (Yates-corrected)", float(result.statistic)
        p_value = float(result.pvalue)
        assumption = f"all expected cell counts >= 5 (min {expected.min():.1f})"

    or_value, or_ci = es.odds_ratio(table)
    rd_value, rd_ci = es.risk_difference(table)
    p_focus = table[0, 0] / table[0].sum()
    p_rest = table[1, 0] / table[1].sum()

    return {
        "test_used": test_used,
        "statistic": statistic,
        "p_value": p_value,
        "effect_name": "odds_ratio",
        "effect_value": or_value,
        "effect_ci": or_ci,
        "risk_difference": rd_value,
        "risk_difference_ci": rd_ci,
        "observed_direction": _sign(rd_value),
        "comparison": f"P({outcome_col}={success}) in {label_a} vs {label_b}",
        "group_proportions": {label_a: float(p_focus), label_b: float(p_rest)},
        "assumption_note": assumption,
        "n_used": n_used,
        "n_total": n_total,
    }


def _run_correlation(spec, df, resolved) -> dict:
    col_a = resolved["columns"]["var_a"]
    col_b = resolved["columns"]["var_b"]
    n_total = len(df)

    sub = df[[col_a, col_b]].apply(pd.to_numeric, errors="coerce").dropna()
    n_used = len(sub)
    if n_used < MIN_CORRELATION_N:
        return _insufficient(
            f"only {n_used} complete pairs (need {MIN_CORRELATION_N})", n_used, n_total
        )

    x = sub[col_a].to_numpy(dtype=float)
    y = sub[col_b].to_numpy(dtype=float)
    if x.std() == 0 or y.std() == 0:
        return _insufficient("one of the columns is constant", n_used, n_total)

    from utils import effect_sizes as es

    if n_used >= PARAMETRIC_N:
        normal, assumption = True, f"n>={PARAMETRIC_N}, CLT applies"
    else:
        px = stats.shapiro(x).pvalue
        py = stats.shapiro(y).pvalue
        normal = px > 0.05 and py > 0.05
        assumption = f"Shapiro-Wilk normality p={px:.3f}/{py:.3f}"

    if normal:
        result = stats.pearsonr(x, y)
        ci = result.confidence_interval()
        record = {
            "test_used": "Pearson correlation",
            "effect_name": "pearson_r",
            "effect_ci": (float(ci.low), float(ci.high)),
        }
    else:
        result = stats.spearmanr(x, y)
        # scipy exposes no CI for Spearman; the Fisher z approximation is the
        # standard substitute.
        record = {
            "test_used": "Spearman rank correlation",
            "effect_name": "spearman_r",
            "effect_ci": es.fisher_z_ci(float(result.statistic), n_used),
        }

    r = float(result.statistic)
    record.update({
        "statistic": r,
        "p_value": float(result.pvalue),
        "effect_value": r,
        "observed_direction": _sign(r),
        "comparison": f"{col_a} vs {col_b}",
        "r_squared": round(r * r, 4),
        "assumption_note": assumption,
        "n_used": n_used,
        "n_total": n_total,
    })
    return record


def _run_association(spec, df, resolved) -> dict:
    col_a = resolved["columns"]["var_a"]
    col_b = resolved["columns"]["var_b"]
    n_total = len(df)

    sub = df[[col_a, col_b]].dropna()
    n_used = len(sub)
    if n_used < MIN_TOTAL_N:
        return _insufficient(f"only {n_used} complete rows (need {MIN_TOTAL_N})", n_used, n_total)

    table = pd.crosstab(sub[col_a], sub[col_b])
    table = table.loc[table.sum(axis=1) > 0, table.sum(axis=0) > 0]
    if table.shape[0] < 2 or table.shape[1] < 2:
        return _insufficient(
            f"contingency table collapsed to {table.shape} — need at least 2x2",
            n_used, n_total,
        )

    from utils import effect_sizes as es

    counts = table.to_numpy()
    expected = stats.contingency.expected_freq(counts)
    low_cells = float((expected < 5).mean())

    if counts.shape == (2, 2) and expected.min() < 5:
        result = stats.fisher_exact(counts, alternative="two-sided")
        test_used, statistic = "Fisher's exact test", float(result.statistic)
        p_value = float(result.pvalue)
    else:
        result = stats.chi2_contingency(counts, correction=(counts.shape == (2, 2)))
        test_used, statistic = "Chi-square test of independence", float(result.statistic)
        p_value = float(result.pvalue)

    record = {
        "test_used": test_used,
        "statistic": statistic,
        "p_value": p_value,
        "effect_name": "cramers_v",
        "effect_value": es.cramers_v(counts),
        "effect_ci": None,
        "observed_direction": 0,
        "comparison": f"{col_a} x {col_b} ({table.shape[0]}x{table.shape[1]} table)",
        "assumption_note": (
            f"{low_cells:.0%} of expected cell counts below 5"
            if low_cells else "all expected cell counts >= 5"
        ),
        "n_used": n_used,
        "n_total": n_total,
    }
    # Cochran's rule: chi-square is not trustworthy once a fifth of the expected
    # counts fall below 5, so report it rather than let it pass silently.
    if low_cells > 0.2 and test_used.startswith("Chi-square"):
        record["verdict_hint"] = Verdict.INSUFFICIENT_DATA
        record["detail"] = (
            f"{low_cells:.0%} of expected cell counts are below 5, so the "
            f"chi-square approximation is unreliable"
        )
    return record


_EXECUTORS = {
    ClaimType.GROUP_DIFFERENCE_NUMERIC: _run_group_difference,
    ClaimType.PROPORTION_DIFFERENCE: _run_proportion_difference,
    ClaimType.CORRELATION_NUMERIC: _run_correlation,
    ClaimType.ASSOCIATION_CATEGORICAL: _run_association,
}


def _sign(value) -> int:
    if value is None or not np.isfinite(value):
        return 0
    return 1 if value > 0 else (-1 if value < 0 else 0)


def run_test(spec: TestSpec, df: pd.DataFrame) -> dict:
    """Resolve and execute one hypothesis. Never raises; returns a record."""
    resolved, failure = resolve(spec, df)
    if failure is not None:
        return {
            "verdict_hint": Verdict.NOT_TESTABLE,
            "reason": failure["reason"],
            "detail": failure["detail"],
            "n_used": 0,
            "n_total": len(df),
        }
    try:
        record = _EXECUTORS[spec.claim_type](spec, df, resolved)
    except Exception as exc:  # a malformed column shape shouldn't sink the run
        return {
            "verdict_hint": Verdict.INSUFFICIENT_DATA,
            "detail": f"test could not be computed: {exc}",
            "n_used": 0,
            "n_total": len(df),
        }
    record.setdefault("coercions", resolved["coercions"])
    return record


# ── Multiple-comparison correction ──────────────────────────────────────────

def benjamini_hochberg(pvalues) -> list[float]:
    """
    BH (FDR) adjusted p-values. The descending cumulative-minimum step enforces
    monotonicity and handles ties without special-casing them.
    """
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(m)
    adjusted[order] = np.clip(ranked, 0.0, 1.0)
    return [float(v) for v in adjusted]


def holm_bonferroni(pvalues) -> list[float]:
    """
    Holm (FWER) adjusted p-values.

    Arguably the better default for a handful of curated, human-facing claims —
    it bounds the chance of *any* false confirmation and assumes nothing about
    independence — but BH is the configured default.
    """
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p)
    ranked = p[order] * (m - np.arange(m))
    ranked = np.maximum.accumulate(ranked)
    adjusted = np.empty(m)
    adjusted[order] = np.clip(ranked, 0.0, 1.0)
    return [float(v) for v in adjusted]


CORRECTION_METHODS = {
    "benjamini-hochberg": benjamini_hochberg,
    "holm": holm_bonferroni,
}


# ── Verdict ─────────────────────────────────────────────────────────────────

def _practically_significant(effect_name: str, effect_value) -> Optional[bool]:
    if effect_value is None or not np.isfinite(effect_value):
        return None
    if effect_name == "odds_ratio":
        # Convert to a Cohen's-d equivalent so one threshold covers it.
        if effect_value <= 0:
            return None
        d_equivalent = abs(math.log(effect_value)) * math.sqrt(3) / math.pi
        return d_equivalent >= PRACTICAL_THRESHOLDS["hedges_g"]
    threshold = PRACTICAL_THRESHOLDS.get(effect_name)
    if threshold is None:
        return None
    return abs(effect_value) >= threshold


def _direction_matches(direction: Direction, observed: int) -> Optional[bool]:
    if direction == Direction.ANY:
        return None
    if direction in (Direction.HIGHER_IN_FOCUS, Direction.POSITIVE):
        return observed > 0
    return observed < 0


def decide_verdict(spec: TestSpec, record: dict, alpha: float = ALPHA) -> dict:
    """
    The single seam that determines what ADA claims. Pure: takes a spec and a
    computed record, returns a verdict. No dataframe, no model, no I/O.
    """
    if "verdict_hint" in record:
        return {
            "verdict": record["verdict_hint"],
            "practically_significant": None,
            "direction_match": None,
        }

    p_adj = record.get("p_adjusted", record.get("p_value"))
    practical = _practically_significant(record.get("effect_name"), record.get("effect_value"))
    match = _direction_matches(spec.direction, record.get("observed_direction", 0))

    if p_adj is None or not np.isfinite(p_adj):
        verdict = Verdict.INSUFFICIENT_DATA
    elif p_adj >= alpha:
        verdict = Verdict.NOT_SUPPORTED
    elif match is False:
        # Significant, but the effect runs the opposite way to the claim.
        verdict = Verdict.REJECTED
    elif practical is False:
        verdict = Verdict.SIGNIFICANT_BUT_TRIVIAL
    else:
        verdict = Verdict.CONFIRMED

    return {
        "verdict": verdict,
        "practically_significant": practical,
        "direction_match": match,
    }


# ── Evidence rendering (code-generated, never LLM-written) ───────────────────

def _fmt_p(p) -> str:
    if p is None or not np.isfinite(p):
        return "n/a"
    return "<0.0001" if p < 0.0001 else f"{p:.4f}"


def _fmt_ci(ci) -> str:
    if not ci or any(v is None or not np.isfinite(v) for v in ci):
        return ""
    return f" [95% CI {ci[0]:.3f}, {ci[1]:.3f}]"


def render_evidence(record: dict, verdict: Verdict) -> str:
    """Build the evidence sentence from the numbers, so it cannot drift from them."""
    if verdict == Verdict.NOT_TESTABLE:
        return f"Not testable — {record.get('detail', 'could not be mapped to a test')}."
    if verdict == Verdict.INSUFFICIENT_DATA:
        return f"Insufficient data — {record.get('detail', 'preconditions not met')}."

    effect = record.get("effect_value")
    effect_text = (
        f", {record.get('effect_name', 'effect')}="
        f"{effect:.3f}{_fmt_ci(record.get('effect_ci'))}"
        if effect is not None and np.isfinite(effect) else ""
    )
    parts = [
        f"{record.get('test_used', 'test')} on {record.get('comparison', 'the data')}: "
        f"statistic={record.get('statistic', float('nan')):.4f}, "
        f"p={_fmt_p(record.get('p_value'))}",
    ]
    if record.get("p_adjusted") is not None:
        parts.append(f" ({record.get('correction_method', 'BH')}-adjusted q="
                     f"{_fmt_p(record['p_adjusted'])})")
    parts.append(effect_text)
    parts.append(f", n={record.get('n_used')} of {record.get('n_total')} rows")

    sentence = "".join(parts) + "."
    if verdict == Verdict.NOT_SUPPORTED:
        sentence += (
            " This means the data did not provide enough evidence to reject the null "
            "hypothesis - not that the hypothesis is false."
        )
    elif verdict == Verdict.SIGNIFICANT_BUT_TRIVIAL:
        sentence += (
            " The result is statistically significant but the effect is below the "
            "conventional threshold for a practically meaningful difference."
        )
    return sentence


# ── Top-level orchestration of the maths (no LLM anywhere in here) ───────────

def run_statistical_validation(specs: dict[str, TestSpec], df: pd.DataFrame,
                                alpha: float = ALPHA,
                                correction: str = "benjamini-hochberg") -> dict[str, dict]:
    """
    Run every hypothesis, correct across the family, then decide verdicts.

    The correction family is the set of tests that actually produced a p-value —
    including NOT_TESTABLE slots would distort the q-values of the real tests.
    """
    records = {hid: run_test(spec, df) for hid, spec in specs.items()}

    tested = [hid for hid, r in records.items() if r.get("p_value") is not None
              and "verdict_hint" not in r]
    if tested:
        adjust = CORRECTION_METHODS.get(correction, benjamini_hochberg)
        adjusted = adjust([records[hid]["p_value"] for hid in tested])
        for hid, q in zip(tested, adjusted):
            records[hid]["p_adjusted"] = q
            records[hid]["correction_method"] = correction
            records[hid]["family_size"] = len(tested)

    for hid, spec in specs.items():
        outcome = decide_verdict(spec, records[hid], alpha=alpha)
        records[hid].update(outcome)
        records[hid]["evidence"] = render_evidence(records[hid], outcome["verdict"])
    return records
