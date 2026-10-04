"""
Effect sizes and confidence intervals.

Significance without magnitude is the failure mode this whole feature exists to
fix, so every test ADA runs reports an effect size with an interval around it.

Pure numpy/scipy — no LLM import, no I/O. statsmodels is not a dependency, so
anything without a closed form is bootstrapped with a fixed seed (a CI that
moved between runs of the same data would defeat the point).
"""

import numpy as np
from scipy import stats

Z_95 = 1.959963984540054
BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 2000


def _finite(a) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]


def bootstrap_ci(statistic_fn, *samples, method: str = "basic",
                 n_resamples: int = BOOTSTRAP_RESAMPLES,
                 seed: int = BOOTSTRAP_SEED):
    """
    Percentile-family bootstrap CI for statistics with no usable closed form.

    Defaults to 'basic' rather than scipy's 'BCa': BCa's jackknife acceleration
    step misbehaves on discrete and sparse inputs (contingency tables in
    particular), where 'basic' degrades gracefully.
    """
    try:
        res = stats.bootstrap(
            samples,
            statistic_fn,
            method=method,
            n_resamples=n_resamples,
            random_state=np.random.default_rng(seed),
            vectorized=False,
        )
        lo = float(res.confidence_interval.low)
        hi = float(res.confidence_interval.high)
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return None
        return lo, hi
    except Exception:
        # A degenerate resample (constant sample, empty group) is not worth
        # failing a whole hypothesis over — report the point estimate alone.
        return None


# ── Standardised mean difference ────────────────────────────────────────────

def hedges_g(a, b):
    """
    Hedges' g with its Hedges-Olkin normal-approximation CI.

    Reported instead of raw Cohen's d because d is biased upward at small n,
    and small groups are exactly the case that needs care here.
    """
    a, b = _finite(a), _finite(b)
    n1, n2 = a.size, b.size
    if n1 < 2 or n2 < 2:
        return None, None

    s_pooled = np.sqrt(
        ((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2)
    )
    if s_pooled == 0:
        return None, None

    d = (a.mean() - b.mean()) / s_pooled
    correction = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
    g = d * correction

    se_d = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2.0 * (n1 + n2)))
    se_g = se_d * correction
    return float(g), (float(g - Z_95 * se_g), float(g + Z_95 * se_g))


def mean_difference(a, b):
    """Raw mean difference with the exact Welch CI scipy gives us for free."""
    a, b = _finite(a), _finite(b)
    if a.size < 2 or b.size < 2:
        return None, None
    diff = float(a.mean() - b.mean())
    try:
        ci = stats.ttest_ind(a, b, equal_var=False).confidence_interval()
        return diff, (float(ci.low), float(ci.high))
    except Exception:
        return diff, None


# ── Rank-based ──────────────────────────────────────────────────────────────

def rank_biserial(a, b):
    """
    Rank-biserial correlation from the Mann-Whitney U of `a` against `b`.

    Sign convention: positive means `a` tends to rank above `b`. The argument
    order therefore has to be caller-controlled (focus group first), not
    alphabetical, or the sign is uninterpretable.
    """
    a, b = _finite(a), _finite(b)
    if a.size < 1 or b.size < 1:
        return None, None
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    r = 2.0 * u / (a.size * b.size) - 1.0

    def _stat(x, y):
        if x.size < 1 or y.size < 1:
            return np.nan
        uu = stats.mannwhitneyu(x, y, alternative="two-sided").statistic
        return 2.0 * uu / (x.size * y.size) - 1.0

    return float(r), bootstrap_ci(_stat, a, b)


# ── Variance explained ──────────────────────────────────────────────────────

def eta_squared(groups):
    """eta-squared for a one-way layout: SS_between / SS_total."""
    groups = [_finite(g) for g in groups]
    groups = [g for g in groups if g.size > 0]
    if len(groups) < 2:
        return None, None

    all_values = np.concatenate(groups)
    grand_mean = all_values.mean()
    ss_total = float(((all_values - grand_mean) ** 2).sum())
    if ss_total == 0:
        return None, None
    ss_between = float(sum(g.size * (g.mean() - grand_mean) ** 2 for g in groups))
    eta2 = ss_between / ss_total

    def _stat(*samples):
        vals = np.concatenate(samples)
        gm = vals.mean()
        sst = ((vals - gm) ** 2).sum()
        if sst == 0:
            return np.nan
        ssb = sum(s.size * (s.mean() - gm) ** 2 for s in samples)
        return ssb / sst

    return float(eta2), bootstrap_ci(_stat, *groups)


def epsilon_squared(h_statistic: float, n: int):
    """
    epsilon-squared for Kruskal-Wallis: H(n+1)/(n^2-1).

    This is the Tomczak & Tomczak form; the H/(n-1) version that circulates
    informally is a different (cruder) quantity.
    """
    if n < 3:
        return None
    return float(h_statistic * (n + 1) / (n**2 - 1))


# ── Contingency tables ──────────────────────────────────────────────────────

def cramers_v(table) -> float | None:
    """
    Bias-corrected Cramer's V (Bergsma & Wicherts).

    The naive sqrt(chi2 / (n * min(r-1, c-1))) is biased upward at small n,
    which would inflate exactly the small-sample cases where a false
    "CONFIRMED" is most damaging.
    """
    table = np.asarray(table, dtype=float)
    n = table.sum()
    if n <= 1:
        return None
    r, c = table.shape
    if r < 2 or c < 2:
        return None

    chi2 = stats.chi2_contingency(table, correction=False).statistic
    phi2 = chi2 / n
    phi2_corrected = max(0.0, phi2 - (r - 1) * (c - 1) / (n - 1))
    r_corrected = r - (r - 1) ** 2 / (n - 1)
    c_corrected = c - (c - 1) ** 2 / (n - 1)
    denom = min(r_corrected - 1, c_corrected - 1)
    if denom <= 0:
        return None
    return float(np.sqrt(phi2_corrected / denom))


def odds_ratio(table):
    """
    Odds ratio for a 2x2 table with its Woolf log-scale CI.

    Applies the Haldane-Anscombe correction (+0.5 to every cell) when any cell
    is zero, which would otherwise make the OR or its SE undefined.
    """
    table = np.asarray(table, dtype=float)
    if table.shape != (2, 2):
        return None, None
    a, b, c, d = table[0, 0], table[0, 1], table[1, 0], table[1, 1]
    if min(a, b, c, d) == 0:
        a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5

    or_value = (a * d) / (b * c)
    se_log = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    log_or = np.log(or_value)
    return (
        float(or_value),
        (float(np.exp(log_or - Z_95 * se_log)), float(np.exp(log_or + Z_95 * se_log))),
    )


def risk_difference(table):
    """
    Difference in proportions (row 0 minus row 1) with an Agresti-Caffo CI.

    The naive Wald interval has poor coverage at small n and extreme
    proportions; Agresti-Caffo costs four extra lines and fixes it.
    """
    table = np.asarray(table, dtype=float)
    if table.shape != (2, 2):
        return None, None
    x1, n1 = table[0, 0], table[0].sum()
    x2, n2 = table[1, 0], table[1].sum()
    if n1 == 0 or n2 == 0:
        return None, None

    diff = float(x1 / n1 - x2 / n2)
    p1 = (x1 + 1) / (n1 + 2)
    p2 = (x2 + 1) / (n2 + 2)
    se = np.sqrt(p1 * (1 - p1) / (n1 + 2) + p2 * (1 - p2) / (n2 + 2))
    centre = p1 - p2
    return diff, (float(centre - Z_95 * se), float(centre + Z_95 * se))


# ── Correlation ─────────────────────────────────────────────────────────────

def fisher_z_ci(r: float, n: int):
    """Fisher z-transform CI for a correlation coefficient."""
    if n < 4 or not np.isfinite(r) or abs(r) >= 1.0:
        return None
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    return float(np.tanh(z - Z_95 * se)), float(np.tanh(z + Z_95 * se))
