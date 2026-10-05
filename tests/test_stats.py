"""
Tests for the statistical validation path.

Everything here runs without an API key, a network call, or Streamlit — that is
the point of keeping the maths in modules that never import ada.llm.

Run with `python tests/test_stats.py` (pytest is not a project dependency,
but the test functions are pytest-compatible if you have it).
"""

import ast
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ada.stats import effect_sizes as es
from ada.stats.hypothesis_schema import ClaimType, Direction, TestSpec, Verdict
from ada.stats.stat_tests import (
    benjamini_hochberg,
    decide_verdict,
    holm_bonferroni,
    resolve,
    run_statistical_validation,
    run_test,
)


def approx(a, b, tol=1e-3):
    return abs(a - b) < tol


# ── The invariant that makes every verdict trustworthy ──────────────────────

def test_stat_tests_never_imports_the_llm():
    """
    If ada/stats/stat_tests.py could reach ada.llm, "the verdict is computed in
    code" would be a convention rather than a guarantee. Enforce it.
    """
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "ada", "stats", "stat_tests.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any("llm" in name for name in imported), \
        f"stat_tests.py must not import the LLM layer, found: {imported}"


# ── Effect sizes, against hand-computed values ──────────────────────────────

def test_hedges_g_known_value():
    a = [1, 2, 3, 4, 5]
    b = [3, 4, 5, 6, 7]
    # s_pooled = sqrt(2.5) = 1.58114, d = -2/1.58114 = -1.26491,
    # correction = 1 - 3/31 = 0.903226 -> g = -1.14249
    g, ci = es.hedges_g(a, b)
    assert approx(g, -1.14249), g
    assert ci[0] < g < ci[1]


def test_hedges_g_is_zero_for_identical_samples():
    g, _ = es.hedges_g([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    assert approx(g, 0.0)


def test_rank_biserial_sign_follows_argument_order():
    a = [10, 11, 12, 13, 14]
    b = [1, 2, 3, 4, 5]
    r, _ = es.rank_biserial(a, b)
    assert approx(r, 1.0), r          # a is entirely above b
    r_flipped, _ = es.rank_biserial(b, a)
    assert approx(r_flipped, -1.0), r_flipped


def test_odds_ratio_and_risk_difference_known_values():
    table = np.array([[30, 10], [12, 28]])
    or_value, or_ci = es.odds_ratio(table)
    assert approx(or_value, 7.0)      # (30*28)/(10*12)
    assert or_ci[0] < 7.0 < or_ci[1]

    rd, rd_ci = es.risk_difference(table)
    assert approx(rd, 0.45)           # 30/40 - 12/40
    assert rd_ci[0] < rd < rd_ci[1]


def test_odds_ratio_survives_a_zero_cell():
    """Haldane-Anscombe correction, or this is a division by zero."""
    or_value, or_ci = es.odds_ratio(np.array([[5, 0], [3, 9]]))
    assert or_value is not None and math.isfinite(or_value)
    assert all(math.isfinite(v) for v in or_ci)


def test_cramers_v_is_bias_corrected_downward():
    table = np.array([[10, 5], [5, 10]])
    v = es.cramers_v(table)
    naive = math.sqrt(
        __import__("scipy.stats", fromlist=["stats"]).chi2_contingency(
            table, correction=False).statistic / table.sum()
    )
    assert 0.0 <= v <= 1.0
    assert v < naive, "bias-corrected V should be below the naive value"


def test_fisher_z_ci_known_value():
    lo, hi = es.fisher_z_ci(0.5, 103)
    assert approx(lo, 0.3395, tol=1e-3), lo
    assert approx(hi, 0.6323, tol=1e-3), hi


# ── Multiple-comparison correction ──────────────────────────────────────────

def test_benjamini_hochberg_known_example():
    got = benjamini_hochberg([0.005, 0.011, 0.02, 0.04, 0.13])
    expected = [0.025, 0.0275, 0.03333, 0.05, 0.13]
    assert all(approx(a, b) for a, b in zip(got, expected)), got


def test_holm_known_example():
    got = holm_bonferroni([0.005, 0.011, 0.02, 0.04, 0.13])
    expected = [0.025, 0.044, 0.06, 0.08, 0.13]
    assert all(approx(a, b) for a, b in zip(got, expected)), got


def test_corrections_are_monotone_and_handle_ties():
    pvals = [0.01, 0.01, 0.04, 0.9, 0.9]
    for adjust in (benjamini_hochberg, holm_bonferroni):
        adjusted = adjust(pvals)
        assert all(0.0 <= q <= 1.0 for q in adjusted)
        # equal inputs must produce equal outputs
        assert approx(adjusted[0], adjusted[1])
        assert approx(adjusted[3], adjusted[4])
        # order must be preserved
        paired = sorted(zip(pvals, adjusted))
        assert all(paired[i][1] <= paired[i + 1][1] + 1e-12
                   for i in range(len(paired) - 1))
        # adjustment never makes a p-value more significant
        assert all(q >= p - 1e-12 for p, q in zip(pvals, adjusted))


def test_correction_family_excludes_untestable_hypotheses():
    """
    Correcting across NOT_TESTABLE slots would distort the q-values of the
    hypotheses that actually ran, so the family is the tests performed.
    """
    df = pd.DataFrame({
        "grp": ["a"] * 40 + ["b"] * 40,
        "val": list(np.random.default_rng(0).normal(0, 1, 40))
               + list(np.random.default_rng(1).normal(2, 1, 40)),
    })
    specs = {
        "H1": TestSpec(claim_type=ClaimType.GROUP_DIFFERENCE_NUMERIC,
                       direction=Direction.HIGHER_IN_FOCUS,
                       group_var="grp", outcome_var="val", focus_level="b"),
        "H2": TestSpec(claim_type=ClaimType.GROUP_DIFFERENCE_NUMERIC,
                       direction=Direction.HIGHER_IN_FOCUS,
                       group_var="grp", outcome_var="ghost", focus_level="b"),
    }
    records = run_statistical_validation(specs, df)
    assert records["H2"]["verdict"] == Verdict.NOT_TESTABLE
    assert records["H1"]["family_size"] == 1, "only H1 was actually tested"


# ── Column resolution ───────────────────────────────────────────────────────

def _frame():
    return pd.DataFrame({
        "grp": ["a", "b"] * 30,
        "num": list(range(60)),
        "binary": [0, 1] * 30,
        "many": [f"id_{i}" for i in range(60)],
    })


def test_resolution_rejects_a_hallucinated_column():
    spec = TestSpec(claim_type=ClaimType.GROUP_DIFFERENCE_NUMERIC,
                    direction=Direction.ANY, group_var="grp", outcome_var="nope")
    resolved, failure = resolve(spec, _frame())
    assert resolved is None and failure["reason"] == "unknown_column"


def test_resolution_matches_case_insensitively_but_not_fuzzily():
    spec = TestSpec(claim_type=ClaimType.GROUP_DIFFERENCE_NUMERIC,
                    direction=Direction.ANY, group_var="GRP", outcome_var="num")
    resolved, failure = resolve(spec, _frame())
    assert failure is None and resolved["columns"]["group_var"] == "grp"

    typo = TestSpec(claim_type=ClaimType.GROUP_DIFFERENCE_NUMERIC,
                    direction=Direction.ANY, group_var="grpp", outcome_var="num")
    resolved, failure = resolve(typo, _frame())
    assert resolved is None, "a near-miss must not be silently substituted"


def test_resolution_rejects_wrong_dtype_and_non_binary_outcome():
    wrong = TestSpec(claim_type=ClaimType.GROUP_DIFFERENCE_NUMERIC,
                     direction=Direction.ANY, group_var="grp", outcome_var="grp")
    assert resolve(wrong, _frame())[1]["reason"] == "wrong_dtype"

    not_binary = TestSpec(claim_type=ClaimType.PROPORTION_DIFFERENCE,
                          direction=Direction.ANY, group_var="grp", outcome_var="num")
    assert resolve(not_binary, _frame())[1]["reason"] == "not_binary"


def test_resolution_rejects_an_id_like_group_column():
    spec = TestSpec(claim_type=ClaimType.GROUP_DIFFERENCE_NUMERIC,
                    direction=Direction.ANY, group_var="many", outcome_var="num")
    assert resolve(spec, _frame())[1]["reason"] == "too_many_levels"


def test_resolution_rejects_an_unknown_focus_level():
    spec = TestSpec(claim_type=ClaimType.GROUP_DIFFERENCE_NUMERIC,
                    direction=Direction.HIGHER_IN_FOCUS,
                    group_var="grp", outcome_var="num", focus_level="z")
    assert resolve(spec, _frame())[1]["reason"] == "unknown_focus_level"


# ── The verdict table ───────────────────────────────────────────────────────

_DIRECTIONAL = TestSpec(claim_type=ClaimType.GROUP_DIFFERENCE_NUMERIC,
                        direction=Direction.HIGHER_IN_FOCUS,
                        group_var="grp", outcome_var="num", focus_level="a")


def _record(p_adj, effect, observed):
    return {"p_adjusted": p_adj, "p_value": p_adj, "effect_name": "hedges_g",
            "effect_value": effect, "observed_direction": observed}


def test_verdict_confirmed():
    out = decide_verdict(_DIRECTIONAL, _record(0.001, 0.8, 1))
    assert out["verdict"] == Verdict.CONFIRMED
    assert out["practically_significant"] is True


def test_verdict_rejected_when_direction_is_opposite():
    """A significant result running the wrong way rejects the claim."""
    out = decide_verdict(_DIRECTIONAL, _record(0.001, -0.8, -1))
    assert out["verdict"] == Verdict.REJECTED


def test_verdict_significant_but_trivial():
    """The n=50,000 case: real p-value, meaningless effect."""
    out = decide_verdict(_DIRECTIONAL, _record(0.0001, 0.02, 1))
    assert out["verdict"] == Verdict.SIGNIFICANT_BUT_TRIVIAL
    assert out["practically_significant"] is False


def test_verdict_not_supported():
    out = decide_verdict(_DIRECTIONAL, _record(0.4, 0.8, 1))
    assert out["verdict"] == Verdict.NOT_SUPPORTED


def test_verdict_passes_through_hints():
    for hint in (Verdict.NOT_TESTABLE, Verdict.INSUFFICIENT_DATA):
        out = decide_verdict(_DIRECTIONAL, {"verdict_hint": hint})
        assert out["verdict"] == hint


def test_nondirectional_claim_can_never_be_rejected():
    spec = TestSpec(claim_type=ClaimType.ASSOCIATION_CATEGORICAL,
                    direction=Direction.ANY, var_a="grp", var_b="binary")
    record = {"p_adjusted": 0.001, "p_value": 0.001, "effect_name": "cramers_v",
              "effect_value": 0.4, "observed_direction": 0}
    out = decide_verdict(spec, record)
    assert out["verdict"] == Verdict.CONFIRMED
    assert out["direction_match"] is None


# ── End-to-end against planted effects ──────────────────────────────────────

def test_planted_effect_is_found_and_disappears_when_shuffled():
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "grp": ["a"] * 100 + ["b"] * 100,
        "val": np.concatenate([rng.normal(0, 1, 100), rng.normal(1.0, 1, 100)]),
    })
    spec = TestSpec(claim_type=ClaimType.GROUP_DIFFERENCE_NUMERIC,
                    direction=Direction.HIGHER_IN_FOCUS,
                    group_var="grp", outcome_var="val", focus_level="b")

    found = run_statistical_validation({"H1": spec}, df)["H1"]
    assert found["verdict"] == Verdict.CONFIRMED, found["evidence"]
    assert found["observed_direction"] == 1

    shuffled = df.copy()
    shuffled["grp"] = rng.permutation(shuffled["grp"].to_numpy())
    gone = run_statistical_validation({"H1": spec}, shuffled)["H1"]
    assert gone["verdict"] == Verdict.NOT_SUPPORTED, gone["evidence"]


def test_wrong_direction_on_a_real_effect_is_rejected():
    rng = np.random.default_rng(11)
    df = pd.DataFrame({
        "grp": ["a"] * 100 + ["b"] * 100,
        "val": np.concatenate([rng.normal(0, 1, 100), rng.normal(1.5, 1, 100)]),
    })
    # b is genuinely higher, but the hypothesis claims a is.
    spec = TestSpec(claim_type=ClaimType.GROUP_DIFFERENCE_NUMERIC,
                    direction=Direction.HIGHER_IN_FOCUS,
                    group_var="grp", outcome_var="val", focus_level="a")
    result = run_statistical_validation({"H1": spec}, df)["H1"]
    assert result["verdict"] == Verdict.REJECTED, result["evidence"]


def test_complete_case_filtering_reports_rows_used():
    df = pd.DataFrame({
        "grp": ["a"] * 50 + ["b"] * 50,
        "val": [1.0] * 40 + [None] * 10 + list(np.linspace(2, 3, 50)),
    })
    spec = TestSpec(claim_type=ClaimType.GROUP_DIFFERENCE_NUMERIC,
                    direction=Direction.ANY, group_var="grp", outcome_var="val")
    result = run_test(spec, df)
    assert result["n_total"] == 100
    assert result["n_used"] == 90, "rows missing the tested column must be dropped"


def test_tiny_groups_are_insufficient_not_significant():
    df = pd.DataFrame({"grp": ["a"] * 3 + ["b"] * 30,
                       "val": list(range(3)) + list(range(30))})
    spec = TestSpec(claim_type=ClaimType.GROUP_DIFFERENCE_NUMERIC,
                    direction=Direction.ANY, group_var="grp", outcome_var="val")
    result = run_statistical_validation({"H1": spec}, df)["H1"]
    assert result["verdict"] == Verdict.INSUFFICIENT_DATA


def test_not_supported_evidence_does_not_claim_the_null_is_true():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"grp": ["a"] * 60 + ["b"] * 60,
                       "val": rng.normal(0, 1, 120)})
    spec = TestSpec(claim_type=ClaimType.GROUP_DIFFERENCE_NUMERIC,
                    direction=Direction.HIGHER_IN_FOCUS,
                    group_var="grp", outcome_var="val", focus_level="a")
    result = run_statistical_validation({"H1": spec}, df)["H1"]
    assert result["verdict"] == Verdict.NOT_SUPPORTED
    assert "not that the hypothesis is false" in result["evidence"]


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:
            failures += 1
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
