"""Deterministic verification: grounding, citation and statistical-support checks."""

import pytest

from ada.ledger import EvidenceLedger, Finding
from ada.stats.hypothesis_schema import TestSpec
from ada.stats.stat_tests import run_test
from ada.verify import blocking_failures, check_finding, extract_numbers, lint_memo, ungrounded_numbers


def _ledger_with(result: dict, tool: str = "group_summary", args=None) -> tuple[EvidenceLedger, str]:
    ledger = EvidenceLedger()
    return ledger, ledger.add_evidence("statistician", tool, args or {}, result)


@pytest.mark.parametrize("text, expected", [
    ("rate was 0.742 vs 0.189", [0.742, 0.189]),
    ("42% churned", [42.0, 0.42]),
    ("q = 1.2e-58", [1.2e-58]),
    ("1,204 rows", [1204.0]),
    ("H1 and E12 and var2 are identifiers", []),
])
def test_extract_numbers(text, expected):
    assert extract_numbers(text) == pytest.approx(expected)


@pytest.mark.parametrize("claim, pool, ok", [
    ("Women survived at 74.2%", [0.742038], True),       # proportion -> percent
    ("Age is missing in 19.87% of rows", [19.87], True),  # percent -> percent
    ("odds ratio 12.35", [12.351], True),                 # rounding
    ("odds ratio 15.0", [12.351], False),                 # invented
    ("3 groups and the top 5", [], True),                 # small counts are not measurements
    ("observed in 2019", [], True),                       # years are labels
    # Found in a live run: labels the analyst chose are not measurements.
    ("OR 0.242 (95% CI 0.194-0.302)", [0.242, 0.194, 0.302], True),
    ("churn is 34.2% at 13-24 months and 12.5% at 49–72", [0.342, 0.125], True),
    ("month-to-month customers with tenure ≤12 churn at 57.2%", [0.572], True),
    ("the top 10% of scores capture 31.5% of churners", [0.315], True),
    ("churn is 0.8-0.9", [], False),                      # decimal ranges are still checked
    ("AUC fell by 0.018", [0.7767, 0.7588], False),       # mental arithmetic is not evidence
])
def test_grounding(claim, pool, ok):
    assert (ungrounded_numbers(claim, pool) == []) is ok


def test_hallucinated_number_fails_verification():
    ledger, eid = _ledger_with({"groups": [{"group": "female", "mean": 0.742}]})
    f = Finding(claim="Women survived at a rate of 0.81.", kind="descriptive",
                columns=["Sex"], evidence_ids=[eid])
    failures = blocking_failures(check_finding(f, ledger, {"Sex"}))
    assert [c["check"] for c in failures] == ["numbers_grounded"]


def test_unknown_citation_and_column_fail():
    ledger, _ = _ledger_with({"x": 1})
    f = Finding(claim="Something.", kind="descriptive", columns=["Nope"], evidence_ids=["E42"])
    failed = {c["check"] for c in blocking_failures(check_finding(f, ledger, {"Sex"}))}
    assert failed == {"citations_resolve", "columns_exist"}


def test_statistical_claim_needs_a_confirmed_test(titanic):
    ledger = EvidenceLedger()
    spec = TestSpec(claim_type="correlation_numeric", direction="negative", var_a="Age", var_b="Survived")
    record = run_test(spec, titanic)
    eid = ledger.add_evidence("statistician", "run_hypothesis_test", spec.model_dump(mode="json"),
                              {"status": "tested", "p_value": record["p_value"], "record": record})
    ledger.correct_family()
    assert ledger.test_verdict(eid) == "SIGNIFICANT_BUT_TRIVIAL"   # r = -0.077: real but negligible

    f = Finding(claim="Older passengers were less likely to survive.", kind="statistical",
                columns=["Age", "Survived"], evidence_ids=[eid])
    failed = [c["check"] for c in blocking_failures(check_finding(f, ledger))]
    assert failed == ["statistical_support"]


def test_causal_wording_is_a_warning_not_a_block():
    ledger, eid = _ledger_with({"value": 0.5})
    f = Finding(claim="Being female causes survival.", kind="descriptive", evidence_ids=[eid])
    checks = check_finding(f, ledger)
    causal = next(c for c in checks if c["check"] == "causal_language")
    assert not causal["passed"] and not causal["blocking"]
    assert blocking_failures(checks) == []


def test_memo_lint_keeps_citation_after_full_stop_with_its_sentence():
    ledger, eid = _ledger_with({"rate": 0.742})
    f = ledger.add_finding(Finding(claim="Women survived at 74.2%.", kind="descriptive",
                                   evidence_ids=[eid]))
    f.status = "accepted"
    memo = ("# Memo\nWomen survived at 74.2%. [F1]\nMen survived at 18.9% [F1].\n"
            "Rich people survived at 90% [F9].")
    lint = lint_memo(memo, ledger)
    assert lint["numeric_sentences"] == 3
    assert lint["grounded_sentences"] == 1
    assert lint["invalid_citations"] == ["F9"]
