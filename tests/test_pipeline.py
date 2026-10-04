"""
End-to-end runs of the LangGraph pipeline against data/sample.csv, with the
Anthropic client replaced by tests.fakes.FakeAnthropic. Real pandas, real
scikit-learn/XGBoost training, real SHAP, real statistical tests — only the
language model is simulated.
"""

import json
import os

import pytest

from ada_graph import check_error, route_after_error, run_ada
from tests.conftest import SAMPLE_CSV
from tests.fakes import FakeAnthropic
from utils import llm


def _actions(state):
    return [(e["agent"], e["action"]) for e in state["audit_trail"]]


def test_full_pipeline_runs_offline(fake_llm, in_tmp_dir):
    seen_nodes = []
    state = run_ada(SAMPLE_CSV, target_col="Survived",
                    on_step=lambda s: seen_nodes.append(s.get("current_node")))

    assert state["error"] is None
    assert state["final_report"].startswith("# Final Report")

    # on_step fires once per node, in pipeline order (initial state has no node).
    assert [n for n in seen_nodes if n] == [
        "load_data", "hypothesis", "eda", "cleaning",
        "ml", "explain", "validator", "report",
    ]

    # Cleaning refused to drop the target the fake LLM told it to drop.
    assert "Survived" in state["cleaned_df"].columns
    assert "Cabin" not in state["cleaned_df"].columns

    # The best model is the one with the highest CV F1, not the one the LLM named.
    ml = state["ml_results"]
    scores = {name: r["cv_f1_mean"] for name, r in ml["model_results"].items()}
    assert ml["problem_type"] == "classification"
    assert ml["interpretation"]["best_model"] == max(scores, key=scores.get)

    assert state["explain_results"]["explained_model"] == ml["interpretation"]["best_model"]
    assert (in_tmp_dir / "outputs" / "shap_summary.png").exists()

    verdicts = {v["id"]: v["verdict"] for v in state["validation_results"]["validation_results"]}
    assert verdicts["H1"] == "CONFIRMED"          # women survived at a far higher rate
    assert verdicts["H5"] == "NOT_TESTABLE"       # 'Income' is not a column

    # Report and audit trail are written, and the audit file is valid JSON
    # that includes the final "Pipeline complete!" entry.
    outputs = os.listdir(in_tmp_dir / "outputs")
    audit_file = next(f for f in outputs if f.startswith("ADA_v3_audit_"))
    audit = json.loads((in_tmp_dir / "outputs" / audit_file).read_text(encoding="utf-8"))
    assert audit[-1]["action"] == "Pipeline complete!"


def test_failed_node_is_retried_not_skipped(in_tmp_dir):
    fake = FakeAnthropic(fail_times={"deciding how to clean a dataset": 1})
    llm.set_client(fake)
    try:
        state = run_ada(SAMPLE_CSV, target_col="Survived")
    finally:
        llm.set_client(None)

    actions = _actions(state)
    assert ("ErrorHandler", "Retrying cleaning") in actions
    # The cleaning node ran a second time and succeeded.
    assert actions.count(("Cleaning", "Cleaning complete")) == 1
    assert fake.calls.count("deciding how to clean a dataset") == 2
    assert state["cleaning_strategy"]["reasoning"]
    assert state["final_report"]


def test_node_is_skipped_after_max_retries(in_tmp_dir):
    # call_llm_json makes 3 attempts per node run; 3 node runs = 9 failures.
    fake = FakeAnthropic(fail_times={"deciding how to clean a dataset": 99})
    llm.set_client(fake)
    try:
        state = run_ada(SAMPLE_CSV, target_col="Survived")
    finally:
        llm.set_client(None)

    actions = _actions(state)
    assert ("ErrorHandler", "Max retries reached for cleaning — skipping") in actions
    assert ("ML", "No cleaned data — using raw dataset") in actions
    assert state["ml_results"] is not None
    assert state["final_report"]


def test_missing_file_stops_cleanly(fake_llm, in_tmp_dir):
    # Before the routing fix this looped through the error handler until
    # LangGraph raised GraphRecursionError.
    state = run_ada(str(in_tmp_dir / "does_not_exist.csv"))

    assert state["error"].startswith("CRITICAL: load_data")
    assert fake_llm.calls == []


@pytest.mark.parametrize("state, expected", [
    ({"current_node": "eda", "error": None}, "cleaning"),
    ({"current_node": "eda", "error": "eda: boom"}, "error_handler"),
    ({"current_node": "report", "error": None}, "end"),
])
def test_check_error_routing(state, expected):
    assert check_error(state) == expected


@pytest.mark.parametrize("state, expected", [
    ({"current_node": "load_data", "error": "CRITICAL: x"}, "end"),
    ({"current_node": "ml", "error": None, "retry_target": "ml"}, "ml"),
    ({"current_node": "ml", "error": None, "retry_target": None}, "explain"),
])
def test_route_after_error(state, expected):
    assert route_after_error(state) == expected
