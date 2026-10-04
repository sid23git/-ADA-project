"""Tools, the evidence ledger, team-wide multiple-comparison correction and the sandbox."""

import json

import numpy as np
import pandas as pd
import pytest

from ada.ledger import EvidenceLedger
from ada.tools import sandbox
from ada.tools.catalog import TOOLS
from ada.tools.registry import ToolContext, execute
from ada.tracing import Tracer


@pytest.fixture
def ctx(titanic, tmp_path):
    return ToolContext(df=titanic, ledger=EvidenceLedger(), tracer=Tracer(),
                       run_dir=str(tmp_path), target="Survived", agent="test")


def test_every_tool_schema_is_self_contained():
    for tool in TOOLS.values():
        text = json.dumps(tool.schema())
        assert "$ref" not in text and "$defs" not in text, tool.name


def test_successful_call_is_recorded_as_citable_evidence(ctx):
    content, is_error = execute(TOOLS["group_summary"], {"group_by": "Sex", "metric": "Survived"}, ctx)
    result = json.loads(content)
    assert not is_error and result["evidence_id"] == "E1"
    assert ctx.ledger.evidence("E1").tool == "group_summary"
    rates = {g["group"]: g["mean"] for g in result["groups"]}
    assert rates["female"] == pytest.approx(0.742, abs=1e-3)


def test_bad_arguments_come_back_as_a_readable_error_not_an_exception(ctx):
    content, is_error = execute(TOOLS["run_hypothesis_test"], {"claim_type": "vibes"}, ctx)
    assert is_error and "claim_type" in content
    content, is_error = execute(TOOLS["describe_column"], {"column": "Income"}, ctx)
    assert is_error and "unknown column" in content
    assert ctx.ledger.all_evidence() == []          # failures are not evidence


def test_cannot_submit_findings_citing_evidence_that_does_not_exist(ctx):
    content, is_error = execute(TOOLS["submit_findings"], {"summary": "s", "findings": [{
        "claim": "x", "kind": "descriptive", "columns": [], "evidence_ids": ["E7"],
        "confidence": "low", "implication": "y"}]}, ctx)
    assert is_error and "E7" in content
    assert ctx.submitted == []


def test_correction_spans_every_test_the_team_runs():
    """
    A p = 0.02 result is significant alone, but not once the team has also run
    nine null tests — the ledger must re-adjust earlier verdicts as tests accumulate.
    """
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({f"noise{i}": rng.normal(size=n) for i in range(9)})
    y = rng.normal(size=n)
    # Build x with a sample correlation of exactly r = 0.135 (p ~ 0.02 at n = 300):
    # combine standardised y with noise made orthogonal to it.
    z_y = (y - y.mean()) / y.std()
    e = rng.normal(size=n)
    e = e - (e @ z_y) / (z_y @ z_y) * z_y
    e = (e - e.mean()) / e.std()
    r = 0.135
    df["y"] = y
    df["x"] = r * z_y + np.sqrt(1 - r ** 2) * e
    ctx = ToolContext(df=df, ledger=EvidenceLedger(), tracer=Tracer(), run_dir=".", agent="t")

    def test(var):
        content, _ = execute(TOOLS["run_hypothesis_test"], {
            "claim_type": "correlation_numeric", "direction": "positive", "var_a": var, "var_b": "y"}, ctx)
        return json.loads(content)

    first = test("x")
    assert first["p_value"] < 0.05 and first["verdict"] in ("CONFIRMED", "SIGNIFICANT_BUT_TRIVIAL")
    for i in range(9):
        test(f"noise{i}")
    ev = ctx.ledger.evidence(first["evidence_id"])
    assert ev.result["family_size"] == 10
    assert ev.result["p_adjusted"] > 0.05
    assert ev.result["verdict"] == "NOT_SUPPORTED"


def test_train_models_flags_a_leaked_column(ctx):
    ctx.df = ctx.df.assign(refund_issued=1 - ctx.df["Survived"])
    content, _ = execute(TOOLS["train_models"], {"exclude_columns": ["PassengerId", "Name", "Ticket"]}, ctx)
    result = json.loads(content)
    assert [s["feature"] for s in result["leakage_suspects"]] == ["refund_issued"]


# ── sandbox ─────────────────────────────────────────────────────────────────

def test_sandbox_runs_pandas_and_prints_the_last_expression(titanic):
    out = sandbox.run_python("df.groupby('Sex')['Survived'].mean().round(3)", titanic)
    assert out["ok"] and "0.742" in out["stdout"]


@pytest.mark.parametrize("code", [
    "import os",
    "open('secrets.txt')",
    "df.__class__.__bases__",
    "pd.read_csv('/etc/passwd')",
    "df.to_csv('out.csv')",
    "getattr(df, 'to_csv')('x')",
    "x = '__import__'",
])
def test_sandbox_blocks_escapes(code, titanic):
    with pytest.raises(sandbox.UnsafeCode):
        sandbox.run_python(code, titanic)


def test_sandbox_times_out_runaway_code(titanic):
    out = sandbox.run_python("while True:\n    pass", titanic, timeout=2)
    assert not out["ok"] and "timed out" in out["error"]
