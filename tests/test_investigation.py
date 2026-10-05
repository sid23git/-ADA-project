"""
End-to-end investigations through the real LangGraph orchestration, with the
model replaced by ScriptedClaude — real tools, ledger, verification and tracing.
"""

import json
import os

from ada.config import Budget, Settings
from ada.offline import ScriptedClaude
from ada.run import investigate
from evals.graders import grade, passed
from evals.scenarios import load
from tests.conftest import SAMPLE_CSV

QUESTION = "What determined who survived the Titanic?"


def test_full_investigation_runs_offline(scripted, tmp_path):
    result = investigate(SAMPLE_CSV, QUESTION, target="Survived", run_dir=str(tmp_path))

    # The lead dispatched all three specialists in round 1, then stopped.
    assert {t["specialist"] for t in result.task_log} == {"data_quality", "statistician", "ml_engineer"}
    assert all(t["stopped"] == "submitted" for t in result.task_log)
    assert [n["done"] for n in result.lead_notes] == [False, True]

    accepted = result.ledger.findings("accepted")
    assert {f.kind for f in accepted} >= {"data_quality", "statistical", "predictive"}
    assert all(f.evidence_ids for f in accepted)
    assert result.lint["grounding_rate"] == 1.0 and not result.lint["invalid_citations"]

    summary = result.tracer.summary()
    assert set(summary["by_agent"]) >= {"lead", "data_quality", "statistician", "ml_engineer",
                                        "critic", "reporter"}
    saved = json.loads((tmp_path / "investigation.json").read_text(encoding="utf-8"))
    assert saved["memo"] == result.memo and saved["evidence"]
    assert os.path.exists(tmp_path / "shap_Survived.png")


class HallucinatingStatistician(ScriptedClaude):
    """Submits one honest finding and one with an invented number."""

    def _statistician(self, prompt, messages):
        calls = super()._statistician(prompt, messages)
        if calls[0][0] == "submit_findings" and calls[0][1]["findings"]:
            honest = calls[0][1]["findings"][0]
            calls[0][1]["findings"].append(honest | {"claim": "Women were 9.99 times as likely to survive."})
        return calls


def test_invented_number_is_rejected_before_the_critic_sees_it(tmp_path):
    from ada import llm
    client = HallucinatingStatistician()
    llm.set_client(client)
    try:
        result = investigate(SAMPLE_CSV, QUESTION, target="Survived", run_dir=str(tmp_path))
    finally:
        llm.set_client(None)

    bad = next(f for f in result.ledger.findings() if "9.99" in f.claim)
    assert bad.status == "rejected"
    assert bad.critic_reason.startswith("Failed verification")
    assert "9.99" not in result.memo


class CrashingMLEngineer(ScriptedClaude):
    def _ml_engineer(self, prompt, messages):
        raise RuntimeError("model provider outage")


def test_one_failed_specialist_does_not_sink_the_investigation(tmp_path):
    from ada import llm
    llm.set_client(CrashingMLEngineer())
    try:
        result = investigate(SAMPLE_CSV, QUESTION, target="Survived", run_dir=str(tmp_path))
    finally:
        llm.set_client(None)

    failed = [t for t in result.task_log if t.get("error")]
    assert [t["specialist"] for t in failed] == ["ml_engineer"]
    assert result.ledger.findings("accepted")          # the others still delivered
    assert result.memo


def test_budget_exhaustion_ends_gracefully_with_a_fallback_memo(scripted, tmp_path):
    settings = Settings(budget=Budget(max_cost_usd=0.02))
    result = investigate(SAMPLE_CSV, QUESTION, target="Survived", settings=settings,
                         run_dir=str(tmp_path))
    assert "budget" in result.stop_reason
    assert result.memo                                   # never an empty result
    assert result.tracer.total_cost() < 0.2              # stopped soon after the limit


def test_lead_chooses_the_target_when_none_is_given(scripted, tmp_path):
    result = investigate(SAMPLE_CSV, QUESTION, run_dir=str(tmp_path))
    assert result.target == "Embarked"                  # scripted lead picks the last column


def test_eval_grader_scores_the_naive_baseline_on_the_leakage_trap(scripted, tmp_path):
    """
    The scripted ML engineer flags the leak, but the scripted statistician
    naively tests the most-correlated column — the leaked one — and reports it
    as a driver. The grader must credit the first and penalise the second.
    """
    scenario = load("leakage")
    result = investigate(scenario.df, scenario.question, target=scenario.target,
                         run_dir=str(tmp_path))
    score = grade(scenario, result)
    assert score["completed"] and score["traps_caught"] == {"target_leakage": True}
    assert score["false_discoveries"] == ["exit_survey_completed"]
    assert not passed(score)


class OutOfCreditClaude(ScriptedClaude):
    """Fails the way the real API did in a live eval run: mid-investigation."""

    def _critic(self, prompt):
        import anthropic
        import httpx
        raise anthropic.APIConnectionError(message="Your credit balance is too low",
                                           request=httpx.Request("POST", "https://api.anthropic.com"))


def test_api_failure_mid_run_ends_gracefully_with_a_memo(tmp_path):
    from ada import llm
    llm.set_client(OutOfCreditClaude())
    try:
        result = investigate(SAMPLE_CSV, QUESTION, target="Survived", run_dir=str(tmp_path))
    finally:
        llm.set_client(None)

    assert "unavailable" in result.stop_reason
    assert "No findings were accepted" in result.memo     # the reporter still delivers
    # Nothing unreviewed reaches the memo: the critic never ran, so nothing was accepted.
    assert result.ledger.findings("accepted") == []
    assert all(f.status == "pending" for f in result.ledger.findings())


class ExplodingClient:
    """Installed as the process-wide default: any call that reaches it is a leak."""

    def __getattr__(self, name):
        raise AssertionError("a run used the shared global client instead of its own")


def test_concurrent_runs_each_use_only_their_own_client(tmp_path):
    """Two visitors on the hosted demo, at the same time, with different keys."""
    import threading

    from ada import llm
    llm.set_client(ExplodingClient())
    a, b = ScriptedClaude(), ScriptedClaude()
    results = {}

    def run(name, client):
        results[name] = investigate(SAMPLE_CSV, QUESTION, target="Survived",
                                    run_dir=str(tmp_path / name), client=client)
    try:
        threads = [threading.Thread(target=run, args=("a", a)), threading.Thread(target=run, args=("b", b))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        llm.set_client(None)

    assert set(results) == {"a", "b"}
    assert a.calls and b.calls                      # each client served its own run...
    assert a.calls.count("lead") == b.calls.count("lead") == 2   # ...and only its own


def test_code_execution_can_be_disabled_for_untrusted_data(tmp_path, monkeypatch):
    from ada.agents import team
    seen = {}
    real = team.run_tool_agent

    def spy(cfg, **kw):
        seen[kw["name"]] = kw["tool_names"]
        return real(cfg, **kw)
    monkeypatch.setattr(team, "run_tool_agent", spy)

    investigate(SAMPLE_CSV, QUESTION, target="Survived", run_dir=str(tmp_path),
                settings=Settings(allow_code_execution=False), client=ScriptedClaude())
    assert seen and all("run_python" not in tools for tools in seen.values())
