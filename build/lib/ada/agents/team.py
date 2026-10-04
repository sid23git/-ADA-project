"""The lead (planner), specialists, critic (reviewer) and reporter."""

import json
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from ada import llm
from ada.agents.base import AgentOutcome, run_tool_agent
from ada.agents.prompts import CRITIC_SYSTEM, LEAD_SYSTEM, REPORTER_SYSTEM, specialist_system
from ada.config import Settings
from ada.ledger import EvidenceLedger, Finding
from ada.tools.catalog import SPECIALIST_TOOLS
from ada.tools.registry import ToolContext
from ada.tracing import Tracer
from ada.verify import blocking_failures, check_finding, lint_memo

Specialist = Literal["data_quality", "statistician", "ml_engineer"]


# ── Lead ────────────────────────────────────────────────────────────────────

class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")
    specialist: Specialist
    objective: str


class LeadDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")
    target_column: Optional[str] = None
    reasoning: str
    tasks: list[Task] = Field(default_factory=list)
    done: bool = False


def _finding_line(f: Finding) -> dict:
    return {"id": f.id, "agent": f.agent, "status": f.status, "claim": f.claim,
            "critic": f.critic_reason or None}


def lead_plan(settings: Settings, tracer: Tracer, ledger: EvidenceLedger, *, question: str,
              profile: dict, target: Optional[str], round_no: int, task_log: list[dict],
              follow_ups: list[str]) -> LeadDecision:
    budget = settings.budget
    prompt = f"""QUESTION: {question}

TARGET COLUMN: {target or "not specified — choose one if the question implies one"}

DATASET PROFILE:
{json.dumps(profile)}

ROUND: {round_no} of {budget.max_rounds} (max {budget.max_tasks_per_round} tasks this round)
BUDGET USED: ${tracer.total_cost():.2f} of ${budget.max_cost_usd:.2f}

TASKS SO FAR:
{json.dumps(task_log, indent=1) if task_log else "none"}

FINDINGS SO FAR:
{json.dumps([_finding_line(f) for f in ledger.findings()], indent=1) if ledger.findings() else "none"}

CRITIC FOLLOW-UP REQUESTS:
{json.dumps(follow_ups) if follow_ups else "none"}"""

    decision = llm.call_json(settings.lead, agent="lead", tracer=tracer, system=LEAD_SYSTEM,
                             prompt=prompt, model_cls=LeadDecision)
    decision.tasks = decision.tasks[:budget.max_tasks_per_round]
    return decision


# ── Specialists ─────────────────────────────────────────────────────────────

def run_specialist(settings: Settings, ctx: ToolContext, *, specialist: str, objective: str,
                   question: str, profile: dict, context: list[str]) -> AgentOutcome:
    prompt = f"""INVESTIGATION QUESTION: {question}
TARGET COLUMN: {ctx.target or "none"}

YOUR OBJECTIVE:
{objective}

DATASET PROFILE:
{json.dumps(profile)}

ALREADY ESTABLISHED BY THE TEAM (do not repeat; build on it):
{chr(10).join(context) if context else "nothing yet"}"""

    tools = [t for t in SPECIALIST_TOOLS[specialist]
             if settings.allow_code_execution or t != "run_python"]
    return run_tool_agent(settings.specialist, name=specialist, system=specialist_system(specialist),
                          prompt=prompt, tool_names=tools, ctx=ctx,
                          max_steps=settings.budget.max_agent_steps)


# ── Critic ──────────────────────────────────────────────────────────────────

class Review(BaseModel):
    model_config = ConfigDict(extra="ignore")
    finding_id: str
    verdict: Literal["accept", "reject"]
    reason: str
    follow_up: Optional[str] = None


class ReviewBatch(BaseModel):
    model_config = ConfigDict(extra="ignore")
    reviews: list[Review]


def _evidence_digest(ledger: EvidenceLedger, eids: list[str], limit: int = 1200) -> dict:
    out = {}
    for eid in eids:
        ev = ledger.evidence(eid)
        if ev is not None:
            text = json.dumps({"tool": ev.tool, "args": ev.args, "result": ev.result}, default=str)
            out[eid] = text[:limit]
    return out


def critic_review(settings: Settings, tracer: Tracer, ledger: EvidenceLedger, *,
                  question: str, columns: set, round_no: int) -> list[str]:
    """
    Review every pending finding. Deterministic checks run first and their
    failures are final; the critic model only judges findings that pass them,
    so it can make a verdict stricter but never rescue a finding code rejected.
    Returns follow-up requests for the lead.
    """
    ledger.correct_family()
    pending = ledger.findings("pending")
    to_judge = []
    for f in pending:
        f.checks = check_finding(f, ledger, columns)
        failures = blocking_failures(f.checks)
        if failures:
            f.status = "rejected"
            f.critic_reason = "Failed verification: " + "; ".join(c["detail"] for c in failures)
            tracer.event("critic", "rejected", f"{f.id} (verification) {failures[0]['detail'][:120]}")
        else:
            to_judge.append(f)

    if not to_judge:
        return []

    context = [_finding_line(f) | {"claim": f.claim}
               for f in ledger.findings() if f.status == "accepted" or f.agent == "data_quality"]
    payload = [{
        "id": f.id, "agent": f.agent, "kind": f.kind, "claim": f.claim, "columns": f.columns,
        "implication": f.implication,
        "warnings": [c["detail"] for c in f.checks if not c["passed"]],
        "evidence": _evidence_digest(ledger, f.evidence_ids),
    } for f in to_judge]
    prompt = f"""QUESTION: {question}

FINDINGS TO REVIEW:
{json.dumps(payload, indent=1)}

OTHER FINDINGS ON RECORD (for context, e.g. data-quality issues):
{json.dumps(context, indent=1) if context else "none"}"""

    batch = llm.call_json(settings.critic, agent="critic", tracer=tracer, system=CRITIC_SYSTEM,
                          prompt=prompt, model_cls=ReviewBatch)
    by_id = {r.finding_id: r for r in batch.reviews}
    follow_ups = []
    for f in to_judge:
        review = by_id.get(f.id)
        if review is None:
            # Unreviewed is not accepted — silence from the critic is not approval.
            f.status, f.critic_reason = "rejected", "Critic returned no review for this finding."
            continue
        f.status = "accepted" if review.verdict == "accept" else "rejected"
        f.critic_reason = review.reason
        if review.follow_up:
            follow_ups.append(f"{f.id}: {review.follow_up}")
        tracer.event("critic", f.status, f"{f.id}: {review.reason[:140]}")
    return follow_ups


def final_reverification(ledger: EvidenceLedger, tracer: Tracer, columns: set) -> list[str]:
    """
    Tests run in later rounds enlarge the correction family, which can push an
    earlier test above alpha. Re-check every accepted finding against the final
    family before anything is reported.
    """
    ledger.correct_family()
    demoted = []
    for f in ledger.findings("accepted"):
        failures = blocking_failures(check_finding(f, ledger, columns))
        if failures:
            f.status = "rejected"
            f.critic_reason = "Lost support at final re-verification: " + failures[0]["detail"]
            demoted.append(f.id)
            tracer.event("verifier", "demoted", f"{f.id}: {failures[0]['detail'][:140]}")
    return demoted


# ── Reporter ────────────────────────────────────────────────────────────────

def _ruled_out(ledger: EvidenceLedger) -> list[dict]:
    out = []
    for ev in ledger.hypothesis_tests():
        if ev.result.get("verdict") in ("NOT_SUPPORTED", "SIGNIFICANT_BUT_TRIVIAL"):
            out.append({"evidence_id": ev.id, "comparison": ev.result["record"].get("comparison"),
                        "verdict": ev.result.get("verdict")})
    return out


def write_memo(settings: Settings, tracer: Tracer, ledger: EvidenceLedger, *, question: str) -> dict:
    accepted = ledger.findings("accepted")
    findings = [{"id": f.id, "kind": f.kind, "claim": f.claim, "implication": f.implication,
                 "confidence": f.confidence} for f in accepted]
    prompt = f"""QUESTION: {question}

ACCEPTED FINDINGS:
{json.dumps(findings, indent=1) if findings else "none — say plainly that the evidence was insufficient"}

NOT SUPPORTED BY THE DATA (you may cite these as [E#] in "What we ruled out"):
{json.dumps(_ruled_out(ledger), indent=1) or "none"}"""

    messages = [{"role": "user", "content": prompt}]
    response = llm.create(settings.reporter, agent="reporter", tracer=tracer,
                          system=REPORTER_SYSTEM, messages=messages)
    memo = llm.text_of(response)
    lint = lint_memo(memo, ledger)

    # One revision pass with the linter's complaints, then accept the result.
    if lint["ungrounded"] or lint["invalid_citations"]:
        tracer.event("reporter", "revise", f"{len(lint['ungrounded'])} ungrounded sentence(s)")
        messages += [
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": (
                "The grounding check failed. Fix ONLY these problems and return the full memo.\n"
                f"Sentences with numbers not supported by their citations: {json.dumps(lint['ungrounded'])}\n"
                f"Invalid citations: {lint['invalid_citations']}\n"
                "Cite the accepted finding that contains each number, or remove the number.")},
        ]
        response = llm.create(settings.reporter, agent="reporter", tracer=tracer,
                              system=REPORTER_SYSTEM, messages=messages)
        memo = llm.text_of(response)
        lint = lint_memo(memo, ledger)
    return {"memo": memo, "lint": lint}


def fallback_memo(ledger: EvidenceLedger, question: str, reason: str) -> dict:
    """Template memo used when the reporter cannot run (e.g. budget spent)."""
    lines = [f"# Findings: {question}", "",
             f"_Automatic summary — the reporting step did not run ({reason})._", ""]
    for f in ledger.findings("accepted"):
        lines.append(f"- {f.claim} [{f.id}]")
    if len(lines) == 4:
        lines.append("No findings were accepted before the run stopped.")
    memo = "\n".join(lines)
    return {"memo": memo, "lint": lint_memo(memo, ledger)}
