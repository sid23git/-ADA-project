"""
The investigation as a LangGraph state machine.

    START -> lead --(tasks)--> specialist x N (parallel, via Send) -> critic -> lead -> ...
               \\--(done / out of rounds / out of budget)--> report -> END

The lead decides dynamically which specialists to run and when to stop; the
graph only enforces the protocol (every finding is reviewed before the lead sees
it again, the report always runs) and the limits.
"""

import operator
from dataclasses import dataclass, field
from typing import Annotated, Optional, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ada.agents.team import (
    critic_review,
    fallback_memo,
    final_reverification,
    lead_plan,
    run_specialist,
    write_memo,
)
from ada.config import Settings
from ada.ledger import EvidenceLedger
from ada.tools.registry import ToolContext
from ada.tracing import RunHalted, Tracer


@dataclass
class Runtime:
    """Live objects shared by every node. Kept out of graph state, which stays plain data."""

    df: pd.DataFrame
    question: str
    profile: dict
    run_dir: str
    settings: Settings
    ledger: EvidenceLedger
    tracer: Tracer
    target: Optional[str] = None
    stop_reason: str = ""
    halted: bool = False          # budget spent or API unavailable: skip to the report
    follow_ups: list = field(default_factory=list)


class InvestigationState(TypedDict, total=False):
    round: int
    tasks: list[dict]
    task_log: Annotated[list[dict], operator.add]
    lead_notes: Annotated[list[dict], operator.add]
    report: dict


def build_graph(rt: Runtime):
    budget = rt.settings.budget
    columns = set(map(str, rt.df.columns))

    def lead(state: InvestigationState) -> dict:
        round_no = state.get("round", 0) + 1
        if round_no > budget.max_rounds:
            rt.stop_reason = f"reached the {budget.max_rounds}-round limit"
            return {"tasks": [], "round": round_no}
        try:
            decision = lead_plan(rt.settings, rt.tracer, rt.ledger, question=rt.question,
                                 profile=rt.profile, target=rt.target, round_no=round_no,
                                 task_log=state.get("task_log", []), follow_ups=rt.follow_ups)
        except RunHalted as exc:
            rt.stop_reason, rt.halted = str(exc), True
            return {"tasks": [], "round": round_no}

        if rt.target is None and decision.target_column in columns:
            rt.target = decision.target_column
            rt.tracer.event("lead", "target", f"target column: {rt.target}")
        rt.follow_ups = []
        tasks = [] if decision.done else [t.model_dump() for t in decision.tasks]
        if not tasks:
            rt.stop_reason = rt.stop_reason or "lead judged the question answered"
        rt.tracer.event("lead", "plan", decision.reasoning,
                        tasks=[f"{t['specialist']}: {t['objective'][:90]}" for t in tasks])
        return {"tasks": tasks, "round": round_no,
                "lead_notes": [{"round": round_no, "reasoning": decision.reasoning,
                                "tasks": tasks, "done": decision.done}]}

    def dispatch(state: InvestigationState):
        tasks = state.get("tasks", [])
        if not tasks:
            return "report"
        return [Send("specialist", {"task": t, "round": state["round"]}) for t in tasks]

    def specialist(payload: dict) -> dict:
        task, round_no = payload["task"], payload["round"]
        name = task["specialist"]
        ctx = ToolContext(df=rt.df, ledger=rt.ledger, tracer=rt.tracer,
                          run_dir=rt.run_dir, target=rt.target, agent=name)
        established = [f"{f.id} ({f.agent}): {f.claim}" for f in rt.ledger.findings("accepted")]
        rt.tracer.event(name, "start", task["objective"][:200])
        entry = {"round": round_no, "specialist": name, "objective": task["objective"]}
        try:
            outcome = run_specialist(rt.settings, ctx, specialist=name, objective=task["objective"],
                                     question=rt.question, profile=rt.profile, context=established)
        except Exception as exc:  # one failed specialist must not sink the round
            if isinstance(exc, RunHalted):
                rt.stop_reason, rt.halted = rt.stop_reason or str(exc), True
            rt.tracer.event(name, "error", f"{type(exc).__name__}: {exc}"[:300])
            return {"task_log": [entry | {"error": f"{type(exc).__name__}: {exc}"[:300]}]}

        ids = []
        for f in outcome.findings:
            f.round = round_no
            ids.append(rt.ledger.add_finding(f).id)
        return {"task_log": [entry | {"summary": outcome.summary, "finding_ids": ids,
                                      "steps": outcome.steps, "stopped": outcome.stopped}]}

    def critic(state: InvestigationState) -> dict:
        if rt.halted:
            return {}
        try:
            rt.follow_ups = critic_review(rt.settings, rt.tracer, rt.ledger, question=rt.question,
                                          columns=columns, round_no=state.get("round", 0))
        except RunHalted as exc:
            # Unreviewed findings stay pending, so they never reach the memo.
            rt.stop_reason, rt.halted = str(exc), True
        return {}

    def after_critic(state: InvestigationState) -> str:
        return "report" if rt.halted else "lead"

    def report(state: InvestigationState) -> dict:
        final_reverification(rt.ledger, rt.tracer, columns)
        try:
            result = write_memo(rt.settings, rt.tracer, rt.ledger, question=rt.question)
        except RunHalted as exc:
            rt.stop_reason = rt.stop_reason or str(exc)
            result = fallback_memo(rt.ledger, rt.question, str(exc)[:200])
        rt.tracer.event("reporter", "done",
                        f"grounding {result['lint']['grounding_rate']:.0%} of numeric sentences")
        return {"report": result}

    graph = StateGraph(InvestigationState)
    graph.add_node("lead", lead)
    graph.add_node("specialist", specialist)
    graph.add_node("critic", critic)
    graph.add_node("report", report)
    graph.add_edge(START, "lead")
    graph.add_conditional_edges("lead", dispatch, ["specialist", "report"])
    graph.add_edge("specialist", "critic")
    graph.add_conditional_edges("critic", after_critic, ["lead", "report"])
    graph.add_edge("report", END)
    return graph.compile()
