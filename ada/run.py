"""Public entry point: run one investigation and persist everything it produced."""

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pandas as pd

from ada.config import Settings
from ada.graph import Runtime, build_graph
from ada.ledger import EvidenceLedger
from ada.tools.catalog import profile_dataset
from ada.tracing import Tracer

RUNS_DIR = "runs"


@dataclass
class InvestigationResult:
    question: str
    target: Optional[str]
    memo: str
    lint: dict
    ledger: EvidenceLedger
    tracer: Tracer
    lead_notes: list
    task_log: list
    stop_reason: str
    run_dir: str

    def to_dict(self) -> dict:
        return {
            "question": self.question, "target": self.target, "memo": self.memo,
            "lint": self.lint, "stop_reason": self.stop_reason, "run_dir": self.run_dir,
            "lead_notes": self.lead_notes, "task_log": self.task_log,
            **self.ledger.to_dict(), "trace": self.tracer.to_dict(),
        }


def investigate(data: pd.DataFrame | str, question: str, target: Optional[str] = None,
                settings: Optional[Settings] = None,
                on_event: Optional[Callable[[dict], None]] = None,
                run_dir: Optional[str] = None) -> InvestigationResult:
    settings = settings or Settings()
    df = pd.read_csv(data) if isinstance(data, str) else data
    if target is not None and target not in df.columns:
        raise ValueError(f"target column '{target}' is not in the dataset")

    run_dir = run_dir or os.path.join(RUNS_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    tracer = Tracer(max_cost_usd=settings.budget.max_cost_usd, on_event=on_event)
    rt = Runtime(df=df, question=question, profile=profile_dataset(df), run_dir=run_dir,
                 settings=settings, ledger=EvidenceLedger(alpha=settings.alpha),
                 tracer=tracer, target=target)
    tracer.event("system", "start", f"Investigating: {question}",
                 rows=len(df), columns=len(df.columns))

    graph = build_graph(rt)
    # Each round is lead + fan-out + critic; allow every round plus the report.
    state = graph.invoke({}, config={"recursion_limit": 10 * settings.budget.max_rounds + 10})

    report = state.get("report") or {"memo": "", "lint": {}}
    tracer.event("system", "finish", rt.stop_reason or "complete",
                 cost_usd=round(tracer.total_cost(), 4))
    result = InvestigationResult(
        question=question, target=rt.target, memo=report["memo"], lint=report["lint"],
        ledger=rt.ledger, tracer=tracer, lead_notes=state.get("lead_notes", []),
        task_log=state.get("task_log", []), stop_reason=rt.stop_reason, run_dir=run_dir,
    )
    _persist(result)
    return result


def _persist(result: InvestigationResult) -> None:
    with open(os.path.join(result.run_dir, "memo.md"), "w", encoding="utf-8") as f:
        f.write(result.memo)
    with open(os.path.join(result.run_dir, "investigation.json"), "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)
