"""
Tool plumbing shared by every agent.

A tool is a pydantic input model plus a pure function. The model is the single
source of truth: it generates the JSON schema Claude sees, and it validates the
arguments Claude sends back — a bad call returns an error the agent can read
and correct, instead of an exception that kills the run.

Every successful tool call is written to the evidence ledger before the agent
sees the result, and the result carries its evidence ID so the agent can cite it.
"""

import copy
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from pydantic import BaseModel, ValidationError

from ada.ledger import EvidenceLedger
from ada.tracing import Tracer

MAX_RESULT_CHARS = 6000


@dataclass
class ToolContext:
    df: pd.DataFrame
    ledger: EvidenceLedger
    tracer: Tracer
    run_dir: str
    target: Optional[str] = None
    agent: str = ""
    submitted: list = field(default_factory=list)   # findings from submit_findings
    cache: dict = field(default_factory=dict)       # e.g. trained-model results per target


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    fn: Callable[[ToolContext, BaseModel], dict]
    record_evidence: bool = True
    # Runs once the result has an evidence ID; returns the result the agent sees.
    after_record: Optional[Callable[[ToolContext, str], dict]] = None

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": inline_refs(self.input_model.model_json_schema()),
        }


def inline_refs(schema: dict) -> dict:
    """Resolve $ref/$defs so the tool schema is a single self-contained object."""
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                target = defs[node["$ref"].split("/")[-1]]
                merged = {**resolve(copy.deepcopy(target)),
                          **{k: v for k, v in node.items() if k != "$ref"}}
                return merged
            # pydantic's auto-generated "title" strings are noise in a tool
            # schema; a property that is literally named "title" is kept.
            return {k: resolve(v) for k, v in node.items()
                    if not (k == "title" and isinstance(v, str))}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    return resolve(schema)


def _compact(result: dict) -> str:
    text = json.dumps(result, default=str)
    if len(text) > MAX_RESULT_CHARS:
        text = text[:MAX_RESULT_CHARS] + ' ... [truncated]"'
    return text


def execute(tool: Tool, raw_input: dict, ctx: ToolContext) -> tuple[str, bool]:
    """Run one tool call. Returns (content for the tool_result block, is_error)."""
    started = time.time()
    try:
        args = tool.input_model.model_validate(raw_input)
    except ValidationError as exc:
        ctx.tracer.record_tool(ctx.agent, tool.name, started, ok=False, detail="invalid input")
        return f"Invalid arguments for {tool.name}: {exc}", True

    try:
        result = tool.fn(ctx, args)
    except Exception as exc:  # a tool failure is information for the agent, not a crash
        ctx.tracer.record_tool(ctx.agent, tool.name, started, ok=False, detail=str(exc)[:200])
        return f"{tool.name} failed: {type(exc).__name__}: {exc}", True

    ctx.tracer.record_tool(ctx.agent, tool.name, started, ok=True)
    if tool.record_evidence:
        eid = ctx.ledger.add_evidence(ctx.agent, tool.name,
                                      args.model_dump(mode="json"), result)
        ctx.tracer.event(ctx.agent, "evidence", f"{eid} ← {tool.name}", evidence_id=eid)
        if tool.after_record is not None:
            result = tool.after_record(ctx, eid)
        result = {"evidence_id": eid, **result}
    return _compact(result), False
