"""
The agent loop every specialist runs.

A manual loop rather than the SDK's tool runner, because the loop is where
ADA's guarantees live: every tool call is validated and written to the evidence
ledger, every model call is budget-checked and traced, and the agent is held to
a step budget and must end by submitting findings.
"""

from dataclasses import dataclass

from ada import llm
from ada.config import AgentConfig
from ada.ledger import Finding
from ada.tools.catalog import TOOLS
from ada.tools.registry import ToolContext, execute

NUDGE_FINISH = ("You have not called submit_findings. Call it now with your findings "
                "(an empty list is fine if you found nothing reliable).")
NUDGE_BUDGET = ("Step budget reached. Do not call any more analysis tools. Call submit_findings "
                "now with what your evidence already supports.")


@dataclass
class AgentOutcome:
    findings: list[Finding]
    summary: str
    steps: int
    stopped: str          # "submitted" | "no_submission" | "error: ..."


def run_tool_agent(cfg: AgentConfig, *, name: str, system: str, prompt: str,
                   tool_names: list[str], ctx: ToolContext, max_steps: int) -> AgentOutcome:
    tools = [TOOLS[t].schema() for t in tool_names]
    messages: list = [{"role": "user", "content": prompt}]
    ctx.agent = name
    nudged = False
    steps = 0

    while True:
        response = llm.create(cfg, agent=name, tracer=ctx.tracer, system=system,
                              messages=messages, tools=tools)
        # The full content goes back unchanged — thinking blocks included — so
        # the history stays append-only and the model's reasoning carries over.
        messages.append({"role": "assistant", "content": response.content})
        calls = [b for b in response.content if getattr(b, "type", None) == "tool_use"]

        for block in response.content:
            if getattr(block, "type", None) == "text" and block.text.strip():
                ctx.tracer.event(name, "thought", block.text.strip()[:300])

        if not calls:
            if nudged:
                return AgentOutcome([], "", steps, "no_submission")
            nudged = True
            messages.append({"role": "user", "content": NUDGE_FINISH})
            continue

        steps += 1
        results = []
        submitted = False
        for call in calls:
            if call.name not in tool_names:
                content, is_error = f"Tool {call.name} is not available to you.", True
            else:
                ctx.tracer.event(name, "tool_call", call.name, input=call.input)
                content, is_error = execute(TOOLS[call.name], call.input, ctx)
                submitted |= call.name == "submit_findings" and not is_error
            results.append({"type": "tool_result", "tool_use_id": call.id,
                            "content": content, "is_error": is_error})

        if submitted:
            findings = list(ctx.submitted)
            ctx.tracer.event(name, "submitted", f"{len(findings)} finding(s)")
            return AgentOutcome(findings, ctx.cache.get("summary", ""), steps, "submitted")

        # All results for one turn go back in a single user message.
        if steps >= max_steps:
            results.append({"type": "text", "text": NUDGE_BUDGET})
            nudged = True
        messages.append({"role": "user", "content": results})

        if steps > max_steps + 1:
            return AgentOutcome([], "", steps, "no_submission")
