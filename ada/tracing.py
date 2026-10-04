"""
Observability for one investigation: every LLM call and tool call becomes a
span with timing, tokens and cost, and every notable step becomes an event the
UI can stream.

Specialists run in parallel threads, so everything here is lock-protected.
"""

import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Optional

from ada.config import cost_usd


class RunHalted(RuntimeError):
    """The investigation cannot continue making model calls; wrap up with what exists."""


class BudgetExceeded(RunHalted):
    """Raised before an LLM call that would start after the cost budget is spent."""


@dataclass
class Span:
    kind: str                 # "llm" | "tool"
    agent: str
    name: str                 # model id for llm spans, tool name for tool spans
    started: float
    duration_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    ok: bool = True
    detail: str = ""


@dataclass
class Tracer:
    max_cost_usd: float = float("inf")
    on_event: Optional[Callable[[dict], None]] = None
    spans: list[Span] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    # Re-entrant: to_dict() holds the lock while calling summary().
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _t0: float = field(default_factory=time.time, repr=False)

    # ── events ──────────────────────────────────────────────────────────────

    def event(self, agent: str, kind: str, message: str, **data) -> None:
        entry = {"t": round(time.time() - self._t0, 2), "agent": agent,
                 "kind": kind, "message": message, **data}
        with self._lock:
            self.events.append(entry)
        if self.on_event is not None:
            # Observability must never break the thing it observes: a failing
            # UI or logging callback is recorded and ignored.
            try:
                self.on_event(entry)
            except Exception as exc:
                with self._lock:
                    self.events.append({"t": entry["t"], "agent": "system", "kind": "callback_error",
                                        "message": f"{type(exc).__name__}: {exc}"[:200]})

    # ── spans ───────────────────────────────────────────────────────────────

    def check_budget(self, agent: str) -> None:
        if self.total_cost() >= self.max_cost_usd:
            raise BudgetExceeded(
                f"{agent}: budget of ${self.max_cost_usd:.2f} spent "
                f"(${self.total_cost():.2f}) — stopping new LLM calls"
            )

    def record_llm(self, agent: str, model: str, started: float, usage, ok: bool = True,
                   detail: str = "") -> Span:
        def _get(name):
            return int(getattr(usage, name, 0) or 0) if usage is not None else 0

        span = Span(
            kind="llm", agent=agent, name=model, started=started - self._t0,
            duration_s=round(time.time() - started, 3),
            input_tokens=_get("input_tokens"),
            output_tokens=_get("output_tokens"),
            cache_read_tokens=_get("cache_read_input_tokens"),
            cache_write_tokens=_get("cache_creation_input_tokens"),
            ok=ok, detail=detail,
        )
        span.cost_usd = cost_usd(model, span.input_tokens, span.output_tokens,
                                 span.cache_read_tokens, span.cache_write_tokens)
        with self._lock:
            self.spans.append(span)
        return span

    def record_tool(self, agent: str, tool: str, started: float, ok: bool, detail: str = "") -> None:
        with self._lock:
            self.spans.append(Span(kind="tool", agent=agent, name=tool,
                                   started=started - self._t0,
                                   duration_s=round(time.time() - started, 3),
                                   ok=ok, detail=detail))

    # ── summaries ───────────────────────────────────────────────────────────

    def total_cost(self) -> float:
        with self._lock:
            return sum(s.cost_usd for s in self.spans)

    def summary(self) -> dict:
        with self._lock:
            spans = list(self.spans)
        llm = [s for s in spans if s.kind == "llm"]
        by_agent: dict[str, dict] = {}
        for s in spans:
            row = by_agent.setdefault(s.agent, {
                "llm_calls": 0, "tool_calls": 0, "input_tokens": 0, "output_tokens": 0,
                "cache_read_tokens": 0, "cost_usd": 0.0, "llm_seconds": 0.0,
            })
            if s.kind == "llm":
                row["llm_calls"] += 1
                row["input_tokens"] += s.input_tokens
                row["output_tokens"] += s.output_tokens
                row["cache_read_tokens"] += s.cache_read_tokens
                row["cost_usd"] += s.cost_usd
                row["llm_seconds"] += s.duration_s
            else:
                row["tool_calls"] += 1
        for row in by_agent.values():
            row["cost_usd"] = round(row["cost_usd"], 4)
            row["llm_seconds"] = round(row["llm_seconds"], 1)

        cacheable = sum(s.input_tokens + s.cache_read_tokens + s.cache_write_tokens for s in llm)
        return {
            "total_cost_usd": round(sum(s.cost_usd for s in llm), 4),
            "llm_calls": len(llm),
            "tool_calls": sum(1 for s in spans if s.kind == "tool"),
            "input_tokens": sum(s.input_tokens for s in llm),
            "output_tokens": sum(s.output_tokens for s in llm),
            "cache_hit_rate": round(sum(s.cache_read_tokens for s in llm) / cacheable, 3)
            if cacheable else 0.0,
            "wall_seconds": round(time.time() - self._t0, 1),
            "by_agent": by_agent,
        }

    def to_dict(self) -> dict:
        with self._lock:
            return {"spans": [asdict(s) for s in self.spans], "events": list(self.events),
                    "summary": self.summary()}
