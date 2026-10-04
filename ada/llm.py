"""
The single choke point between ADA and Claude.

Every model call in the system goes through `create()`: it applies budget
checks, server-side refusal fallbacks, prompt caching and cost tracing in one
place. Tests swap the client for an offline fake via `set_client()`.
"""

import json
import time
from typing import Optional, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from ada.config import FALLBACK_BETA, USE_FALLBACKS, AgentConfig
from ada.tracing import RunHalted, Tracer

_client = None

T = TypeVar("T", bound=BaseModel)


class RefusalError(RuntimeError):
    pass


class LLMUnavailable(RunHalted):
    """
    The API failed in a way retrying this run will not fix: bad key, no
    credits, or an outage that outlasted the SDK's own retries. Raised as a
    RunHalted so the graph wraps up gracefully instead of crashing.
    """


def get_client():
    """Created on first use so importing ADA never requires credentials."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def set_client(client) -> None:
    global _client
    _client = client


def create(cfg: AgentConfig, *, agent: str, tracer: Tracer, system: str,
           messages: list, tools: Optional[list] = None):
    """
    One Messages API call, traced and budget-checked.

    The system prompt is sent as a cached block: each agent's system prompt and
    tool list are fixed for the whole investigation, so every call after an
    agent's first reads that prefix from cache.
    """
    tracer.check_budget(agent)

    params = {
        "model": cfg.model,
        "max_tokens": cfg.max_tokens,
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": messages,
        "output_config": {"effort": cfg.effort},
        # Automatic caching on the last block: in a tool loop each turn re-sends
        # the whole conversation, so every turn reads the previous turn's prefix
        # from cache instead of paying for it again.
        "cache_control": {"type": "ephemeral"},
    }
    if tools:
        params["tools"] = tools

    client = get_client()
    started = time.time()
    try:
        if USE_FALLBACKS:
            # On a policy decline the API re-runs the request on a fallback
            # model inside the same call, instead of the run just stopping.
            response = client.beta.messages.create(
                betas=[FALLBACK_BETA], extra_body={"fallbacks": "default"}, **params)
        else:
            response = client.messages.create(**params)
    except anthropic.APIError as exc:
        tracer.record_llm(agent, cfg.model, started, None, ok=False, detail=str(exc)[:300])
        tracer.event(agent, "error", f"API error: {exc}"[:300])
        raise LLMUnavailable(f"{agent}: Claude API unavailable — {exc}"[:400]) from exc

    tracer.record_llm(agent, getattr(response, "model", cfg.model) or cfg.model,
                      started, getattr(response, "usage", None))

    if response.stop_reason == "refusal":
        raise RefusalError(f"{agent}: request declined by the model")
    return response


def text_of(response) -> str:
    """Concatenate text blocks. Thinking and tool blocks are skipped."""
    return "".join(b.text for b in response.content if getattr(b, "type", None) == "text").strip()


def strip_json_fence(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def call_json(cfg: AgentConfig, *, agent: str, tracer: Tracer, system: str, prompt: str,
              model_cls: type[T], retries: int = 2) -> T:
    """
    Ask for JSON and validate it against a pydantic model. On failure the
    validation error is fed back and the model tries again, so one malformed
    response costs a retry rather than the run.
    """
    messages = [{"role": "user", "content": prompt}]
    last_error = ""
    for _ in range(retries + 1):
        response = create(cfg, agent=agent, tracer=tracer, system=system, messages=messages)
        raw = text_of(response)
        try:
            return model_cls.model_validate(json.loads(strip_json_fence(raw)))
        except json.JSONDecodeError as exc:
            last_error = f"That was not valid JSON ({exc})."
        except ValidationError as exc:
            last_error = f"That JSON did not match the required schema:\n{exc}"

        # Append-only history: the rejected answer stays, the correction follows.
        messages = messages + [
            {"role": "assistant", "content": response.content},
            {"role": "user", "content": f"YOUR PREVIOUS RESPONSE WAS REJECTED. {last_error}\n"
                                        "Respond again with corrected JSON only."},
        ]
    raise ValueError(f"{agent}: {model_cls.__name__} invalid after {retries + 1} attempts. "
                     f"Last error: {last_error}")
