import json
import os
from typing import Optional

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

load_dotenv()

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
MODEL = os.getenv("ADA_MODEL", DEFAULT_MODEL)

_client: Optional[anthropic.Anthropic] = None


def get_client() -> anthropic.Anthropic:
    """
    Build the client on first use rather than at import time, so modules that
    import this one (and the test suite, which swaps the client for a fake) never
    need an API key just to load.
    """
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def set_client(client) -> None:
    """Install a client — the real one, or a test double with the same shape."""
    global _client
    _client = client


def call_llm(prompt: str,
             system: str = "You are a helpful AI assistant.",
             max_tokens: int = 1024) -> str:
    """
    Central wrapper for all LLM calls in ADA.
    All agents call this function — so switching
    between Claude, GPT, or any other model only
    requires changing this ONE function.

    This is called the 'abstraction layer' pattern —
    your agents don't know or care which LLM is used.
    They just call call_llm() and get a response.
    """
    response = get_client().messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.content[0].text


def strip_json_fence(raw: str) -> str:
    """Strip a ``` or ```json fence if the model wrapped its JSON in one."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def call_llm_json(prompt: str,
                  system: str,
                  model_cls: type[BaseModel],
                  max_tokens: int = 2048,
                  retries: int = 2) -> BaseModel:
    """
    call_llm() for structured output: parse the response as JSON, validate it
    against a pydantic model, and on failure retry with the error fed back.

    The plain json.loads() path used elsewhere in ADA breaks outright on a
    single non-conforming response. Anything asking the model for a schema as
    detailed as a typed hypothesis needs the retry, and validating against a
    model is also what makes it safe to hand LLM output to code that indexes it.
    """
    last_error = None
    attempt_prompt = prompt

    for _ in range(retries + 1):
        raw = call_llm(prompt=attempt_prompt, system=system, max_tokens=max_tokens)
        try:
            payload = json.loads(strip_json_fence(raw))
        except json.JSONDecodeError as exc:
            last_error = f"That was not valid JSON ({exc})."
        else:
            try:
                return model_cls.model_validate(payload)
            except ValidationError as exc:
                last_error = f"That JSON did not match the required schema:\n{exc}"

        attempt_prompt = (
            f"{prompt}\n\n"
            f"YOUR PREVIOUS RESPONSE WAS REJECTED. {last_error}\n"
            f"Respond again with corrected JSON only — no prose, no code fence."
        )

    raise ValueError(
        f"{model_cls.__name__} could not be parsed after {retries + 1} attempts. "
        f"Last error: {last_error}"
    )
