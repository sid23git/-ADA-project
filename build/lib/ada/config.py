"""
Run-time configuration: model routing, budgets, pricing.

Model routing is deliberate. The roles that make judgement calls for the whole
investigation (lead, critic, reporter) run on the strongest model; the
specialists, which do many short tool-driven steps, run on a faster, cheaper
one. Everything is overridable from the environment.
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

LEAD_MODEL = os.getenv("ADA_LEAD_MODEL", "claude-opus-5-5")
WORKER_MODEL = os.getenv("ADA_WORKER_MODEL", "claude-sonnet-5-5")

# USD per million tokens: (input, output). Cache reads bill at 0.1x input and
# cache writes at 1.25x input.
PRICING = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
CACHE_READ_MULTIPLIER = 0.10
CACHE_WRITE_MULTIPLIER = 1.25

# Server-side refusal fallback (Claude API). Off switch for accounts or
# gateways that reject the beta.
USE_FALLBACKS = os.getenv("ADA_FALLBACKS", "1") == "1"

# Hosted public demo: visitors bring their own API key (the server's key is
# never used), budgets are capped, and agent-written code execution is off —
# on a shared server an uploaded CSV is untrusted input, and text inside it
# could try to steer an agent into misusing a code tool.
PUBLIC_DEMO = os.getenv("ADA_PUBLIC_DEMO", "0") == "1"
PUBLIC_MAX_COST_USD = 2.0
PUBLIC_MAX_ROWS = 200_000
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass(frozen=True)
class Budget:
    """Hard limits for one investigation. Exceeding one ends the run gracefully."""

    max_cost_usd: float = float(os.getenv("ADA_MAX_COST_USD", "3.0"))
    max_rounds: int = 3                 # lead -> specialists -> critic cycles
    max_tasks_per_round: int = 4
    max_agent_steps: int = 10           # tool-use turns per specialist task


@dataclass(frozen=True)
class AgentConfig:
    model: str
    effort: str = "medium"
    max_tokens: int = 16000


@dataclass(frozen=True)
class Settings:
    lead: AgentConfig = field(default_factory=lambda: AgentConfig(LEAD_MODEL, effort="high"))
    critic: AgentConfig = field(default_factory=lambda: AgentConfig(LEAD_MODEL, effort="high"))
    reporter: AgentConfig = field(default_factory=lambda: AgentConfig(LEAD_MODEL, effort="medium"))
    specialist: AgentConfig = field(default_factory=lambda: AgentConfig(WORKER_MODEL, effort="medium"))
    budget: Budget = field(default_factory=Budget)
    alpha: float = 0.05
    allow_code_execution: bool = not PUBLIC_DEMO


def cost_usd(model: str, input_tokens: int, output_tokens: int,
             cache_read: int = 0, cache_write: int = 0) -> float:
    price_in, price_out = PRICING.get(model, PRICING["claude-opus-5-5"])
    return (
        input_tokens * price_in
        + cache_read * price_in * CACHE_READ_MULTIPLIER
        + cache_write * price_in * CACHE_WRITE_MULTIPLIER
        + output_tokens * price_out
    ) / 1_000_000
