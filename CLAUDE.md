# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ADA is a multi-agent AI data-science team. Given a CSV and a business question:

- A **lead** agent plans the investigation.
- **Specialists** (`data_quality`, `statistician`, `ml_engineer`) run in parallel and use real
  analysis tools.
- A **critic** reviews every finding.
- A **reporter** writes a decision memo.

Every number in a finding or in the memo must trace to an evidence item produced by a tool, and
code verifies this. Built on LangGraph and the Anthropic SDK (manual tool-use loop), with a
Streamlit UI.

## Commands

- Install: `pip install -r requirements-dev.txt`
- UI: `streamlit run app.py`
- CLI: `python -m ada data/sample.csv "question" --target Survived`. Add `--offline` to run with
  no API key.
- Tests: `pytest`. About 80 tests that need no API key. The full investigations run against
  `ada.offline.ScriptedClaude`.
- Lint: `ruff check .` (line length 120).
- Evals: `python -m evals.run [scenario ...] [--offline]`. Live mode calls Claude and costs money,
  capped by `--max-cost` per scenario. Results go to `evals/results/`.

Live runs need `ANTHROPIC_API_KEY`. Models are set by `ADA_LEAD_MODEL` (default
`claude-opus-5-5`, used for lead, critic and reporter) and `ADA_WORKER_MODEL` (default
`claude-sonnet-5-5`, used for specialists). `ADA_FALLBACKS=0` disables the server-side refusal
fallback beta.

## Architecture (`ada/`)

- `graph.py` is a LangGraph loop: `lead → Send() fan-out to specialist ×N (parallel threads) →
  critic → lead … → report`.
  - Live objects (df, ledger, tracer, settings) live on a `Runtime` closed over by the nodes.
    Graph state is plain data.
  - `Runtime.halted` (set on budget exhaustion) routes straight to `report`.
- `agents/base.py` holds the manual tool-use loop.
  - It appends `response.content` unchanged (thinking blocks included, append-only history).
  - All tool results for a turn go back in one user message.
  - The loop ends when `submit_findings` succeeds. There is one nudge, then a step budget.
- `agents/team.py` holds `lead_plan`, `run_specialist`, `critic_review`,
  `final_reverification` and `write_memo`; `agents/prompts.py` holds the system prompts.
  - The critic only judges findings that already passed `verify.check_finding`. Blocking
    failures are rejected in code, and the LLM can make a verdict stricter but never rescue a
    finding.
- `tools/catalog.py` defines each tool as a pydantic input model plus a pure function.
  - `tools/registry.execute` validates the input, runs the tool, records evidence (E#) and
    returns errors as `is_error` results instead of raising.
  - `run_hypothesis_test` takes `stats.hypothesis_schema.TestSpec` as its input schema.
- `ledger.py` holds `EvidenceLedger` (thread-safe) with evidence and findings.
  - `correct_family()` re-runs BH correction across *every* test the team has run and
    recomputes verdicts. Earlier verdicts can change as tests accumulate.
- `verify.py` holds the deterministic checks:
  - `check_finding`: citations resolve, numbers are grounded, statistical support, predictive
    support, columns exist, and a causal-language warning.
  - `lint_memo`: the memo grounding rate.
- `stats/` is the statistical engine. It must never import `ada.llm`; `tests/test_stats.py`
  checks this via AST.
- `ml/` handles modeling (problem type, CV, model selection, `detect_leakage`) and SHAP
  explanations. It is pure computation.
- `llm.py` is the only module that calls the API. `create()` handles budget checks, the
  fallback beta, `output_config.effort` and prompt caching.
  - `set_client()` swaps in `ScriptedClaude`.
  - `call_json()` validates against pydantic and retries with the error.
- `tracing.py` holds `Tracer`: spans, events, cost, and `BudgetExceeded`. Event callbacks are
  exception-isolated.
- `offline.py` holds `ScriptedClaude`, rule-based stand-ins keyed on each agent's system-prompt
  opening. If you change the first sentence of a system prompt, update its routing.
- `sandbox.py` (`run_python`) uses AST validation, restricted builtins and a subprocess timeout.
  It is defense in depth, not a security boundary.

`evals/` holds seeded synthetic scenarios with planted drivers, null columns and traps (leakage,
-999 sentinels, duplicates). `graders.py` scores structured findings (driver recall, false
discoveries, traps caught, grounding, cost). Scripted agents intentionally fail `null_world`:
that's the naive baseline.

Each run writes `runs/<timestamp>/` (gitignored) containing `memo.md`, `investigation.json` and
`shap_<target>.png`. The v3 linear pipeline is preserved at git tag `v3.1`.
