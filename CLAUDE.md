# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ADA (Autonomous Data Analysis Agent) is a Streamlit app that takes an uploaded CSV and runs it
through a multi-agent LangGraph pipeline:

hypothesis generation → EDA → cleaning → ML training → SHAP explanation → hypothesis validation → report

It renders the results across tabs. All LLM calls go through Claude (Anthropic API). The README
is the public-facing description; keep it in sync when behaviour changes, especially the
"LLM proposes, code decides" table and the test counts.

## Commands

- Install: `pip install -r requirements-dev.txt` (runtime-only deps: `requirements.txt`)
- UI: `streamlit run app.py`
- CLI: `python main.py [csv] [--target COL]`. With no arguments it runs `data/sample.csv` with
  target `Survived`.
- Tests: `pytest`. This runs the whole pipeline offline in about 15 seconds and needs no API key.
  Run a single test with `pytest tests/test_pipeline.py::test_failed_node_is_retried_not_skipped`.
- Lint: `ruff check .` (config in `pyproject.toml`). CI (`.github/workflows/ci.yml`) runs both.

A real run needs `ANTHROPIC_API_KEY` in `.env` (see `.env.example`). `ADA_MODEL` overrides the
default model (`claude-haiku-4-5-20251001`).

## Core design rule: the LLM proposes, code decides

Anything with a correct answer is computed in code, and the LLM only proposes or narrates:

- Hypothesis verdicts
- Multiple-comparison correction
- Problem type
- Best-model selection
- Cleaning safety (never drop or impute the target, `mean`/`median` only on numeric columns)

Preserve this when changing agents. Concretely:

- `utils/stat_tests.py` must never import `utils.llm`. An AST test enforces it.
- LLM output that code consumes goes through `call_llm_json(prompt, system, model_cls)`. It
  validates against a pydantic model and re-prompts with the error on failure. Don't hand-parse
  JSON with `json.loads`.
- Set code-decided values *after* the LLM call, so the response can't override them. For
  example, `ml_agent.interpret_results` sets `best_model` after the call.

## Architecture

### LangGraph pipeline (`ada_graph.py`)

`run_ada(filepath, target_col, on_step)` builds and streams a `StateGraph` with 8 work nodes plus
`error_handler`. (`run_ada_v2` is a backwards-compatible alias.) The UI passes `on_step` to drive
its progress bar.

- **State.** `ADAState` (`utils/ada_state.py`) is a `TypedDict`. Nodes return
  `{**state, ...updates}` and append to `audit_trail` via `log()`.
- **Routing.** `NEXT_NODE` is the single ordering table.
- **Error path.** `check_error` sends any node with `state["error"]` set to `error_handler`.
  `error_handler` then picks one of three outcomes:
  - **Critical node** (`load_data`): the error stays set and the run ends.
  - **Retry:** it sets `retry_target`, and the failed node runs again, up to `MAX_RETRIES=2`
    times per node.
  - **Give up:** it clears the error and skips to the next node.

  `route_after_error` acts on that decision. Don't route the error handler through
  `check_error`. That was the old bug: retries silently skipped ahead, and a `CRITICAL` error
  looped until LangGraph's recursion limit.
- **Optional nodes.** `hypothesis`, `explain` and `validator` catch their own exceptions and
  return `{}`, so they never trigger retries. If `cleaning` is skipped, `ml` falls back to
  `raw_df`. The report node must tolerate `None` for any skipped stage (`state.get(x) or {}`).

### Agents (`agents/`)

Each agent is a module of plain functions with a `run_*_agent()` entry point.

- `hypothesis_agent` profiles columns and asks for 5 typed hypotheses (`HypothesisSet` in
  `utils/hypothesis_schema.py`).
- `eda_agent` computes `analyze_dataframe` stats; the LLM narrates them.
- `cleaning_agent` produces a `CleaningStrategy` (pydantic, `Literal` strategy names);
  `apply_cleaning_strategy` applies it with guards.
- `ml_agent` handles framing and training:
  - `decide_problem_type` uses dtype and cardinality.
  - `prepare_features` label-encodes, median-fills leftover NaN, and encodes classification
    targets to 0..k-1 for XGBoost.
  - `build_model` wraps linear models in a `StandardScaler` pipeline.
  - `split_data` holds the shared train/test split; `select_best_model` picks by
    `SELECTION_METRIC` (CV F1 or CV R²).
- `explain_agent` reuses `prepare_features`, `build_model` and `split_data` from `ml_agent`, so
  it explains exactly the trained model. It unwraps pipelines for SHAP and saves the plot under
  `outputs/`.
- `validator_agent` runs the tests from `utils/stat_tests.py` (effect sizes are in
  `utils/effect_sizes.py`) on `raw_df`, never the imputed frame. The LLM writes only
  `insight`/`caveat`.

### LLM layer (`utils/llm.py`)

`call_llm` and `call_llm_json` are the only way to reach Claude. The client is created lazily by
`get_client()`. `set_client()` swaps it, and tests install `tests/fakes.FakeAnthropic` this way.
That fake routes on a phrase unique to each agent's prompt. If you change one of those prompt
phrases, update `ROUTES` in `tests/fakes.py`.

### Frontend (`app.py`)

The flow is upload → preview → run. The upload is saved to `data/uploaded_<timestamp>.csv` and
deleted in `finally`. Eight result tabs map 1:1 to `ADAState` fields.

### Outputs

`outputs/` is gitignored and created on demand. It holds `ADA_v3_report_<ts>.txt`,
`ADA_v3_audit_<ts>.json` and `shap_summary.png`. `docs/images/shap_summary.png` is a committed
example for the README.
