# ADA: The Complete Project Guide

**From zero to interview-ready.** This guide explains ADA from the basics up. It assumes no
background in AI agents, statistics or machine learning, and builds each idea before using it.
By the end you should be able to explain every part of the project, defend every design decision,
and answer the questions an interviewer is likely to ask.

**Live demo:** https://lgla6kmbujefu7glenjaqt.streamlit.app/ · **Repo:** https://github.com/sid23git/ada-agent-team

---

## Contents

1. [ADA in one minute](#1-ada-in-one-minute)
2. [The concepts you need first](#2-the-concepts-you-need-first)
3. [The problem ADA solves](#3-the-problem-ada-solves)
4. [A full run, step by step](#4-a-full-run-step-by-step)
5. [Architecture and file map](#5-architecture-and-file-map)
6. [Each component in depth](#6-each-component-in-depth)
7. [Design decisions and why they were made](#7-design-decisions-and-why-they-were-made)
8. [How ADA is evaluated](#8-how-ada-is-evaluated)
9. [Tech stack: what and why](#9-tech-stack-what-and-why)
10. [Running the project yourself](#10-running-the-project-yourself)
11. [The numbers sheet](#11-the-numbers-sheet)
12. [Using ADA on your resume](#12-using-ada-on-your-resume)
13. [How to talk about ADA](#13-how-to-talk-about-ada)
14. [Interview questions and answers](#14-interview-questions-and-answers)
15. [Limitations and future work](#15-limitations-and-future-work)
16. [Glossary](#16-glossary)

---

## 1. ADA in one minute

You give ADA two things:

1. **A CSV file**, for example a table of telecom customers.
2. **A business question**, for example *"Which customers are most likely to churn, and what should
   the retention team focus on?"*

ADA then runs a **team of AI agents** that work like a real analytics team:

| Role | What it does | Model |
|---|---|---|
| 🧭 **Lead** | Reads the dataset summary, plans the investigation and assigns tasks | Claude Opus 5.5 |
| 🧹 **Data-quality engineer** | Hunts for problems in the data: duplicates, `-999` placeholders, missing values | Claude Sonnet 5.5 |
| 📐 **Statistician** | Runs real hypothesis tests with effect sizes | Claude Sonnet 5.5 |
| 🤖 **ML engineer** | Trains models, checks for leakage and explains the drivers with SHAP | Claude Sonnet 5.5 |
| 🕵️ **Critic** | Reviews every claim and rejects the weak ones | Claude Opus 5.5 |
| 📝 **Reporter** | Writes the final decision memo | Claude Opus 5.5 |

The output is a **decision memo**: a short report with a bottom line, key findings, recommended
actions, what was ruled out, and caveats.

**What makes ADA special:** every number in that memo traces back to something a tool actually
computed, and *code* checks this, not the AI. If an agent invents a number, the claim is rejected
automatically before anyone sees it.

> **One-sentence pitch:** ADA is an AI data-science team whose work you can audit. Every number
> is backed by computed evidence, every statistical claim survives multiple-comparison correction,
> and every finding is reviewed by a critic.

---

## 2. The concepts you need first

Skip any you already know. Each one comes up later.

### 2.1 AI and agent concepts

**Large Language Model (LLM).** A model such as Claude that reads text and writes text. It is very
good at reasoning and writing, but it can **hallucinate**: produce a confident, plausible answer
that is false. For data analysis that is dangerous. An LLM might write "churn is 34% for
month-to-month customers" without having computed anything.

**Tool use (function calling).** Instead of making up a number, the LLM can ask to run a function.
The program tells the model, "You have a tool called `group_summary` that takes a column name."
The model replies "call `group_summary(column='contract')`". *Your code* runs the function and
sends back the real result, and the model continues from there. The model decides **what** to
compute; code does the computing.

**Agent.** An LLM in a loop with tools. It thinks, calls a tool, reads the result, thinks again,
calls another tool, and keeps going until it decides it is done. In ADA, each specialist is an
agent.

**Multi-agent system.** Several agents with different roles that cooperate, like a team of
people. The advantages:
- each agent gets a focused job and a focused set of tools;
- independent agents can work **in parallel**;
- one agent (the critic) can check another's work.

**Orchestration.** The logic that decides which agent runs when, and what information passes
between them. ADA uses **LangGraph** for this.

**LangGraph.** A Python library for building agent workflows as a **graph**. Nodes are steps (lead,
specialist, critic, report) and edges say what runs next. It supports loops (lead → specialists
→ critic → back to lead) and **fan-out**: running the same node several times in parallel through
`Send()`.

**System prompt.** The standing instructions given to an agent, for example "You are the
skeptical reviewer...". It defines the agent's role and rules.

**Prompt caching.** Each agent's system prompt is identical on every call. Anthropic's API can
cache it, so repeat calls cost about one tenth as much for that part. ADA uses this to cut costs.

**Tokens and cost.** LLM APIs charge per token (roughly ¾ of a word) of input and output. ADA
tracks the tokens and dollar cost of every call and enforces a hard budget.

### 2.2 Statistics concepts

**Hypothesis test.** A formal check of whether a pattern in data is likely to be real or just
random noise. Example: "Do month-to-month customers churn more than two-year customers?"

**p-value.** If there were actually *no* difference, how surprising would data like ours be? A
small p-value (conventionally below 0.05) means "this would be quite surprising by chance alone".

**The multiple-comparisons problem.** At p < 0.05, if you run 20 tests on pure noise you expect
about one false "discovery". The more tests you run, the more fake findings you get. This is the
statistical core of **p-hacking**.

**Benjamini-Hochberg (BH) correction.** A standard fix. It adjusts all the p-values together,
taking into account how many tests were run, and controls the **false discovery rate**: the
expected share of your "discoveries" that are false. ADA applies BH across *every test the whole
team has run*, not just within each agent. That point is important, and Section 7 explains why.

**Effect size.** A p-value says whether a difference is likely real, not whether it is *big*. On a
huge dataset a tiny, useless difference can still give p < 0.05. Effect size measures magnitude.
ADA uses these:
- Hedges' g and rank-biserial correlation for comparing two groups;
- η² and ε² for comparing many groups;
- Cramér's V, odds ratio and risk difference for categorical data;
- Pearson's and Spearman's r for correlations.

**Confidence interval (CI).** A range that likely contains the true value, for example "odds
ratio 1.49 (95% CI 1.40-1.59)". It shows how precise an estimate is.

**Which test to use.** It depends on the data:

| Situation | Test if assumptions hold | Fallback test |
|---|---|---|
| Two groups, numeric outcome | Welch's t-test | Mann-Whitney U |
| Many groups, numeric outcome | ANOVA | Kruskal-Wallis |
| Categorical vs categorical | Chi-square (χ²) | Fisher's exact (small counts) |
| Numeric vs numeric | Pearson correlation | Spearman correlation |

ADA checks the assumptions in code before choosing: Shapiro-Wilk for normality, Levene for equal
variances, and sample size. The model never picks the test.

**Observational data and causation.** Business data is usually *observational*: nobody ran a
controlled experiment. You can show that two things are *associated*, but not that one *causes*
the other. ADA's verifier flags causal words ("causes", "leads to") and its prompts require
"associated with".

**Confounder.** A third variable that explains an apparent link. New customers might churn more
*because* they are on month-to-month contracts, not because they are new. A good analyst
checks whether an effect survives controlling for such variables.

### 2.3 Machine-learning concepts

**Classification vs regression.** Classification predicts a category (churn: yes or no).
Regression predicts a number (revenue). ADA decides which it is from the target column.

**Models used.** Logistic or Linear Regression (simple and interpretable), Random Forest, and
XGBoost (gradient-boosted trees, often the strongest on tables).

**Cross-validation (CV).** Rather than trusting one train/test split, the data is split into 5
"folds". The model is trained on 4 folds and tested on the 5th, five times over. The average score
is far more reliable. ADA uses 5-fold CV to pick the best model.

**Majority-class baseline.** If 75% of customers don't churn, a "model" that always predicts "no
churn" scores 75% accuracy. Any real model must beat that. ADA always reports the baseline.

**SHAP.** A method that explains *how much each feature pushed each prediction up or down*.
Averaging across all rows ranks the features by importance and shows their direction.

**Data leakage (target leakage).** A feature that contains information recorded *after* the
outcome. Example: `exit_survey_completed` is only filled in when a customer cancels, so it
"predicts" churn almost perfectly. A model using it looks brilliant in testing and is useless in
real life, because when you need the prediction, the survey hasn't happened yet. ADA flags any
single feature that predicts the target with AUC ≥ 0.95 on its own.

**Sentinel values.** Placeholder values that mean "unknown", such as `-999` for an unknown age.
If you don't catch them, they wreck averages: the mean age of patients becomes negative.

---

## 3. The problem ADA solves

Using an LLM as a data analyst is attractive and risky. LLM analysts tend to **fail quietly**:

| Failure | What it looks like | How ADA prevents it |
|---|---|---|
| **Invented numbers** | "Churn is 41% in the north region" when nothing computed 41% | Every number in a claim must appear in the cited evidence; code checks this |
| **p-hacking** | Many tests run, one comes up p = 0.03, reported as a discovery | BH correction across every test the whole team ran |
| **Trivial "significance"** | p < 0.001 on 50,000 rows for a negligible difference | Effect-size thresholds; a separate `SIGNIFICANT_BUT_TRIVIAL` verdict |
| **Leakage reported as insight** | "Exit survey completion is the #1 driver of churn!" | Automatic leakage detection; the ML engineer must exclude the column and retrain |
| **Dirty data** | Averages distorted by `-999` values or duplicate rows | A dedicated data-quality specialist; the critic rejects findings undermined by known data issues |
| **Overclaiming** | "Support calls cause churn" | Causal-language check plus a critic agent |
| **Unbounded cost** | An agent loops forever and burns money | Hard USD budget, step limits and round limits, with graceful wrap-up |

**ADA's central idea:** *the model proposes, code verifies.* Agents can be wrong. The system is
built so their mistakes get caught deterministically, without relying on the model to police
itself.

---

## 4. A full run, step by step

We'll follow the real churn example. Its memo is in
[`docs/example_run/churn_memo.md`](example_run/churn_memo.md) and the full trace is in
[`churn_investigation.json`](example_run/churn_investigation.json).

**Input:** 4,000 customers, with the columns `customer_id`, `tenure_months`, `contract`,
`monthly_charges`, `support_calls`, `region`, `payment_method`, `signup_channel` and `churned`.
**Question:** *"Which customers are most likely to churn, and what should the retention team focus
on?"*

### Step 0: Profile the data (code, no AI)
`profile_dataset()` builds a compact summary of the dataset: each column's type, missing
percentage and example values. This is what the lead sees. Agents never receive the raw CSV.
They only see summaries and tool results.

### Step 1: The lead plans (round 1)
The lead (Opus) receives the question, the profile and the budget. It replies with structured
JSON, which is validated against a pydantic model (`LeadDecision`):

```json
{"target_column": "churned",
 "reasoning": "We need to audit data quality, test the main hypotheses, and model churn...",
 "tasks": [
   {"specialist": "data_quality", "objective": "Audit for duplicates, sentinels, ID columns..."},
   {"specialist": "statistician", "objective": "Test whether month-to-month contracts have higher churn..."},
   {"specialist": "ml_engineer", "objective": "Train models on churned, check leakage, explain with SHAP..."}],
 "done": false}
```

### Step 2: Specialists run in parallel
LangGraph's `Send()` starts all three specialists **at the same time**. Each runs its own
**tool-use loop**:

```
model: "call group_summary(group_col='contract', metric='churned')"
code:  runs it → stores result as evidence E4 → returns {"evidence_id": "E4", ...}
model: "call run_hypothesis_test({...month-to-month higher churn...})"
code:  runs a chi-square test → stores as E7 → returns verdict CONFIRMED
model: "call submit_findings([{claim: 'Churn is 34.4% for month-to-month...', evidence_ids: ['E4']}])"
```

**Key rule:** every tool result is written to the **evidence ledger** and given an ID (E1, E2, …)
*before* the agent sees it. Agents finish by submitting **findings**, and every finding must cite
evidence IDs.

### Step 3: Code verifies every finding
Before the critic sees anything, `verify.check_finding()` runs on each finding:

1. **Citations resolve.** Does E4 actually exist?
2. **Numbers are grounded.** Does every number in the claim (34.4%, 12.8%) appear in E4? It
   allows for rounding and treats `34.4%` and `0.344` as the same value.
3. **Statistical support.** A "statistical" claim must cite a hypothesis test whose verdict is
   CONFIRMED *after* team-wide BH correction.
4. **Predictive support.** A "predictive" claim must cite a model or SHAP result.
5. **Columns exist.** Every column the finding names must be in the dataset.
6. **Causal language.** A warning only: "causes" or "leads to" gets flagged for the critic.

A finding that fails checks 1-5 is **rejected in code**. In the live churn runs, 4-6 findings per
run were rejected here, usually because a specialist did arithmetic in its head (for example,
subtracting two AUCs) instead of citing a computed value.

### Step 4: The critic reviews the survivors
The critic (Opus) sees each surviving finding with its evidence and judges what code can't:
- Is it overclaiming?
- Is the evidence misread?
- Is a data-quality issue undermining it?
- Could a confounder explain it?

It can **make a verdict stricter but never rescue a finding code rejected**. It can also send
**follow-up requests** to the lead.

A real rejection from a live run:
> *"Tenure under 24 months on a two-year contract is the normal state of any customer still in
> their first contract term, so flagging these 265 rows as potential errors misreads ordinary
> data."*

### Step 5: The lead decides again (rounds 2-3)
The lead sees what was accepted, what was rejected and why, and the critic's follow-ups. In the
churn run it asked, without being prompted, to *"check that the contract effect survives
controlling for tenure"*, which is a confounding check. When the question is answered, the lead
sets `done: true`. The run is capped at 3 rounds.

### Step 6: Final re-verification
New tests in later rounds enlarge the BH correction family, so a result that was significant in
round 1 might no longer be significant. `final_reverification()` re-runs the correction and the
checks on every accepted finding, and demotes any that lost support.

### Step 7: The reporter writes the memo
The reporter (Opus) writes the memo **only from accepted findings**, citing them as `[F9]`,
`[F27]` and so on. Then `lint_memo()` checks every sentence that contains a number: it must cite a
finding whose evidence contains that number. If any sentence fails, the reporter gets **one
revision pass** with the exact complaints.

### Step 8: Output
- `runs/<timestamp>/memo.md`: the decision memo.
- `runs/<timestamp>/investigation.json`: every finding, evidence item, verdict, LLM call, cost and
  timing.
- `runs/<timestamp>/shap_<target>.png`: the SHAP plot.

In the churn example, **all 24 numeric sentences** in the memo cite a finding and pass the
grounding check.

### The whole flow as a picture

```
             CSV + question
                   │
                   ▼
            ┌─────────────┐
     ┌─────▶│    Lead     │ plans tasks / decides "done"
     │      └──────┬──────┘
     │             │ Send() fan-out (parallel)
     │   ┌─────────┼──────────┐
     │   ▼         ▼          ▼
     │ Data      Stati-     ML
     │ quality   stician    engineer   ──tool calls──▶ Tools ──▶ Evidence ledger (E1, E2…)
     │   └─────────┼──────────┘
     │             │ findings citing E#
     │             ▼
     │   ┌───────────────────┐
     │   │ Code verification │──fail──▶ rejected (final)
     │   └─────────┬─────────┘
     │             │ pass
     │             ▼
     │      ┌─────────────┐
     └──────│   Critic    │ accept / reject + follow-ups
            └──────┬──────┘
                   │ lead says done, or limits reached
                   ▼
        Final re-verification (BH across all tests)
                   ▼
            ┌─────────────┐
            │  Reporter   │──▶ Memo grounding lint ──▶ decision memo
            └─────────────┘
```

---

## 5. Architecture and file map

```
ada/
├── __init__.py          Public API: investigate(), Settings, ...
├── __main__.py          CLI entry point (`ada data.csv "question"` / `python -m ada ...`)
├── run.py               investigate(): builds the run, invokes the graph, saves outputs
├── graph.py             LangGraph state machine: lead → specialists (parallel) → critic → … → report
├── config.py            Model routing, budgets, pricing, public-demo settings
├── llm.py               The ONLY module that calls the Claude API (budget, caching, retries, JSON)
├── ledger.py            Evidence ledger + findings + team-wide BH correction (thread-safe)
├── verify.py            Deterministic checks: check_finding() and lint_memo()
├── tracing.py           Spans, events, token and cost tracking, budget enforcement
├── offline.py           ScriptedClaude: a fake LLM so everything runs without an API key
├── agents/
│   ├── base.py          The manual tool-use loop every specialist runs
│   ├── team.py          lead_plan, run_specialist, critic_review, final_reverification, write_memo
│   └── prompts.py       System prompts for every role
├── tools/
│   ├── catalog.py       The 9 tools: pydantic input model + pure function each
│   ├── registry.py      execute(): validate → run → record evidence → return result or error
│   └── sandbox.py       run_python: AST validation + restricted builtins + subprocess timeout
├── stats/
│   ├── hypothesis_schema.py  TestSpec, ClaimType, Direction, Verdict (typed vocabulary)
│   ├── stat_tests.py         Test selection, 8 tests, BH/Holm correction, verdict logic
│   └── effect_sizes.py       Hedges' g, rank-biserial, η², ε², Cramér's V, OR, risk diff, bootstrap CIs
└── ml/
    ├── modeling.py      Problem type, feature prep, 3 models, 5-fold CV, selection, leakage detection
    └── explain.py       SHAP values, importance ranking, direction, plot
app.py                   Streamlit web UI
evals/                   Synthetic scenarios with planted ground truth + graders
tests/                   88 pytest tests, no API key needed
docs/                    Example run, deployment guide, this guide
Dockerfile               Container image
.github/workflows/ci.yml Lint + tests on every push and PR
```

**Size:** about 4,050 lines of application code in `ada/`, 900 lines of tests and 350 lines of
evaluation code.

### The layering rule
```
  UI / CLI  (app.py, __main__.py)
      │
  Orchestration  (graph.py, run.py)
      │
  Agents  (agents/*)  ── talk to Claude only through ──▶  llm.py
      │
  Tools  (tools/*)  ── record into ──▶  ledger.py  ◀── checked by ──  verify.py
      │
  Pure computation  (stats/*, ml/*)   ← never imports llm.py (a test enforces this)
```

Statistics and ML are **pure computation**: no AI can reach into them. `tests/test_stats.py`
parses the source code (AST) and fails if `stats/` ever imports `ada.llm`. That turns the
guarantee "every verdict is decided by code" into something the test suite enforces.

---

## 6. Each component in depth

### 6.1 The orchestration graph (`graph.py`)

- **Nodes:** `lead`, `specialist`, `critic`, `report`.
- **Edges:**
  - `START → lead`
  - `lead → specialist ×N`, or `lead → report` when there are no tasks
  - `specialist → critic`
  - `critic → lead`, or `critic → report` when the run is halted
  - `report → END`
- **Fan-out:** the `dispatch` function returns a list of `Send("specialist", {...})` objects.
  LangGraph runs them in parallel threads.
- **State vs runtime:** the graph state holds only plain data (round number, tasks, task log, lead
  notes, report). Live objects (the DataFrame, ledger, tracer, settings) live on a `Runtime`
  dataclass that the node functions close over. This keeps the state simple and serializable.
- **The task log uses `operator.add`**, so the results of parallel specialists are merged into one
  list instead of overwriting each other.
- **Fault isolation:** if one specialist crashes, its error is logged and the round continues
  without it.
- **Halting:** when the budget runs out or the API becomes unavailable, `Runtime.halted = True`
  and the graph jumps straight to `report`. You always get a memo, even if only a fallback one.

### 6.2 The agent loop (`agents/base.py`)

ADA writes its own tool-use loop instead of using a ready-made SDK runner, because **the loop is
where the guarantees live**:

1. Call Claude with the system prompt, the conversation history and the tool schemas.
2. Append the model's full response to the history **unchanged**, including its thinking blocks.
   The history is append-only.
3. Find every `tool_use` block. Run each call through `registry.execute`.
4. Send **all** tool results back in **one** user message, as the API expects.
5. If `submit_findings` succeeded, stop and return the findings.
6. If the model stopped without submitting, send one nudge: "Call submit_findings now."
7. After `max_agent_steps` (10) tool turns, tell it to finish; one more turn is allowed, then the
   loop stops regardless.

Agents can only call tools on their own list. A request for any other tool returns an error
result.

### 6.3 Tools (`tools/catalog.py`, `tools/registry.py`)

Each tool is a **pydantic input model plus a pure function**. The pydantic model does double duty:
it generates the JSON schema Claude sees, and it validates the arguments Claude sends back.

| Tool | Purpose | Used by |
|---|---|---|
| `check_data_quality` | Scan the dataset for duplicates, missing values, sentinels, ID columns | Data quality |
| `describe_column` | Summary stats, missingness, top values, possible sentinels | All |
| `group_summary` | Mean, median or rate of a metric per group | Data quality, Statistician |
| `correlations` | Rank columns by association with a target | Statistician, ML |
| `run_hypothesis_test` | A real statistical test with effect size, CI and verdict | Statistician |
| `train_models` | Train and cross-validate 3 models, select the best, detect leakage | ML |
| `explain_model` | SHAP top features and their direction, plus a plot | ML |
| `run_python` | Sandboxed pandas for anything the other tools can't do | Data quality, Statistician |
| `submit_findings` | Submit findings and finish | All |

**`registry.execute()` does the following:**
1. Validate the input. If it's invalid, return a **readable error** as an `is_error` result so
   the agent can fix its call. Nothing crashes.
2. Run the function. An exception is also returned as an error result.
3. Record the result in the ledger, which assigns an evidence ID.
4. Return `{"evidence_id": "E7", ...result}`, truncated to 6,000 characters.

### 6.4 The evidence ledger (`ledger.py`)

This is the team's shared memory.
- **`Evidence`** has an ID, the agent, the tool, the arguments, the result and a timestamp.
- **`Finding`** has a claim, a kind (statistical, predictive, descriptive or data_quality), the
  columns, the evidence IDs (at least one), a confidence, an implication and an effect (present,
  absent or not applicable). It also carries review fields: status, checks and the critic's
  reason.
- **Thread safety:** specialists run in parallel threads and all write to the ledger, so every
  operation takes a re-entrant lock (`threading.RLock`).
- **`correct_family()`** gathers every hypothesis test anyone has run, applies BH to all their
  p-values together, and **recomputes each verdict**. Verdicts can therefore change as the team
  runs more tests.

### 6.5 Verification (`verify.py`)

**Number extraction.** A regex finds numbers like `12`, `-0.31`, `1,204`, `3.5e-4` and `42%`. It
skips digits inside identifiers such as `E12`, `F3` or `var2`.

**What is *not* treated as a claim** (these rules came from real false alarms in live runs):
- confidence levels: "95% CI";
- bin edges: "13-36 months", "tenure ≤ 12";
- cut-offs: "top 10%";
- small whole numbers up to 10 ("3 segments") and years.

**Matching with rounding.** A claimed `0.78` matches the evidence `0.7767`, because rounding to
the decimals shown is allowed, within a 1% relative tolerance. A claimed `34.4%` matches either
`34.4` or `0.344`.

**`lint_memo()`** splits the memo into sentences, keeping a trailing citation like
"… 0.543. [F5]" attached to its sentence. For each sentence that contains a number, it collects
every number from the cited findings and their evidence and checks that the sentence's numbers
are covered. It returns a **grounding rate**: grounded numeric sentences divided by all numeric
sentences.

### 6.6 The statistics engine (`stats/`)

**A typed hypothesis.** The statistician doesn't write free text. It fills in a `TestSpec`:
- `claim_type`: group_difference_numeric, proportion_difference, correlation_numeric or
  association_categorical;
- `direction`: higher_in_focus, lower_in_focus, positive, negative or any;
- the variables involved and the focus level.

**The pipeline:** `resolve` → `run_test` → `correct` → `decide_verdict`.
- **Resolve** checks that the columns exist and have the right types. There is deliberately **no
  fuzzy matching**, so a hallucinated column `icome` is never silently mapped to `income`.
- **Run** picks the test from assumption checks (see the table in 2.2) and computes an effect size
  with a CI.
- **Correct** applies BH (Holm-Bonferroni is also available).
- **Decide verdict:**
  - `CONFIRMED`: significant after correction, the effect is above the practical threshold, and
    the direction matches the hypothesis.
  - `REJECTED`: significant, but in the opposite direction.
  - `SIGNIFICANT_BUT_TRIVIAL`: significant, but the effect is too small to matter. For example,
    Hedges' g < 0.2 or Cramér's V < 0.1.
  - `NOT_SUPPORTED`: not significant.
  - `INSUFFICIENT_DATA`: groups too small. Each group needs at least 5 rows and the test at least
    20 in total.
  - `NOT_TESTABLE`: an unknown column, wrong type, too many levels, and so on, with a
    machine-readable reason.

### 6.7 The ML engine (`ml/`)

- **Problem type:** a non-numeric target is classification; an integer target with ≤ 20 distinct
  values is classification; anything else is regression.
- **Models:** Logistic or Linear Regression (with the scaler *inside* a pipeline, so CV doesn't
  leak scaling statistics), Random Forest and XGBoost.
- **Selection:** by 5-fold cross-validated **F1** for classification (not accuracy, which rewards
  always predicting the majority class) or **R²** for regression.
- **Leakage detection:** any single feature with univariate AUC ≥ 0.95 (or |Spearman ρ| ≥ 0.95
  for regression) is flagged. A CV score ≥ 0.99 is flagged as "implausibly high for real data".
- **Baseline:** the majority-class accuracy is always reported, so "78.6% accuracy" can be
  compared with "75.4% by always guessing no".
- **SHAP:** ranks features by mean |SHAP|, gives each one's direction, and saves a plot.

### 6.8 The sandbox (`tools/sandbox.py`)

`run_python` lets agents write their own pandas code. It has three layers of protection:
1. **AST validation.** The code is parsed and rejected if it contains imports, `open`, `eval`,
   `exec`, `getattr`, any `_private` or `__dunder__` attribute access, file I/O methods
   (`read_csv`, `to_csv`, …) or strings containing `__`.
2. **Restricted builtins.** The code runs with a whitelist of safe builtins (`len`, `sum`,
   `sorted`, …) instead of the normal ones.
3. **Subprocess with a timeout.** The code runs in a separate Python process on a copy of the data,
   with a 20-second timeout and a stripped environment, so it can't see API keys.

The code is honest about the limits: this is **defense in depth, not a security boundary**. That's
why public demo mode turns `run_python` off completely.

### 6.9 The LLM layer (`llm.py`)

`llm.py` is **the only module that talks to the Claude API**. Every call goes through `create()`,
which:
- checks the budget *before* the call and raises `BudgetExceeded` if it's spent;
- marks the system prompt and the conversation prefix for **prompt caching**;
- sets `output_config.effort`: "high" for the lead and critic, "medium" otherwise;
- uses a **server-side refusal fallback**, so a policy decline is retried on a fallback model
  inside the same call;
- turns API errors (bad key, no credits, outage) into `LLMUnavailable`. The graph handles that by
  wrapping up gracefully rather than crashing;
- records tokens, cache usage, cost and latency on the tracer.

`call_json()` asks for JSON and validates it against a pydantic model. If validation fails, it
sends the **validation error back to the model** and retries, up to 2 times.

### 6.10 Tracing and cost (`tracing.py`, `config.py`)

- Every LLM call and every tool call becomes a **span** with agent, name, duration, tokens, cost
  and success.
- **Events** form the live feed in the UI: "lead plan", "E4 ← group_summary", "critic rejected
  F7", and so on.
- Event callbacks are wrapped in exception handling, so a buggy UI can't kill an agent.
- Cost is computed from a pricing table. Cache reads cost 0.1× the input price and cache writes
  1.25×.
- **Default budget:** $3 per run, 3 rounds, 4 tasks per round, 10 tool steps per specialist. The
  public demo caps runs at $2.

### 6.11 Offline mode (`offline.py`)

`ScriptedClaude` is a fake Claude client with the same interface as the real one. It decides
which agent is calling from the first sentence of the system prompt, then returns rule-based
responses: plans, tool calls, reviews and a memo. **Everything else is real**: the tools, ledger,
verifier, LangGraph orchestration and statistics. This means:
- the 88 tests run the *entire* multi-agent system in CI for free;
- the public demo has a free offline mode;
- the offline agents double as a **naive baseline** for the evals (see Section 8).

### 6.12 The web UI (`app.py`)

A Streamlit app. You pick a demo dataset or upload a CSV, type a question, and choose Offline or
Live mode (Live uses your own API key). It shows a **live agent feed** while the run is in
progress, then five tabs:
1. 📝 **Memo**: the decision memo with its citations.
2. 🧾 **Findings & review**: every finding, its checks and the critic's verdict.
3. 📎 **Evidence ledger**: every evidence item, and the hypothesis tests after team-wide
   correction.
4. 🧭 **Agent timeline**: a timeline in which you can see the parallel specialists overlap.
5. 💰 **Trace & cost**: cost and tokens per agent.

### 6.13 Public demo mode and packaging

`ADA_PUBLIC_DEMO=1` makes ADA safe to host for strangers:
- The server's API key is ignored. Visitors bring their own key, which is used only for their
  session.
- **Each run gets its own API client** (`Tracer.client`), so two simultaneous visitors can never
  share a key. Tests check this.
- Budgets are capped at $2, uploads at 25 MB / 200,000 rows, and `run_python` is disabled. An
  uploaded CSV is untrusted input, and text inside it could try to steer an agent (prompt
  injection).

**Packaging:**
- Installable with `pip install git+https://github.com/sid23git/ada-agent-team.git`, which
  provides an `ada` console command.
- Usable as a Python library through `from ada import investigate`.
- Shipped with a `Dockerfile`.
- Tested in GitHub Actions CI: ruff lint plus pytest on every push and PR.

---

## 7. Design decisions and why they were made

Interviewers love "why" questions. These are the main decisions and the reasoning behind each.

**1. Verification in code, not by another LLM.**
An LLM checking an LLM can make the same mistake. Number matching, citation checks and
statistical thresholds have correct answers, so they belong in deterministic code. The critic LLM
handles only what code *can't* judge: overclaiming, confounders and relevance. And the critic
**can't rescue** a finding code rejected.

**2. Team-wide multiple-comparison correction.**
With three agents testing in parallel, each one might run 5 tests and see "p = 0.03, significant".
Across the team, that's 15 tests, and some "discoveries" are now expected by chance. Correcting
per agent would hide this. ADA corrects across *every* test in the ledger and re-checks at the
end, because later tests can push earlier ones above the threshold.

**3. A separate `SIGNIFICANT_BUT_TRIVIAL` verdict.**
With large datasets almost everything is statistically significant. Reporting those as findings
is the most common way analyses overstate their results. ADA requires a practically meaningful
effect size as well.

**4. Opus for judgement, Sonnet for volume (cost-aware routing).**
The lead, critic and reporter make a few high-stakes calls, so they get the strongest model. The
specialists make many short tool-calling turns, so they get a faster, cheaper model. Live runs
cost $0.87-$1.00.

**5. A manual tool-use loop instead of a framework's agent runner.**
The loop is where the guarantees are enforced: validation, evidence recording, budget checks,
step limits, the nudge, and the required `submit_findings`. Owning the loop makes those
guarantees explicit and testable.

**6. Pydantic for everything structured.**
Tool inputs, lead decisions, critic reviews and findings are all pydantic models. One definition
gives you the schema the model sees *and* the validation of what it sends back. Invalid output
becomes a correctable error, not a crash.

**7. Plain-data graph state; live objects on a Runtime.**
LangGraph state should be simple and serializable. DataFrames, locks and clients don't belong in
it.

**8. Agents never see the raw data.**
They see a profile and tool results, capped at 6,000 characters each. This keeps prompts small and
cheap, limits what is sent to the API, and forces analysis through the tools, which means through
the evidence ledger.

**9. Fail gracefully, always produce a memo.**
Budget exhaustion, API outages and crashed specialists all lead to a report built from what was
accepted, or to a template fallback memo. Unreviewed findings stay "pending" and never reach the
memo.

**10. An offline fake for testing.**
Testing an LLM system against the real API is slow, costly and non-deterministic. `ScriptedClaude`
makes the full system testable in CI in about a minute for free.

**11. Per-run API clients.**
A global client would let concurrent users on the hosted demo share each other's keys. Attaching
the client to each run's tracer prevents that.

---

## 8. How ADA is evaluated

"It produced a nice-looking memo" isn't evidence that it works. ADA has an **evaluation suite with
planted ground truth**.

### 8.1 Synthetic scenarios (`evals/scenarios.py`)
Each dataset is generated from a known formula with a fixed random seed, so the grader knows
exactly which columns truly matter:

| Scenario | Rows | Real drivers | Noise columns | Trap |
|---|---|---|---|---|
| `churn` | 4,000 | contract, tenure_months, support_calls, monthly_charges | region, payment_method, signup_channel | none |
| `leakage` | 4,000 | same as churn | same | `exit_survey_completed` matches churn 97% of the time (recorded *after* cancellation) |
| `dirty_readmission` | ~3,090 | prior_admissions, diabetic, length_of_stay_days | ward, num_medications | `-999` ages (5%), 3% duplicated rows |
| `null_world` | 2,500 | **none** | everything | the only correct answer is "nothing predicts conversion" |

### 8.2 Grading (`evals/graders.py`)
The grader scores the **structured findings**, not the memo's wording, so scores don't depend on
phrasing:
- **Driver recall:** what fraction of the real drivers were found?
- **False discoveries:** were any noise columns (or the leaked column) claimed as drivers?
- **Traps caught:** was the leak, sentinel or duplicate explicitly flagged?
- **Grounding rate:** what share of the memo's numeric sentences are grounded?
- **Cost, LLM calls, wall time, and findings rejected by verification vs. by the critic.**
- **Pass** requires: the run completed, recall ≥ 50%, zero false discoveries, every trap caught,
  and grounding ≥ 90%.

### 8.3 Results

**The Claude agent team (live):**

| Run | Driver recall | False discoveries | Trap | Grounding | Rejected (code / critic) | Accepted / total | Tests | LLM calls | Time | Cost | Pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| churn #1 | 100% | 0 | — | 100% | 6 / 4 | 21 / 31 | 7 | 30 | 283 s | $1.00 | ✅ |
| churn #2 | 100% | 0 | — | 100% | 4 / 3 | 26 / 33 | 9 | 25 | 268 s | $0.96 | ✅ |
| leakage | 100% | 0 | ✅ caught and excluded | 100% | 5 / 3 | 22 / 30 | 11 | 27 | 249 s | $0.87 | ✅ |

**The naive baseline (offline scripted agents):**

| Scenario | Driver recall | Result |
|---|---|---|
| churn | 75% | ✅ pass |
| leakage | 75% | ❌ **reported the leaked column as a driver** |
| dirty_readmission | 100% | ✅ caught the sentinel and duplicates |
| null_world | — | ❌ **reported a noise column as a driver** |

**The takeaway:** the naive approach falls into exactly the traps the suite was designed around.
The Claude team recognized `exit_survey_completed` as post-outcome data, excluded it, and rebuilt
the model without it.

**Be honest about coverage:** `dirty_readmission` and `null_world` have **not yet been run live**.
Running them costs about $1 each with `python -m evals.run dirty_readmission null_world`.

### 8.4 The evals improved the system
The live runs exposed three bugs, each now fixed and covered by a regression test:
- the verifier was too strict, treating "95% CI" and "13-36 months" as claimed numbers;
- the grader misread correct null findings;
- the run crashed when the API ran out of credit.

Because graders are deterministic and runs are saved, `python -m evals.regrade runs/<dir>`
re-scores old live runs for free after a grader fix.

---

## 9. Tech stack: what and why

| Technology | Role | Why this choice |
|---|---|---|
| **Python 3.12** | Language | The data-science ecosystem |
| **Anthropic SDK / Claude API** | LLM calls | Tool use, prompt caching, effort control; strong reasoning models |
| **Claude Opus 5.5 / Sonnet 5.5** | Models | Opus for judgement roles, Sonnet for high-volume tool loops |
| **LangGraph** | Orchestration | Graph with loops, conditional edges and parallel `Send()` fan-out |
| **Pydantic v2** | Schemas and validation | One source of truth for tool schemas and output validation |
| **pandas / NumPy** | Data handling | Standard |
| **SciPy** | Statistical tests | Trusted implementations of t-test, Mann-Whitney, χ², Fisher, ANOVA, … |
| **scikit-learn** | ML models, CV, metrics | Standard, reliable |
| **XGBoost** | Gradient boosting | Often the strongest model on tabular data |
| **SHAP** | Explainability | The standard way to explain feature contributions |
| **Matplotlib** | SHAP plots | |
| **Streamlit** | Web UI | Fast to build, free hosting on Community Cloud |
| **pytest + ruff** | Testing and linting | 88 tests; lint enforced in CI |
| **GitHub Actions** | CI | Lint and tests on every push and PR |
| **Docker** | Deployment | Reproducible container |

---

## 10. Running the project yourself

**Try it with no install:** open the [live demo](https://lgla6kmbujefu7glenjaqt.streamlit.app/),
pick a dataset and choose **Offline** mode. It's free.

**Run locally:**
```bash
git clone https://github.com/sid23git/ada-agent-team.git
cd ada-agent-team
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements-dev.txt

# Web UI
streamlit run app.py

# CLI, free, no API key
python -m ada data/sample.csv "What determined who survived?" --target Survived --offline

# CLI with real Claude agents (needs a key; capped at $2)
set ANTHROPIC_API_KEY=sk-ant-...      # macOS/Linux: export ANTHROPIC_API_KEY=...
python -m ada data/sample.csv "What determined who survived?" --target Survived --max-cost 2

# Tests and lint
pytest
ruff check .

# Evals
python -m evals.run --offline          # free
python -m evals.run churn leakage      # live, about $1 per scenario
```

**Use it as a library:**
```python
from ada import investigate
result = investigate("sales.csv", "What drives repeat purchases?", target="repeat_buyer")
print(result.memo)
for f in result.ledger.findings("accepted"):
    print(f.id, f.claim, f.evidence_ids)
print(result.tracer.summary()["total_cost_usd"])
```

**Before an interview:** run it once offline and once live, and open `investigation.json`. Being
able to say "let me show you" is very persuasive.

---

## 11. The numbers sheet

Every number below comes from the code or from saved eval results. Know where each one comes from.

| Number | What it means | Source |
|---|---|---|
| **5 agent roles / 6 agents** | Lead, 3 specialists, critic, reporter | `agents/team.py` |
| **3 specialists in parallel** | LangGraph `Send()` fan-out | `graph.py` |
| **9 tools** | Analysis tools available to agents | `tools/catalog.py` |
| **8 statistical tests** | Welch t, Mann-Whitney U, ANOVA, Kruskal-Wallis, χ², Fisher, Pearson, Spearman | `stats/stat_tests.py` |
| **7+ effect-size measures** | Hedges' g, rank-biserial, η², ε², Cramér's V, odds ratio, risk difference (plus r) | `stats/effect_sizes.py` |
| **6 verdict types** | CONFIRMED … NOT_TESTABLE | `stats/hypothesis_schema.py` |
| **5 blocking checks + 1 warning** | Per-finding verification | `verify.py` |
| **3 ML models, 5-fold CV** | LR, RF, XGBoost | `ml/modeling.py` |
| **AUC ≥ 0.95** | Leakage-flag threshold | `ml/modeling.py` |
| **3-layer sandbox, 20 s timeout** | AST + builtins + subprocess | `tools/sandbox.py` |
| **100% driver recall** | All planted drivers found, 3 of 3 live runs | `evals/results/live_regraded.json` |
| **0 false discoveries** | No noise column claimed as a driver, live runs | same |
| **100% memo grounding** | Every numeric sentence grounded, live runs | same |
| **~27% of findings rejected** | 25 of 94 findings: 15 by code, 10 by the critic | same |
| **4-6 rejected by code per run** | Mostly in-head arithmetic or uncited numbers | same |
| **$0.87-$1.00 per run** | Live cost | same |
| **~4-5 minutes per run** | 249-283 s wall time | same |
| **25-30 LLM calls per run** | | same |
| **24 / 24 numeric sentences grounded** | The churn example memo | `docs/example_run/` |
| **75% → 100% recall** | Naive baseline vs. Claude team on churn/leakage | README / evals |
| **88 tests, no API key, ~1 min** | Full system tested in CI | `tests/` |
| **~4,000 lines** | Application code in `ada/` | `wc -l` |
| **$2 cap, 25 MB / 200k rows** | Public demo limits | `config.py`, `docs/DEPLOY.md` |

---

## 12. Using ADA on your resume

### Full entry
**ADA: Multi-Agent AI Data Science Team** | Python, LangGraph, Claude API, scikit-learn, XGBoost,
SHAP, Streamlit, Docker | [Live demo](https://lgla6kmbujefu7glenjaqt.streamlit.app/) ·
[GitHub](https://github.com/sid23git/ada-agent-team)

- Built a **5-role autonomous AI analytics team** (lead, 3 parallel specialists, critic,
  reporter) on LangGraph and the Claude API. It turns a raw CSV and a business question into a
  cited decision memo in **~4.5 minutes for under $1**.
- Removed hallucinated numbers with an **evidence ledger and deterministic verifier**: every
  figure must trace back to a computed tool result. It achieved **100% memo grounding and 0 false
  discoveries** in live runs and rejected **~27% of agent findings** (25 of 94) as ungrounded or
  overclaimed.
- Achieved **100% driver recall** on seeded benchmarks with planted ground truth. It caught a
  **target-leakage trap** that a naive baseline fell for; the baseline reached 75% recall and
  reported the leaked column as a driver.
- Engineered a statistical and ML engine with **8 hypothesis tests chosen by assumption checks**,
  effect sizes with 95% CIs, **team-wide Benjamini-Hochberg correction** against p-hacking,
  cross-validated model selection, leakage detection and SHAP.
- Shipped to production standards: a **sandboxed code-execution** tool, cost-aware Opus/Sonnet
  routing with prompt caching and hard budgets, per-call tracing, **88 CI tests needing no API
  key**, and a public multi-user demo shipped as a pip package and a Docker image.

### Short version
- Built a multi-agent AI data-science team (LangGraph + Claude) that analyzes any CSV and writes a
  decision memo in which every number is code-verified. It reached **100% recall, 0 false
  discoveries and 100% grounding** on ground-truth benchmarks at under $1 per run.

### Skills this project demonstrates
LLM agents and tool use · multi-agent orchestration · LangGraph · prompt engineering · structured
outputs · LLM evaluation · hallucination mitigation · statistical inference · multiple-testing
correction · ML model selection · explainable AI (SHAP) · data-quality auditing · concurrency and
thread safety · sandboxing · cost optimization · observability · testing LLM systems · CI/CD ·
deployment.

---

## 13. How to talk about ADA

### The 30-second pitch
> "ADA is a multi-agent AI data-science team. You give it a CSV and a business question. A lead
> agent plans, three specialists (data quality, statistics, ML) investigate in parallel with real
> analysis tools, a critic reviews everything, and a reporter writes a decision memo. The key idea
> is that the model proposes and code verifies: every number in the memo must trace back to a
> computed result, and code checks that. On benchmark datasets with planted ground truth, it found
> 100% of the real drivers with zero false discoveries and caught a data-leakage trap that a naive
> approach fell for, for about a dollar per run."

### The 2-minute walkthrough (STAR)
- **Situation.** LLMs are tempting as analysts but fail quietly: invented numbers, p-hacking,
  leaked columns presented as insights.
- **Task.** Build an AI analytics system whose output you can *trust and audit*, and *measure*
  whether it finds the truth.
- **Action.**
  1. I designed a LangGraph workflow: a lead plans, specialists fan out in parallel and run a
     manual tool-use loop, then a critic reviews, and the cycle repeats for up to 3 rounds.
  2. Every tool result goes into an evidence ledger with an ID, and findings must cite those IDs.
  3. A deterministic verifier checks citations, number grounding with rounding tolerance, and
     statistical support after team-wide Benjamini-Hochberg correction.
  4. The critic can only make verdicts stricter.
  5. I built an eval suite of synthetic datasets with planted drivers, noise columns and traps,
     and a naive baseline to compare against.
- **Result.**
  - 100% driver recall, 0 false discoveries and 100% memo grounding in live runs, at under $1
    each.
  - The leakage trap was caught where the baseline failed.
  - About 27% of agent claims were rejected before reaching the memo, which shows the verifier
    does real work.
  - The project is deployed publicly with 88 CI tests.

### Stories worth telling
1. **The critic catching a real mistake.** It rejected a "data error" finding: "tenure under 24
   months on a two-year contract is the normal state of any customer still in their first
   contract term."
2. **The lead testing for confounding unprompted.** It asked whether the contract effect survives
   controlling for tenure. It does: among customers in their first 12 months, churn is 57.2% for
   month-to-month vs. 19.9% for two-year.
3. **The eval finding bugs in the system itself.** The verifier treated "95% CI" as a claim, and
   the run crashed when credits ran out. Both are fixed and have regression tests.
4. **Agents doing math in their heads.** The verifier rejected claims where a specialist
   subtracted two numbers itself instead of citing a computed value. This is exactly the
   hallucination class ADA is built to stop.

---

## 14. Interview questions and answers

### Architecture and agents

**Q: Why multiple agents instead of one big prompt?**
A: Separation of concerns and checks and balances. Each specialist has a focused role, prompt and
toolset, which improves quality and keeps each prompt small. They can run in parallel, which cuts
wall time. Most importantly, a separate critic reviews the work. A single agent grading its own
work is weak oversight.

**Q: Why LangGraph?**
A: The workflow is a loop with conditional routing (continue or stop, halted or not) and parallel
fan-out. LangGraph models that directly with nodes, conditional edges and `Send()`, and handles
merging the parallel results through a reducer (`operator.add` on the task log).

**Q: Who decides the workflow, the code or the model?**
A: Both, at different levels. The **lead model** decides at run time which specialists to run,
what they should investigate, and when to stop. The **graph** enforces the protocol: every finding
is reviewed before the lead sees it again, the report always runs, and round, step and budget
limits are hard.

**Q: How do parallel agents avoid corrupting shared state?**
A: The evidence ledger and tracer are protected by re-entrant locks, and ID assignment happens
inside the lock. The graph state merges parallel outputs with a reducer instead of overwriting.

**Q: Why write your own tool-use loop?**
A: Because that's where the guarantees live. Every call is validated, recorded as evidence and
budget-checked. There is a step limit, a single nudge, and agents must finish with
`submit_findings`. Owning the loop makes all of that explicit and testable.

**Q: What happens if a specialist crashes?**
A: Its error is logged in the task log and the round continues with the others. If the budget
runs out or the API fails, a `RunHalted` exception sends the graph straight to the report, so you
always get a memo.

### Hallucination and verification

**Q: How exactly do you stop hallucinated numbers?**
A: Three layers:
1. Agents can only get numbers from tools, and every result is stored with an evidence ID.
2. Every finding must cite IDs, and code checks that every number in the claim appears in the
   cited evidence, allowing for rounding and percentage/proportion forms.
3. The memo is linted: every numeric sentence must cite an accepted finding containing that
   number, and the reporter gets one revision pass if any fail.

**Q: Why not have an LLM check the numbers?**
A: Checking whether a number appears in a result has a correct answer, so code does it reliably,
cheaply and the same way every time. An LLM checker can share the same blind spots. I use the
critic LLM only for judgement calls code can't make.

**Q: What does the grounding check deliberately ignore, and why?**
A: Labels rather than measurements: confidence levels ("95% CI"), bin edges ("13-36 months"),
cut-offs ("top 10%"), small counts ≤ 10, and years. Live runs showed that treating these as
claims caused false rejections.

**Q: Can the critic override the verifier?**
A: Only to make things stricter. Findings that fail blocking checks are rejected before the critic
sees them, and the critic can't rescue them. If the critic returns no review for a finding, it is
rejected: silence isn't approval.

**Q: What are the limits of the grounding check?**
A: It verifies that a number *appears* in the cited evidence, not that it's used correctly. For
example, a correct number attached to the wrong group would pass. That kind of semantic misreading
is what the critic is for. It's a deliberate split between what code and the model each check.

### Statistics

**Q: What is the multiple-comparisons problem, and how does ADA handle it?**
A: Every test at α = 0.05 has a 5% false-positive chance, so many tests produce false discoveries.
ADA applies Benjamini-Hochberg across every test the whole team has run, recomputes verdicts after
each round, and re-verifies everything before the report.

**Q: Why BH and not Bonferroni?**
A: Bonferroni controls the chance of *any* false positive, which is very conservative and loses
power when there are many tests. BH controls the *false discovery rate*, the expected share of
false positives among the discoveries, which suits exploratory analysis. Holm-Bonferroni is
implemented as an option.

**Q: Why correct across the whole team instead of per agent?**
A: The false-positive risk depends on the total number of tests run, not on who ran them. Three
agents each running 5 tests is 15 tests. Correcting per agent would understate the risk.

**Q: Can a finding's verdict change?**
A: Yes. More tests in later rounds enlarge the correction family, which can push an earlier
p-value above the threshold. Final re-verification catches and demotes those findings.

**Q: How does ADA choose a test?**
A: In code, from assumption checks. Two numeric groups use Welch's t-test if both groups have
n ≥ 30 (the central limit theorem applies) or pass a Shapiro-Wilk normality check, otherwise
Mann-Whitney U. Many groups use ANOVA if every group is normal and the variances are equal
(Levene), otherwise Kruskal-Wallis. Categorical data uses χ² (Yates-corrected), or Fisher's exact
test when any expected cell count is below 5. Correlations use Pearson or Spearman.

**Q: What does `SIGNIFICANT_BUT_TRIVIAL` mean and why have it?**
A: The p-value is significant but the effect size is below a practical threshold, for example
Hedges' g < 0.2. Large datasets make tiny effects significant, and reporting those is a common way
analyses overclaim.

**Q: Why are directional hypotheses preferred?**
A: "Month-to-month customers churn *more*" is more falsifiable and more useful than "contract and
churn are related". ADA checks the observed direction against the claimed one. A significant
result in the wrong direction gets `REJECTED`.

**Q: How do you handle causation?**
A: The data is observational, so ADA claims association only. The verifier flags causal wording,
the prompts require "associated with", and the critic rejects overclaims. The agents also test for
confounding, such as whether the contract effect holds within tenure bands.

### Machine learning

**Q: How does ADA detect target leakage?**
A: It checks each feature alone. If a single feature's AUC is ≥ 0.95 (or |Spearman ρ| ≥ 0.95 for
regression), it's flagged as likely recorded after the outcome. A CV score ≥ 0.99 is also flagged
as implausible. The ML engineer is instructed to exclude flagged columns, retrain, and report the
leak as its own finding.

**Q: Why select models by F1 rather than accuracy?**
A: With imbalanced classes, always predicting the majority class gets high accuracy. F1 takes into
account precision and recall on the positive class. The majority-class baseline is always reported
for context.

**Q: Why cross-validation?**
A: One 80/20 split is noisy, especially on small data. A 5-fold CV average is a more reliable way
to rank models. Scaling sits inside the pipeline so it's fit per fold, which avoids leaking test
statistics.

**Q: What does SHAP give you?**
A: Per-prediction feature contributions. Averaging the absolute values ranks the features, and the
sign shows the direction. In the churn run the top features were contract (0.648), tenure (0.620)
and support calls (0.412).

### Engineering, safety and cost

**Q: How is agent-written code made safe?**
A: Three layers: AST validation (no imports, file access, eval/exec or dunder access), restricted
builtins, and a separate subprocess with a 20-second timeout, a copy of the data and a stripped
environment. It's defense in depth, not a hard security boundary, so the public demo disables it.
A real multi-tenant deployment would use a container sandbox such as gVisor or Firecracker.

**Q: Why disable code execution in the public demo?**
A: Uploaded CSVs are untrusted. Text in a cell could be a prompt injection that tries to steer an
agent into misusing the code tool. Removing the tool removes that risk.

**Q: How do you control cost?**
A: Model routing (Opus only for judgement roles), prompt caching (cache reads are about 10% of the
input price), capped tool-result size, and hard limits: a USD budget checked before every call,
plus round, task and step limits. Running out of budget ends the run gracefully with a memo.

**Q: How did you prevent API keys leaking between users?**
A: Each run has its own client attached to its tracer, and `llm.create` uses that client. A global
client would let concurrent visitors share keys. Tests check this.

**Q: How do you test an LLM system without paying for API calls?**
A: `ScriptedClaude` is a fake client with the same interface as the real one. It returns
rule-based plans, tool calls and reviews. Everything else is real, so 88 tests exercise the full
system in CI in about a minute. They include adversarial cases, such as an agent inventing a
number that must be rejected before the critic sees it.

**Q: How do you know it actually works?**
A: An eval suite with planted ground truth: drivers, noise columns and traps. Graders score
structured findings for recall, false discoveries, traps caught, grounding and cost, and the
results are compared with a naive baseline. Saved runs can be re-graded for free.

### Reflection

**Q: What was the hardest part?**
A good answer: getting verification strict enough to catch fabrications but not so strict it
rejects honest claims. The first verifier rejected "95% CI" and "13-36 months" as ungrounded.
Live evals exposed this, and I added label-stripping rules and regression tests.

**Q: What would you do next?**
See Section 15. Run the remaining scenarios live and with repeated seeds for variance, add a
real container sandbox, support causal-inference methods, and handle larger datasets.

**Q: What did you learn?**
That with LLM systems, the most valuable engineering goes into what surrounds the model:
verification, evaluation, budgets, observability and graceful failure. And that you can't claim a
system works until you've measured it against ground truth.

---

## 15. Limitations and future work

Being upfront about limitations makes you more credible in an interview.

**Current limitations:**
- **Small live eval sample.** The live results come from 3 runs on 2 scenarios.
  `dirty_readmission` and `null_world` haven't been run live yet.
- **Grounding ≠ correctness.** The verifier checks that numbers *appear* in the evidence, not
  that they're interpreted correctly. The critic covers some of that gap.
- **Observational analysis only.** ADA finds associations. It doesn't do causal inference (no
  propensity scores, instrumental variables or experiments).
- **The sandbox is not a hard security boundary** (see 6.8).
- **Single CSV, in memory.** No database connectors or multi-table joins, and the hosted demo caps
  uploads at 200k rows.
- **Synthetic benchmarks.** Planted ground truth is necessary for measurement but simpler than
  messy real-world data.
- **LLM non-determinism.** Live runs vary, so results should be reported over several seeds.

**Future work:**
- Run all scenarios live with several seeds and report the mean and variance.
- Add harder scenarios: interaction effects, Simpson's paradox, time-based leakage.
- Use a real container sandbox, or the Claude API's server-side code execution tool.
- Add causal-inference tools and multi-table / SQL support.
- Add human-in-the-loop review in the UI (approve or reject findings).

---

## 16. Glossary

| Term | Meaning |
|---|---|
| **Agent** | An LLM in a loop with tools, working toward a goal |
| **AST** | Abstract Syntax Tree: code parsed into a tree so it can be inspected safely before running |
| **AUC** | Area under the ROC curve: 0.5 is random guessing, 1.0 is perfect separation |
| **Baseline (majority class)** | Accuracy from always predicting the most common class |
| **Benjamini-Hochberg (BH)** | A multiple-testing correction that controls the false discovery rate |
| **Confidence interval** | A range likely to contain the true value |
| **Confounder** | A third variable that explains an apparent relationship |
| **Critic** | ADA's reviewer agent; it can make verdicts stricter but never rescue a finding |
| **Cross-validation** | Repeated train/test splits for a reliable score |
| **Effect size** | How large a difference or relationship is, independent of sample size |
| **Evidence (E#)** | A recorded tool result in the ledger |
| **Evidence ledger** | The shared, thread-safe store of evidence and findings |
| **F1 score** | The harmonic mean of precision and recall |
| **Fan-out** | Running several tasks in parallel (LangGraph `Send()`) |
| **False discovery** | Claiming a noise column matters |
| **Finding (F#)** | A claim by a specialist that must cite evidence |
| **Grounding rate** | Share of numeric memo sentences whose numbers trace to cited evidence |
| **Hallucination** | An LLM producing confident but false content |
| **Hypothesis test** | A formal check of whether a pattern is likely real |
| **LangGraph** | A library for building agent workflows as graphs |
| **Leakage (target)** | A feature recorded after the outcome, which makes models look falsely accurate |
| **p-value** | How surprising the data would be if there were no real effect |
| **p-hacking** | Running many tests and reporting the ones that happen to be significant |
| **Prompt caching** | Reusing an already-processed prompt prefix to cut cost and latency |
| **Prompt injection** | Malicious text in data that tries to steer an AI agent |
| **Pydantic** | A Python library for typed data models and validation |
| **Recall (driver)** | Fraction of the true drivers the system found |
| **Sentinel value** | A placeholder like `-999` meaning "unknown" |
| **SHAP** | A method for explaining each feature's contribution to predictions |
| **Span** | One traced operation (an LLM or tool call) with timing, tokens and cost |
| **System prompt** | Standing instructions defining an agent's role |
| **Tool use** | An LLM requesting a function call that code executes |
| **Verdict** | ADA's code-decided outcome of a test: CONFIRMED, REJECTED, SIGNIFICANT_BUT_TRIVIAL, NOT_SUPPORTED, INSUFFICIENT_DATA, NOT_TESTABLE |
