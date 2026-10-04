"""
ScriptedClaude: an offline stand-in for the Claude API.

It answers each agent with a fixed, rule-based policy — the lead dispatches
all three specialists, each specialist calls real tools and turns their results
into findings, the critic accepts, the reporter templates a memo. Everything
around the model is real: tools, evidence ledger, verification, the LangGraph
orchestration, tracing.

Uses: the test suite and CI (no API key, deterministic), and an offline demo
mode in the UI that is labelled as scripted. It is not an AI and never pretends
to be one — the rules below are deliberately simple.
"""

import json
import re
from types import SimpleNamespace

SCRIPTED_MODEL = "scripted-offline"


def _section(text: str, header: str) -> str:
    # A section runs until the next blank line followed by an UPPERCASE header,
    # which may carry a parenthesised note: "ALREADY ESTABLISHED (do not repeat):".
    match = re.search(rf"{re.escape(header)}:\s*\n?(.*?)(?:\n\n[A-Z][A-Z \-]+(?:\([^)]*\))?:|\Z)",
                      text, re.S)
    return match.group(1).strip() if match else ""


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict):
            parts.append(block.get("text") or block.get("content") or "")
        else:
            parts.append(getattr(block, "text", "") or "")
    return "\n".join(p for p in parts if isinstance(p, str))


def _tool_results(messages) -> list[dict]:
    """Every tool result so far in this conversation, parsed."""
    out = []
    for m in messages:
        if m["role"] != "user" or isinstance(m["content"], str):
            continue
        for block in m["content"]:
            if isinstance(block, dict) and block.get("type") == "tool_result" and not block.get("is_error"):
                try:
                    out.append(json.loads(block["content"]))
                except (json.JSONDecodeError, TypeError):
                    pass
    return out


def _turn(messages) -> int:
    return sum(1 for m in messages if m["role"] == "assistant")


class _Usage(SimpleNamespace):
    pass


class ScriptedClaude:
    """Duck-types `client.messages.create` and `client.beta.messages.create`."""

    def __init__(self):
        self.calls: list[str] = []
        self.messages = SimpleNamespace(create=self._create)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self._ids = 0

    # ── response helpers ────────────────────────────────────────────────────

    def _response(self, blocks, stop, prompt_chars):
        out_chars = sum(len(json.dumps(getattr(b, "input", None) or getattr(b, "text", ""))) for b in blocks)
        usage = _Usage(input_tokens=prompt_chars // 4, output_tokens=out_chars // 4,
                       cache_read_input_tokens=0, cache_creation_input_tokens=0)
        return SimpleNamespace(content=blocks, stop_reason=stop, usage=usage, model=SCRIPTED_MODEL)

    def _text(self, text, prompt_chars):
        return self._response([SimpleNamespace(type="text", text=text)], "end_turn", prompt_chars)

    def _tools(self, calls, prompt_chars, note=""):
        blocks = [SimpleNamespace(type="text", text=note)] if note else []
        for name, args in calls:
            self._ids += 1
            blocks.append(SimpleNamespace(type="tool_use", id=f"toolu_{self._ids}", name=name, input=args))
        return self._response(blocks, "tool_use", prompt_chars)

    # ── dispatch ────────────────────────────────────────────────────────────

    def _create(self, *, model, system, messages, tools=None, **_):
        system_text = _text_of(system)
        prompt = _text_of(messages[0]["content"])
        size = len(system_text) + sum(len(_text_of(m["content"])) for m in messages)

        if system_text.startswith("You are the lead data scientist"):
            self.calls.append("lead")
            return self._text(json.dumps(self._lead(prompt)), size)
        if system_text.startswith("You are the skeptical reviewer"):
            self.calls.append("critic")
            return self._text(json.dumps(self._critic(prompt)), size)
        if system_text.startswith("You are the reporting lead"):
            self.calls.append("reporter")
            return self._text(self._reporter(prompt), size)
        for role, marker in (("data_quality", "Data Quality Engineer"),
                             ("statistician", "Statistician"), ("ml_engineer", "ML Engineer")):
            if marker in system_text:
                self.calls.append(role)
                calls = getattr(self, f"_{role}")(prompt, messages)
                return self._tools(calls, size)
        raise AssertionError("ScriptedClaude: unrecognised agent")

    # ── policies ────────────────────────────────────────────────────────────

    @staticmethod
    def _context(prompt):
        profile = json.loads(_section(prompt, "DATASET PROFILE") or "{}")
        target = _section(prompt, "TARGET COLUMN").split("\n")[0].strip()
        target = None if target.startswith(("none", "not specified")) else target
        return profile, target

    def _lead(self, prompt):
        profile, target = self._context(prompt)
        round_no = int(re.search(r"ROUND: (\d+)", prompt).group(1))
        target = target or (profile["columns"][-1]["name"] if profile.get("columns") else None)
        if round_no > 1:
            return {"target_column": target, "reasoning": "All specialists reported; findings "
                    "reviewed. Further rounds would not change the recommendation.",
                    "tasks": [], "done": True}
        tasks = [{"specialist": "data_quality", "objective": "Audit the dataset for issues that "
                  "would distort the analysis."},
                 {"specialist": "statistician", "objective": f"Test which columns are associated "
                  f"with {target}."}]
        if target:
            tasks.append({"specialist": "ml_engineer", "objective": f"Model {target}, check for "
                          f"leakage and explain the drivers."})
        return {"target_column": target, "reasoning": "Round 1: audit data quality, test the main "
                "associations and model the target in parallel.", "tasks": tasks, "done": False}

    def _data_quality(self, prompt, messages):
        if _turn(messages) == 0:
            return [("check_data_quality", {})]
        result = _tool_results(messages)[0]
        findings = []
        for issue in result["issues"]:
            col = issue.get("column")
            if issue["type"] == "sentinel_value":
                claim = f"'{col}' contains the placeholder value {issue['value']} in {issue['count']} rows."
                action = f"Treat {issue['value']} in '{col}' as missing before any analysis."
            elif issue["type"] == "id_like_column":
                claim = f"'{col}' is unique on every row and carries no signal."
                action = f"Exclude '{col}' from modelling."
            elif issue["type"] == "duplicate_rows":
                claim = f"The data contains {issue['count']} duplicate rows."
                action = "De-duplicate before reporting rates."
            elif issue["type"] == "missing_values" and issue["pct"] >= 5:
                claim = f"'{col}' is missing in {issue['pct']}% of rows."
                action = f"Check whether missingness in '{col}' is related to the outcome."
            else:
                continue
            findings.append({"claim": claim, "kind": "data_quality", "columns": [col] if col else [],
                             "evidence_ids": [result["evidence_id"]], "confidence": "high",
                             "implication": action})
        return [("submit_findings", {"findings": findings[:4],
                                     "summary": f"Scanned the data; {result['issue_count']} issues found."})]

    def _statistician(self, prompt, messages):
        profile, target = self._context(prompt)
        cols = {c["name"]: c for c in profile.get("columns", [])}
        turn = _turn(messages)
        if not target or target not in cols:
            return [("submit_findings", {"findings": [], "summary": "No target column to test against."})]
        if turn == 0:
            return [("correlations", {"column": target, "top_k": 30})]
        if turn == 1:
            ranked = [r for r in _tool_results(messages)[0]["results"]
                      if cols.get(r["column"], {}).get("n_unique", 0) < profile["rows"]]
            picks = ranked[:2] + ranked[-1:] if len(ranked) > 2 else ranked
            return [("run_hypothesis_test", self._spec(cols, target, r["column"])) for r in picks]
        # Pair each test result with the spec that produced it (same order).
        specs = [b.input for b in messages[-2]["content"] if getattr(b, "type", None) == "tool_use"]
        findings = []
        for spec, r in zip(specs, _tool_results(messages)[1:]):
            if r.get("status") != "tested":
                continue
            cols_used = [spec[k] for k in ("group_var", "outcome_var", "var_a", "var_b") if spec.get(k)]
            if r["verdict"] != "CONFIRMED":
                other = next(c for c in cols_used if c != target)
                findings.append({
                    "claim": f"'{other}' shows no meaningful association with {target} ({r['verdict']}).",
                    "kind": "descriptive", "columns": cols_used, "evidence_ids": [r["evidence_id"]],
                    "effect": "absent",
                    "confidence": "medium", "implication": f"Deprioritise '{other}'."})
                continue
            eff = r["effect"]
            findings.append({
                "claim": f"{r['comparison']}: {eff['name']} = {eff['value']:.3f} ({r['test']}, "
                         f"adjusted p = {r['p_adjusted']:.2g}).",
                "kind": "statistical", "columns": cols_used, "effect": "present",
                "evidence_ids": [r["evidence_id"]], "confidence": "high",
                "implication": "Prioritise this factor when segmenting."})
        return [("submit_findings", {"findings": findings, "summary": "Tested the strongest and "
                                                                      "weakest associations."})]

    @staticmethod
    def _spec(cols, target, col):
        target_binary = cols[target]["n_unique"] == 2
        numeric = cols[col]["dtype"].startswith(("int", "float")) and cols[col]["n_unique"] > 10
        if target_binary and numeric:
            return {"claim_type": "group_difference_numeric", "direction": "any",
                    "group_var": target, "outcome_var": col}
        if target_binary:
            return {"claim_type": "association_categorical", "direction": "any",
                    "var_a": col, "var_b": target}
        if numeric:
            return {"claim_type": "correlation_numeric", "direction": "any", "var_a": col, "var_b": target}
        return {"claim_type": "group_difference_numeric", "direction": "any",
                "group_var": col, "outcome_var": target}

    def _ml_engineer(self, prompt, messages):
        profile, target = self._context(prompt)
        if not target:
            return [("submit_findings", {"findings": [], "summary": "No target to model."})]
        ids = [c["name"] for c in profile["columns"] if c["n_unique"] == profile["rows"]]
        results = _tool_results(messages)
        turn = _turn(messages)
        if turn == 0:
            return [("train_models", {"exclude_columns": ids})]
        trained = results[-1] if turn == 1 else next(r for r in reversed(results) if "models" in r)
        leaks = [s["feature"] for s in trained.get("leakage_suspects", [])]
        if turn == 1 and leaks:
            return [("train_models", {"exclude_columns": ids + leaks})]
        if "top_features" not in results[-1]:
            return [("explain_model", {"exclude_columns": trained["excluded"]})]

        shap = results[-1]
        best = trained["best_model"]
        metric = trained["selection_metric"]
        score = trained["models"][best][metric]
        findings = []
        first = next(r for r in results if "models" in r)
        for s in first.get("leakage_suspects", []):
            findings.append({"claim": f"'{s['feature']}' predicts {target} almost perfectly on its own "
                             f"({s['metric']} {s['value']:.3f}), indicating target leakage.",
                             "kind": "data_quality", "columns": [s["feature"]],
                             "evidence_ids": [first["evidence_id"]], "confidence": "high",
                             "implication": f"Exclude '{s['feature']}' from any production model."})
        baseline = trained.get("majority_class_baseline_accuracy")
        base_txt = f" against a majority-class baseline accuracy of {baseline:.3f}" if baseline else ""
        findings.append({"claim": f"{best} predicts {target} with cross-validated {metric} of "
                         f"{score:.3f}{base_txt}.", "kind": "predictive", "columns": [target],
                         "evidence_ids": [trained["evidence_id"]], "confidence": "high",
                         "implication": "The outcome is predictable enough to target interventions."})
        top = shap["top_features"][:3]
        findings.append({"claim": "Top model drivers by mean |SHAP|: " + ", ".join(
                         f"{t['feature']} ({t['mean_abs_shap']:.3f})" for t in top) + ".",
                         "kind": "predictive", "columns": [t["feature"] for t in top], "effect": "present",
                         "evidence_ids": [shap["evidence_id"]], "confidence": "medium",
                         "implication": "Focus interventions on these drivers."})
        return [("submit_findings", {"findings": findings, "summary": f"Best model {best}."})]

    def _critic(self, prompt):
        findings = json.loads(_section(prompt, "FINDINGS TO REVIEW"))
        return {"reviews": [{"finding_id": f["id"], "verdict": "accept",
                             "reason": "Claim matches the cited evidence.", "follow_up": None}
                            for f in findings]}

    def _reporter(self, prompt):
        question = _section(prompt, "QUESTION")
        raw = _section(prompt, "ACCEPTED FINDINGS")
        findings = json.loads(raw) if raw.startswith("[") else []
        ruled = _section(prompt, "NOT SUPPORTED BY THE DATA (you may cite these as [E#] in \"What we ruled out\")")
        ruled = json.loads(ruled) if ruled.startswith("[") else []
        lines = [f"# {question}", "",
                 "**Bottom line:** The accepted findings below are each backed by computed evidence.", "",
                 "## Key findings"]
        lines += [f"- {f['claim']} [{f['id']}]" for f in findings] or ["- No findings were accepted."]
        lines += ["", "## Recommended actions"]
        lines += [f"- {f['implication']} [{f['id']}]" for f in findings if f.get("implication")]
        lines += ["", "## What we ruled out"]
        lines += [f"- {r['comparison']} [{r['evidence_id']}]" for r in ruled] or ["- Nothing was ruled out."]
        lines += ["", "## Caveats", "- Observational data: associations, not causes."]
        return "\n".join(lines)
