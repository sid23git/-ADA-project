"""
Hypothesis validation.

The verdict for every hypothesis is computed in utils/stat_tests.py from a real
statistical test. This module orchestrates that and then asks the LLM for prose
about numbers it did not choose — it never asks the model what the answer is.

Ordering matters: all tests run and all p-values are corrected across the family
BEFORE the LLM sees anything, so the narrative can never describe a p-value that
is about to become non-significant after adjustment.
"""

import json

import pandas as pd
from dotenv import load_dotenv
from pydantic import ValidationError

from utils.hypothesis_schema import InterpretationBatch, TestSpec, Verdict
from utils.llm import call_llm_json
from utils.stat_tests import run_statistical_validation

load_dotenv()

VERDICT_ICONS = {
    Verdict.CONFIRMED: "[CONFIRMED]",
    Verdict.REJECTED: "[REJECTED]",
    Verdict.SIGNIFICANT_BUT_TRIVIAL: "[TRIVIAL]",
    Verdict.NOT_SUPPORTED: "[NOT SUPPORTED]",
    Verdict.INSUFFICIENT_DATA: "[INSUFFICIENT DATA]",
    Verdict.NOT_TESTABLE: "[NOT TESTABLE]",
}


def _parse_specs(hypothesis_items: list) -> tuple[dict, dict]:
    """Rebuild TestSpec objects from stored dicts, keeping the raw items alongside."""
    specs, by_id = {}, {}
    for item in hypothesis_items:
        hid = item.get("id")
        if not hid:
            continue
        by_id[hid] = item
        raw_spec = item.get("test_spec")
        if not raw_spec:
            continue
        try:
            specs[hid] = TestSpec.model_validate(raw_spec)
        except ValidationError:
            # An untyped or malformed hypothesis is reported as untestable
            # rather than quietly handed to the model for a judgment call.
            continue
    return specs, by_id


def _interpret_with_ai(records: dict, by_id: dict) -> dict:
    """
    Ask the LLM for prose only.

    The response model has no verdict field and forbids extra keys, so a model
    that tries to supply its own verdict fails validation and gets retried. If
    interpretation fails entirely the pipeline degrades to no narrative — never
    to a narrative that disagrees with the computed result.
    """
    payload = []
    for hid, record in records.items():
        payload.append({
            "id": hid,
            "hypothesis": by_id.get(hid, {}).get("hypothesis", ""),
            "verdict": record["verdict"].value,
            "evidence": record["evidence"],
            "test_used": record.get("test_used"),
            "p_value": record.get("p_value"),
            "p_adjusted": record.get("p_adjusted"),
            "effect": {
                "name": record.get("effect_name"),
                "value": record.get("effect_value"),
                "ci_95": record.get("effect_ci"),
            },
            "n_used": record.get("n_used"),
            "n_total": record.get("n_total"),
            "practically_significant": record.get("practically_significant"),
        })

    prompt = f"""
You are interpreting the results of statistical tests that have already been run.

RESULTS:
{json.dumps(payload, indent=2, default=str)}

The verdict for each hypothesis was computed in code from the test statistic and
the adjusted p-value. It is final. Do NOT restate, dispute, or imply a different
verdict — your job is only to explain what the computed numbers mean.

For each result write:
- "insight": what this tells us in plain English, referring to the actual effect
  size and sample size. If the verdict is NOT_SUPPORTED, say the data did not
  provide evidence for the claim — never say the claim is proven false.
- "caveat": one limitation a careful reader should note, or null.

Respond ONLY with valid JSON:
{{
    "interpretations": [
        {{"id": "H1", "insight": "...", "caveat": "..."}}
    ]
}}
"""

    try:
        batch = call_llm_json(
            prompt=prompt,
            system=(
                "You are a statistician explaining computed results. You never "
                "decide verdicts. Respond with valid JSON only."
            ),
            model_cls=InterpretationBatch,
            max_tokens=2000,
        )
        return {i.id: {"insight": i.insight, "caveat": i.caveat}
                for i in batch.interpretations}
    except Exception as exc:
        print(f"Interpretation step failed ({exc}) — reporting statistics without narrative.")
        return {}


def _summarise(records: dict, by_id: dict) -> tuple[str, str]:
    """Code-generated summary, so the headline counts always match the results."""
    counts = {}
    for record in records.values():
        counts[record["verdict"]] = counts.get(record["verdict"], 0) + 1

    total = len(records)
    parts = [f"{n} {verdict.value.replace('_', ' ').lower()}"
             for verdict, n in sorted(counts.items(), key=lambda kv: kv[0].value)]
    summary = f"{total} hypotheses tested: " + ", ".join(parts) + "."

    tested = [r for r in records.values() if r.get("p_adjusted") is not None]
    if tested:
        summary += (
            f" P-values are {tested[0].get('correction_method', 'BH')}-adjusted across "
            f"the {tested[0].get('family_size', len(tested))} hypotheses that could be tested."
        )

    # The most surprising result is a claim the agent was confident about that
    # the data did not bear out.
    surprising = ""
    for hid, record in records.items():
        item = by_id.get(hid, {})
        if item.get("confidence") == "high" and record["verdict"] in (
            Verdict.REJECTED, Verdict.NOT_SUPPORTED, Verdict.SIGNIFICANT_BUT_TRIVIAL
        ):
            surprising = (
                f"{hid} was rated high-confidence before analysis but came back "
                f"{record['verdict'].value.replace('_', ' ').lower()}: "
                f"{item.get('hypothesis', '')}"
            )
            break
    return summary, surprising


def validate_hypotheses(hypotheses: dict, df: pd.DataFrame,
                        alpha: float = 0.05,
                        correction: str = "benjamini-hochberg") -> dict:
    items = hypotheses.get("hypotheses", [])
    specs, by_id = _parse_specs(items)

    records = run_statistical_validation(specs, df, alpha=alpha, correction=correction)

    # Hypotheses whose spec would not even parse still get an honest entry.
    for hid in by_id:
        if hid not in records:
            records[hid] = {
                "verdict": Verdict.NOT_TESTABLE,
                "reason": "missing_variable",
                "detail": "hypothesis has no usable test specification",
                "evidence": "Not testable - no valid statistical test specification.",
                "n_used": 0,
                "n_total": len(df),
            }

    interpretations = _interpret_with_ai(records, by_id)

    validation_results = []
    for hid in sorted(records, key=lambda h: (len(h), h)):
        record = records[hid]
        item = by_id.get(hid, {})
        narrative = interpretations.get(hid, {})

        # Computed fields are applied LAST so nothing from the model can
        # overwrite the verdict or the evidence string.
        validation_results.append({
            "insight": narrative.get("insight", "No interpretation available."),
            "caveat": narrative.get("caveat"),
            "id": hid,
            "hypothesis": item.get("hypothesis", ""),
            "verdict": record["verdict"].value,
            "evidence": record["evidence"],
            "test_spec": item.get("test_spec"),
            "statistics": {
                "test_used": record.get("test_used"),
                "statistic": record.get("statistic"),
                "p_value": record.get("p_value"),
                "p_adjusted": record.get("p_adjusted"),
                "correction_method": record.get("correction_method"),
                "family_size": record.get("family_size"),
                "effect_name": record.get("effect_name"),
                "effect_value": record.get("effect_value"),
                "effect_ci": record.get("effect_ci"),
                "practically_significant": record.get("practically_significant"),
                "direction_match": record.get("direction_match"),
                "n_used": record.get("n_used"),
                "n_total": record.get("n_total"),
                "assumption_note": record.get("assumption_note"),
                "coercions": record.get("coercions"),
                "reason": record.get("reason"),
            },
        })

    overall_summary, most_surprising = _summarise(records, by_id)

    print("\n--- Hypothesis Validation Results ---")
    for v in validation_results:
        print(f"{VERDICT_ICONS.get(Verdict(v['verdict']), '')} {v['id']}: {v['verdict']}")
        print(f"    {v['evidence']}")
    print(f"\n{overall_summary}")

    return {
        "validation_results": validation_results,
        "overall_summary": overall_summary,
        "most_surprising": most_surprising,
        "alpha": alpha,
        "correction_method": correction,
    }


def run_validator_agent(hypotheses: dict, df: pd.DataFrame) -> dict:
    print("\nValidator Agent starting...")
    if not hypotheses or not hypotheses.get("hypotheses"):
        print("No hypotheses to validate - skipping.")
        return {}
    if df is None or len(df) == 0:
        print("No data to test hypotheses against - skipping.")
        return {}
    validation = validate_hypotheses(hypotheses, df)
    print("\nValidator Agent complete.")
    return validation
