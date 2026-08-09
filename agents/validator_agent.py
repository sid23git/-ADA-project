import json
from dotenv import load_dotenv
from utils.llm import call_llm

load_dotenv()


def validate_hypotheses(hypotheses: dict, eda_stats: dict, ml_results: dict, explain_results: dict) -> dict:
    findings = {
        "eda": {
            "missing_values": eda_stats.get("missing_values", {}),
            "numeric_summary": list(eda_stats.get("numeric_summary", {}).keys()),
            "categorical_summary": eda_stats.get("categorical_summary", {})
        },
        "ml": {
            "problem_type": ml_results.get("problem_type", ""),
            "best_model": ml_results.get("interpretation", {}).get("best_model", ""),
            "model_results": ml_results.get("model_results", {}),
            "concerns": ml_results.get("interpretation", {}).get("concerns", "")
        },
        "explanation": {
            "top_features": list(explain_results.get("feature_importance", {}).keys())[:5],
            "plain_english": explain_results.get("interpretation", {}).get("plain_english_summary", "")
        }
    }

    prompt = f"""
You are an expert data scientist validating hypotheses against actual findings.

ORIGINAL HYPOTHESES:
{json.dumps(hypotheses.get('hypotheses', []), indent=2)}

ACTUAL FINDINGS:
{json.dumps(findings, indent=2)}

Respond ONLY with valid JSON:
{{
    "validation_results": [
        {{
            "id": "H1",
            "hypothesis": "original hypothesis",
            "verdict": "CONFIRMED/REJECTED/INCONCLUSIVE",
            "evidence": "specific finding supporting this verdict",
            "insight": "what this tells us"
        }}
    ],
    "overall_summary": "2-3 sentence summary",
    "most_surprising": "most unexpected result",
    "scientific_contribution": "what these findings add"
}}
"""

    raw = call_llm(
        prompt=prompt,
        system="You are a data science expert validating hypotheses. Be rigorous. Respond with valid JSON only.",
        max_tokens=1500
    )

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    result = json.loads(raw.strip())

    print("\n--- Hypothesis Validation Results ---")
    confirmed = rejected = inconclusive = 0
    for v in result["validation_results"]:
        icon = "✅" if v["verdict"] == "CONFIRMED" else "❌" if v["verdict"] == "REJECTED" else "⚠️"
        print(f"{icon} {v['id']}: {v['verdict']}")
        if v["verdict"] == "CONFIRMED": confirmed += 1
        elif v["verdict"] == "REJECTED": rejected += 1
        else: inconclusive += 1

    print(f"\nResults: {confirmed} confirmed, {rejected} rejected, {inconclusive} inconclusive")
    return result


def run_validator_agent(hypotheses: dict, eda_stats: dict, ml_results: dict, explain_results: dict) -> dict:
    print("\nValidator Agent starting...")
    if not hypotheses or not hypotheses.get("hypotheses"):
        print("No hypotheses to validate — skipping.")
        return {}
    validation = validate_hypotheses(hypotheses, eda_stats, ml_results, explain_results)
    print("\nValidator Agent complete.")
    return validation
