import json
from dotenv import load_dotenv
import pandas as pd
from utils.llm import call_llm

load_dotenv()


def generate_hypotheses(df: pd.DataFrame, target_col: str = None) -> dict:
    column_info = {
        col: {
            "dtype": str(df[col].dtype),
            "unique_values": int(df[col].nunique()),
            "sample_values": df[col].dropna().head(5).tolist()
        }
        for col in df.columns
    }

    target_info = f"Target column: '{target_col}'" if target_col else f"Likely target: '{df.columns[-1]}'"

    prompt = f"""
You are an expert data scientist about to analyze a dataset.
Before running any analysis, form 5 testable hypotheses.

DATASET COLUMN INFORMATION:
{json.dumps(column_info, indent=2)}

{target_info}

Respond ONLY with valid JSON:
{{
    "dataset_type": "one sentence describing the dataset",
    "hypotheses": [
        {{
            "id": "H1",
            "hypothesis": "clear testable statement",
            "reasoning": "why you think this",
            "expected_evidence": "what would confirm this",
            "confidence": "high/medium/low"
        }}
    ],
    "most_important_hypothesis": "H1/H2/H3/H4/H5",
    "analysis_strategy": "brief note on what to focus on"
}}
"""

    raw = call_llm(
        prompt=prompt,
        system="You are a data science expert. Form clear testable hypotheses. Respond with valid JSON only.",
        max_tokens=1500
    )

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    result = json.loads(raw.strip())

    print(f"\nDataset identified as: {result['dataset_type']}")
    print(f"\nGenerated {len(result['hypotheses'])} hypotheses:")
    for h in result["hypotheses"]:
        print(f"  {h['id']} [{h['confidence']}]: {h['hypothesis']}")
    print(f"\nMost important: {result['most_important_hypothesis']}")

    return result


def run_hypothesis_agent(df: pd.DataFrame, target_col: str = None) -> dict:
    print("\nHypothesis Agent starting...")
    print("Forming hypotheses before analysis...")
    hypotheses = generate_hypotheses(df, target_col)
    print("\nHypothesis Agent complete.")
    return hypotheses
