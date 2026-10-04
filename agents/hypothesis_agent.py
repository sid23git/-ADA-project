import json

import pandas as pd
from dotenv import load_dotenv

from utils.hypothesis_schema import HypothesisSet
from utils.llm import call_llm_json
from utils.stat_tests import MAX_NUMERIC_AS_CATEGORICAL, resolve

load_dotenv()


def profile_columns(df: pd.DataFrame) -> dict:
    """
    Describe each column in the vocabulary the claim types use.

    The LLM has to name real columns and pick a claim type its dtype can
    actually support, so the prompt tells it which columns are usable in which
    role rather than leaving it to infer that from a dtype string.
    """
    profile = {}
    for col in df.columns:
        series = df[col]
        n_unique = int(series.nunique(dropna=True))
        is_numeric = (
            pd.api.types.is_numeric_dtype(series)
            and not pd.api.types.is_bool_dtype(series)
        )
        usable_as_group = (not is_numeric) or n_unique <= MAX_NUMERIC_AS_CATEGORICAL

        profile[str(col)] = {
            "dtype": str(series.dtype),
            "unique_values": n_unique,
            "missing": int(series.isnull().sum()),
            "sample_values": [
                v.item() if hasattr(v, "item") else v
                for v in series.dropna().head(4).tolist()
            ],
            "usable_as": [
                role for role, ok in (
                    ("numeric_outcome", is_numeric),
                    ("group_or_category", usable_as_group),
                    ("binary_outcome", n_unique == 2),
                ) if ok
            ],
            "levels": (
                [str(v) for v in sorted(series.dropna().unique(), key=str)[:12]]
                if usable_as_group and n_unique <= 12 else None
            ),
        }
    return profile


TAXONOMY = """
Every hypothesis MUST be expressed as one of these four claim types, and its
test_spec MUST name real columns from the list above.

1. "group_difference_numeric" — a numeric outcome differs across groups.
   Requires: group_var (a column usable as group_or_category),
             outcome_var (a column usable as numeric_outcome).
   direction: "higher_in_focus" | "lower_in_focus" | "any".
   If direction is higher_in_focus or lower_in_focus you MUST also give
   focus_level: the exact value of group_var your claim says is higher/lower.

2. "proportion_difference" — a binary outcome's rate differs across groups.
   Requires: group_var, outcome_var (a column usable as binary_outcome).
   direction and focus_level: same rules as above.

3. "correlation_numeric" — two numeric columns move together.
   Requires: var_a and var_b, both usable as numeric_outcome.
   direction: "positive" | "negative" | "any".

4. "association_categorical" — two categorical columns are associated.
   Requires: var_a and var_b, both usable as group_or_category.
   direction: MUST be "any" (this test has no direction).

Prefer directional claims — they are more informative and more falsifiable.
Use focus_level values exactly as they appear in the "levels" list.
"""


def generate_hypotheses(df: pd.DataFrame, target_col: str = None) -> dict:
    profile = profile_columns(df)
    target_info = (
        f"The target column is '{target_col}'."
        if target_col else
        f"The likely target column is '{df.columns[-1]}'."
    )

    prompt = f"""
You are an expert data scientist about to analyze a dataset.
Before running any analysis, form 5 testable hypotheses.

Each hypothesis will be checked by running a real statistical test in code, so a
vague or untestable hypothesis is worthless here — it must map exactly onto one
of the claim types below, using real column names.

COLUMN PROFILE:
{json.dumps(profile, indent=2, default=str)}

{target_info}

{TAXONOMY}

Respond ONLY with valid JSON in exactly this shape:
{{
    "dataset_type": "one sentence describing the dataset",
    "hypotheses": [
        {{
            "id": "H1",
            "hypothesis": "clear testable statement in plain English",
            "reasoning": "why you think this",
            "expected_evidence": "what result would confirm this",
            "confidence": "high" or "medium" or "low",
            "test_spec": {{
                "claim_type": "one of the four claim types",
                "direction": "higher_in_focus|lower_in_focus|positive|negative|any",
                "group_var": "column name or null",
                "outcome_var": "column name or null",
                "focus_level": "exact level value or null",
                "var_a": "column name or null",
                "var_b": "column name or null"
            }}
        }}
    ],
    "most_important_hypothesis": "H1",
    "analysis_strategy": "brief note on what to focus on"
}}

Give exactly 5 hypotheses, ids H1 through H5.
"""

    result = call_llm_json(
        prompt=prompt,
        system=(
            "You are a data science expert. Form hypotheses that can be checked "
            "with a statistical test. Respond with valid JSON only."
        ),
        model_cls=HypothesisSet,
        max_tokens=2500,
    )

    # Resolution failures are reported now rather than discovered at validation
    # time, so the audit trail shows the hypothesis was unusable from the start.
    unresolvable = []
    for h in result.hypotheses:
        _, failure = resolve(h.test_spec, df)
        if failure:
            unresolvable.append(f"{h.id}: {failure['detail']}")

    payload = result.model_dump(mode="json")
    payload["unresolvable"] = unresolvable

    print(f"\nDataset identified as: {payload['dataset_type']}")
    print(f"\nGenerated {len(payload['hypotheses'])} hypotheses:")
    for h in payload["hypotheses"]:
        spec = h["test_spec"]
        print(f"  {h['id']} [{h['confidence']}]: {h['hypothesis']}")
        print(f"       -> {spec['claim_type']} / {spec['direction']}")
    if unresolvable:
        print(f"\n{len(unresolvable)} hypotheses cannot be tested against this data:")
        for note in unresolvable:
            print(f"  - {note}")
    print(f"\nMost important: {payload['most_important_hypothesis']}")

    return payload


def run_hypothesis_agent(df: pd.DataFrame, target_col: str = None) -> dict:
    print("\nHypothesis Agent starting...")
    print("Forming hypotheses before analysis...")
    hypotheses = generate_hypotheses(df, target_col)
    print("\nHypothesis Agent complete.")
    return hypotheses
