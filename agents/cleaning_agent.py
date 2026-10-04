import json
from typing import Literal, Optional

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from utils.llm import call_llm_json

load_dotenv()


Strategy = Literal["mean", "median", "mode", "drop_rows", "drop_column", "constant_unknown"]
NUMERIC_ONLY = ("mean", "median")


class CleaningStrategy(BaseModel):
    """
    The LLM's cleaning plan. Validating it before applying it means an invented
    strategy name ("interpolate", "knn") is caught and retried instead of being
    silently skipped by the if/elif chain below.
    """

    model_config = ConfigDict(extra="ignore")

    missing_value_strategies: dict[str, Strategy] = Field(default_factory=dict)
    columns_to_drop: list[str] = Field(default_factory=list)
    reasoning: str = ""


def get_cleaning_strategy(stats: dict, target_col: Optional[str] = None) -> dict:
    target_note = (
        f"\nThe target column is '{target_col}'. Never drop it or impute it.\n"
        if target_col else ""
    )
    prompt = f"""
You are an expert data scientist deciding how to clean a dataset.

DATASET STATISTICS:
{json.dumps(stats, indent=2, default=str)}
{target_note}
Respond ONLY with valid JSON:
{{
    "missing_value_strategies": {{"column_name": "strategy"}},
    "columns_to_drop": ["col1"],
    "reasoning": "brief explanation"
}}

Valid strategies: "mean", "median", "mode", "drop_rows", "drop_column", "constant_unknown"
"mean" and "median" are only valid for numeric columns.
"""
    strategy = call_llm_json(
        prompt=prompt,
        system="You are a data cleaning expert. Respond with valid JSON only.",
        model_cls=CleaningStrategy,
        max_tokens=1024,
    )
    return strategy.model_dump()


def apply_cleaning_strategy(df: pd.DataFrame, strategy: dict,
                            target_col: Optional[str] = None) -> pd.DataFrame:
    """
    Apply the plan deterministically. The LLM chooses *what* to do; this
    function decides whether it is safe, so a bad plan degrades rather than
    corrupting the data or crashing the run.
    """
    df = df.copy()
    print("\n--- Applying cleaning strategy ---")

    unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        print(f"Dropped auto-index columns: {unnamed_cols}")

    for col in strategy.get("columns_to_drop", []):
        if col == target_col:
            print(f"Refused to drop target column '{col}'")
            continue
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"Dropped column: {col}")

    # Rows with no label are unusable for supervised learning, and imputing a
    # label would train the model on values ADA made up.
    if target_col and target_col in df.columns:
        before = len(df)
        df = df.dropna(subset=[target_col])
        if before - len(df):
            print(f"Dropped {before - len(df)} rows with missing target '{target_col}'")

    for col, method in strategy.get("missing_value_strategies", {}).items():
        if col == target_col or col not in df.columns or df[col].isnull().sum() == 0:
            continue
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        if method in NUMERIC_ONLY and not is_numeric:
            print(f"'{method}' is undefined for non-numeric '{col}' — using mode instead")
            method = "mode"

        if method == "mean":
            fill_val = df[col].mean()
            df[col] = df[col].fillna(fill_val)
            print(f"Filled '{col}' with mean ({fill_val:.2f})")
        elif method == "median":
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
            print(f"Filled '{col}' with median ({fill_val:.2f})")
        elif method == "mode":
            fill_val = df[col].mode()[0]
            df[col] = df[col].fillna(fill_val)
            print(f"Filled '{col}' with mode ({fill_val})")
        elif method == "drop_rows":
            before = len(df)
            df = df.dropna(subset=[col])
            print(f"Dropped {before - len(df)} rows with missing '{col}'")
        elif method == "drop_column":
            df = df.drop(columns=[col])
            print(f"Dropped column '{col}'")
        elif method == "constant_unknown":
            df[col] = df[col].fillna("Unknown")
            print(f"Filled '{col}' with 'Unknown'")

    before = len(df)
    df = df.drop_duplicates()
    if before - len(df) > 0:
        print(f"Removed {before - len(df)} duplicate rows")

    return df


def run_cleaning_agent(df: pd.DataFrame, eda_stats: dict,
                       target_col: Optional[str] = None) -> tuple:
    print("\nCleaning Agent starting...")
    print("Asking AI for cleaning strategy...")
    strategy = get_cleaning_strategy(eda_stats, target_col)
    print(f"\nAI Reasoning: {strategy.get('reasoning', 'N/A')}")
    cleaned_df = apply_cleaning_strategy(df, strategy, target_col)
    print("\n--- Cleaning Summary ---")
    print(f"Original shape:           {df.shape}")
    print(f"Cleaned shape:            {cleaned_df.shape}")
    print(f"Missing values remaining: {cleaned_df.isnull().sum().sum()}")
    return cleaned_df, strategy
