import pandas as pd
import numpy as np
from dotenv import load_dotenv
import json
from utils.llm import call_llm

load_dotenv()


def get_cleaning_strategy(stats: dict) -> dict:
    prompt = f"""
You are an expert data scientist deciding how to clean a dataset.

DATASET STATISTICS:
{json.dumps(stats, indent=2)}

Respond ONLY with valid JSON:
{{
    "missing_value_strategies": {{"column_name": "strategy"}},
    "columns_to_drop": ["col1"],
    "reasoning": "brief explanation"
}}

Valid strategies: "mean", "median", "mode", "drop_rows", "drop_column", "constant_unknown"
"""
    raw = call_llm(prompt=prompt, system="You are a data cleaning expert. Respond with valid JSON only.", max_tokens=1024)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def apply_cleaning_strategy(df: pd.DataFrame, strategy: dict) -> pd.DataFrame:
    df = df.copy()
    print("\n--- Applying cleaning strategy ---")

    unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        print(f"Dropped auto-index columns: {unnamed_cols}")

    for col in strategy.get("columns_to_drop", []):
        if col in df.columns:
            df = df.drop(columns=[col])
            print(f"Dropped column: {col}")

    for col, method in strategy.get("missing_value_strategies", {}).items():
        if col not in df.columns or df[col].isnull().sum() == 0:
            continue
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


def run_cleaning_agent(df: pd.DataFrame, eda_stats: dict) -> tuple:
    print("\nCleaning Agent starting...")
    print("Asking AI for cleaning strategy...")
    strategy = get_cleaning_strategy(eda_stats)
    print(f"\nAI Reasoning: {strategy.get('reasoning', 'N/A')}")
    cleaned_df = apply_cleaning_strategy(df, strategy)
    print(f"\n--- Cleaning Summary ---")
    print(f"Original shape:           {df.shape}")
    print(f"Cleaned shape:            {cleaned_df.shape}")
    print(f"Missing values remaining: {cleaned_df.isnull().sum().sum()}")
    return cleaned_df, strategy
