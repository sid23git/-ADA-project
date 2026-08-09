import pandas as pd
import numpy as np
from dotenv import load_dotenv
from utils.llm import call_llm

load_dotenv()


def analyze_dataframe(df: pd.DataFrame) -> dict:
    analysis = {}
    analysis["shape"] = {"rows": df.shape[0], "columns": df.shape[1]}
    analysis["column_types"] = df.dtypes.astype(str).to_dict()
    missing = df.isnull().sum()
    analysis["missing_values"] = {col: int(count) for col, count in missing.items() if count > 0}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        analysis["numeric_summary"] = df[numeric_cols].describe().to_dict()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    analysis["categorical_summary"] = {col: df[col].nunique() for col in categorical_cols}
    analysis["potential_target"] = df.columns[-1]
    return analysis


def run_eda_agent(filepath: str) -> str:
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Extracting statistics...")
    stats = analyze_dataframe(df)
    print("Running AI analysis...")

    prompt = f"""
You are an expert data scientist performing Exploratory Data Analysis (EDA).
Analyze these statistics and produce a professional EDA report.

DATASET STATISTICS:
{stats}

Please provide:
1. Dataset overview
2. Data quality assessment
3. Key observations about numeric columns
4. Key observations about categorical columns
5. Hypothesis — what problem is this dataset likely solving?
6. Recommended next steps for data cleaning
"""

    return call_llm(
        prompt=prompt,
        system="You are an expert data scientist. Provide professional, actionable insights.",
        max_tokens=1500
    )
