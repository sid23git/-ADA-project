"""
Synthetic datasets with planted ground truth.

Each scenario is generated from a known causal recipe, so the eval knows which
columns really drive the outcome, which are pure noise, and which traps (target
leakage, placeholder values, duplicates) a careful analyst should catch.
Seeded, so every run sees identical data.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Scenario:
    name: str
    question: str
    target: str
    df: pd.DataFrame
    drivers: frozenset            # columns that genuinely affect the target
    nulls: frozenset              # columns with no effect at all
    traps: dict = field(default_factory=dict)   # trap name -> column a good analysis flags
    description: str = ""


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def churn(seed: int = 7, n: int = 4000) -> Scenario:
    rng = np.random.default_rng(seed)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.25, 0.20])
    tenure = rng.integers(1, 73, n)
    support_calls = rng.poisson(1.5, n)
    charges = np.round(rng.normal(70, 20, n).clip(20, 140), 2)
    logit = (-1.6 + 1.5 * (contract == "month-to-month") - 0.035 * tenure
             + 0.40 * support_calls + 0.012 * (charges - 70))
    df = pd.DataFrame({
        "customer_id": [f"C{100000 + i}" for i in range(n)],
        "tenure_months": tenure,
        "contract": contract,
        "monthly_charges": charges,
        "support_calls": support_calls,
        "region": rng.choice(["north", "south", "east", "west"], n),
        "payment_method": rng.choice(["card", "bank", "paypal", "invoice"], n),
        "signup_channel": rng.choice(["web", "store", "phone"], n),
        "churned": (rng.random(n) < _sigmoid(logit)).astype(int),
    })
    return Scenario(
        "churn", "Which customers are most likely to churn, and what should the retention team "
        "focus on?", "churned", df,
        drivers=frozenset({"contract", "tenure_months", "support_calls", "monthly_charges"}),
        nulls=frozenset({"region", "payment_method", "signup_channel"}),
        description="Telecom churn driven by contract, tenure, support calls and charges.",
    )


def leakage(seed: int = 11) -> Scenario:
    base = churn(seed)
    rng = np.random.default_rng(seed + 1)
    df = base.df.copy()
    # Recorded during the cancellation process, i.e. after the outcome.
    df["exit_survey_completed"] = np.where(rng.random(len(df)) < 0.97, df["churned"], 1 - df["churned"])
    return Scenario(
        "leakage", "Build us a model to predict churn and tell us what drives it.", "churned", df,
        drivers=base.drivers, nulls=base.nulls,
        traps={"target_leakage": "exit_survey_completed"},
        description="Churn data plus a column recorded after cancellation.",
    )


def dirty_readmission(seed: int = 23, n: int = 3000) -> Scenario:
    rng = np.random.default_rng(seed)
    age = rng.integers(25, 90, n).astype(float)
    prior = rng.poisson(1.0, n)
    diabetic = rng.random(n) < 0.3
    los = np.round(rng.gamma(2.0, 2.5, n), 1)
    logit = -2.3 + 0.55 * prior + 0.8 * diabetic + 0.12 * los + 0.01 * (age - 60)
    df = pd.DataFrame({
        "patient_id": np.arange(1, n + 1),
        "age": age,
        "length_of_stay_days": los,
        "prior_admissions": prior,
        "diabetic": np.where(diabetic, "yes", "no"),
        "num_medications": rng.poisson(8, n),
        "ward": rng.choice(["A", "B", "C", "D", "E"], n),
        "readmitted_30d": (rng.random(n) < _sigmoid(logit)).astype(int),
    })
    # The EHR export writes -999 for unknown age...
    unknown = rng.random(n) < 0.05
    df.loc[unknown, "age"] = -999
    # ...and a bad join duplicated some encounters.
    df = pd.concat([df, df.sample(frac=0.03, random_state=seed)], ignore_index=True)
    return Scenario(
        "dirty_readmission", "What drives 30-day hospital readmission, and which patients "
        "should get follow-up calls?", "readmitted_30d", df,
        drivers=frozenset({"prior_admissions", "diabetic", "length_of_stay_days"}),
        nulls=frozenset({"ward", "num_medications"}),
        traps={"sentinel_value": "age", "duplicate_rows": "*"},
        description="Readmissions with -999 placeholder ages and duplicated encounters.",
    )


def null_world(seed: int = 31, n: int = 2500) -> Scenario:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "visitor_id": np.arange(n),
        "device": rng.choice(["mobile", "desktop", "tablet"], n),
        "traffic_source": rng.choice(["ads", "organic", "email", "social"], n),
        "pages_viewed": rng.poisson(4, n),
        "session_minutes": np.round(rng.exponential(5, n), 2),
        "country": rng.choice(["US", "UK", "DE", "IN", "BR"], n),
        "converted": (rng.random(n) < 0.08).astype(int),     # independent of everything
    })
    return Scenario(
        "null_world", "What drives conversion on our website?", "converted", df,
        drivers=frozenset(),
        nulls=frozenset({"device", "traffic_source", "pages_viewed", "session_minutes", "country"}),
        description="Conversion is pure noise. The honest answer is that nothing here predicts it.",
    )


SCENARIOS = {s.__name__: s for s in (churn, leakage, dirty_readmission, null_world)}


def load(name: str) -> Scenario:
    return SCENARIOS[name]()
