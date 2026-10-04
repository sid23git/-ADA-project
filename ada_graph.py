"""
ADA's LangGraph pipeline.

    load_data -> hypothesis -> eda -> cleaning -> ml -> explain -> validator -> report

Every node returns a new state ({**state, ...}) and appends to the audit trail.
A node that raises sets state["error"]; the conditional edge after it routes to
error_handler, which either stops the run (load_data), retries the failed node,
or — once retries are exhausted — skips ahead to the next node.
"""

import json
import os
from collections.abc import Callable
from datetime import datetime
from typing import Literal, Optional

import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from agents.cleaning_agent import run_cleaning_agent
from agents.eda_agent import analyze_dataframe, run_eda_agent
from agents.explain_agent import run_explain_agent
from agents.hypothesis_agent import run_hypothesis_agent
from agents.ml_agent import run_ml_agent
from agents.validator_agent import run_validator_agent
from utils.ada_state import ADAState
from utils.llm import call_llm

load_dotenv()

OUTPUT_DIR = "outputs"
MAX_RETRIES = 2
CRITICAL_NODES = ("load_data",)

# Linear order of the pipeline. Both normal routing and "skip after giving up"
# read from this one table.
NEXT_NODE = {
    "load_data": "hypothesis",
    "hypothesis": "eda",
    "eda": "cleaning",
    "cleaning": "ml",
    "ml": "explain",
    "explain": "validator",
    "validator": "report",
    "report": "end",
}


# ─────────────────────────────────────────────────
# HELPER — Logging
# ─────────────────────────────────────────────────

def log(state: ADAState, agent: str, action: str, detail: str = "") -> list:
    entry = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "agent": agent,
        "action": action,
        "detail": detail
    }
    print(f"[{entry['timestamp']}] [{agent}] {action}")
    if detail:
        print(f"           {detail}")
    trail = state.get("audit_trail", [])
    return trail + [entry]


# ─────────────────────────────────────────────────
# NODE 1 — Load Data
# ─────────────────────────────────────────────────

def load_data_node(state: ADAState) -> ADAState:
    try:
        trail = log(state, "LoadData", "Loading dataset...", state["filepath"])
        df = pd.read_csv(state["filepath"])
        trail = log({"audit_trail": trail}, "LoadData", "Dataset loaded",
                    f"{df.shape[0]} rows x {df.shape[1]} columns")
        return {**state, "raw_df": df, "current_node": "load_data",
                "audit_trail": trail, "error": None, "retry_count": 0}
    except Exception as e:
        trail = log(state, "LoadData", "FAILED", str(e))
        return {**state, "current_node": "load_data",
                "audit_trail": trail, "error": f"load_data: {str(e)}"}


# ─────────────────────────────────────────────────
# NODE 2 — Hypothesis
# ─────────────────────────────────────────────────

def hypothesis_node(state: ADAState) -> ADAState:
    try:
        trail = log(state, "Hypothesis", "Forming hypotheses before analysis...")
        hypotheses = run_hypothesis_agent(state["raw_df"], state.get("target_col"))
        trail = log({"audit_trail": trail}, "Hypothesis", "Hypotheses formed",
                    f"{len(hypotheses.get('hypotheses', []))} hypotheses generated")
        return {**state, "hypotheses": hypotheses, "current_node": "hypothesis",
                "audit_trail": trail, "error": None}
    except Exception as e:
        trail = log(state, "Hypothesis", "FAILED — skipping", str(e))
        return {**state, "hypotheses": {}, "current_node": "hypothesis",
                "audit_trail": trail, "error": None}


# ─────────────────────────────────────────────────
# NODE 3 — EDA
# ─────────────────────────────────────────────────

def eda_node(state: ADAState) -> ADAState:
    try:
        trail = log(state, "EDA", "Starting exploratory analysis...")
        df = state["raw_df"]
        eda_stats = analyze_dataframe(df)
        eda_report = run_eda_agent(df, eda_stats)
        missing_count = len(eda_stats.get("missing_values", {}))
        trail = log({"audit_trail": trail}, "EDA", "EDA complete",
                    f"Found {missing_count} columns with missing values")
        return {**state, "eda_stats": eda_stats, "eda_report": eda_report,
                "current_node": "eda", "audit_trail": trail, "error": None}
    except Exception as e:
        trail = log(state, "EDA", "FAILED", str(e))
        return {**state, "current_node": "eda",
                "audit_trail": trail, "error": f"eda: {str(e)}"}


# ─────────────────────────────────────────────────
# NODE 4 — Cleaning
# ─────────────────────────────────────────────────

def cleaning_node(state: ADAState) -> ADAState:
    try:
        trail = log(state, "Cleaning", "Starting data cleaning...")
        df = state["raw_df"]
        eda_stats = state["eda_stats"]
        cleaned_df, strategy = run_cleaning_agent(df, eda_stats, state.get("target_col"))
        trail = log({"audit_trail": trail}, "Cleaning", "Cleaning complete",
                    f"Shape: {df.shape} -> {cleaned_df.shape}")
        return {**state, "cleaned_df": cleaned_df, "cleaning_strategy": strategy,
                "current_node": "cleaning", "audit_trail": trail, "error": None}
    except Exception as e:
        trail = log(state, "Cleaning", "FAILED", str(e))
        return {**state, "current_node": "cleaning",
                "audit_trail": trail, "error": f"cleaning: {str(e)}"}


# ─────────────────────────────────────────────────
# NODE 5 — ML
# ─────────────────────────────────────────────────

def ml_node(state: ADAState) -> ADAState:
    try:
        trail = log(state, "ML", "Starting model training...")
        df = state.get("cleaned_df")
        if df is None:
            # Cleaning gave up after its retries; train on the raw frame rather
            # than lose the rest of the run.
            df = state["raw_df"]
            trail = log({"audit_trail": trail}, "ML", "No cleaned data — using raw dataset")
        ml_results = run_ml_agent(df, target_col=state.get("target_col"))
        best_model = ml_results["interpretation"]["best_model"]
        trail = log({"audit_trail": trail}, "ML", "Training complete",
                    f"Best model: {best_model}")
        return {**state, "ml_results": ml_results, "current_node": "ml",
                "audit_trail": trail, "error": None}
    except Exception as e:
        trail = log(state, "ML", "FAILED", str(e))
        return {**state, "current_node": "ml",
                "audit_trail": trail, "error": f"ml: {str(e)}"}


# ─────────────────────────────────────────────────
# NODE 6 — Explanation
# ─────────────────────────────────────────────────

def explain_node(state: ADAState) -> ADAState:
    try:
        trail = log(state, "Explain", "Starting SHAP analysis...")
        ml_results = state["ml_results"]
        explain_results = run_explain_agent(
            df=state["cleaned_df"],
            target_col=ml_results["target_column"],
            problem_type=ml_results["problem_type"],
            best_model_name=ml_results["interpretation"]["best_model"]
        )
        top_feature = list(explain_results["feature_importance"].keys())[0]
        trail = log({"audit_trail": trail}, "Explain", "Explanation complete",
                    f"Top feature: {top_feature}")
        return {**state, "explain_results": explain_results,
                "current_node": "explain", "audit_trail": trail, "error": None}
    except Exception as e:
        trail = log(state, "Explain", "FAILED — skipping", str(e))
        return {**state, "explain_results": {}, "current_node": "explain",
                "audit_trail": trail, "error": None}


# ─────────────────────────────────────────────────
# NODE 7 — Validator
# ─────────────────────────────────────────────────

def validator_node(state: ADAState) -> ADAState:
    try:
        trail = log(state, "Validator", "Running statistical tests on hypotheses...")
        # Tests run on the RAW frame, not the cleaned one: imputed values
        # fabricate certainty and bias p-values downward.
        validation = run_validator_agent(
            hypotheses=state.get("hypotheses", {}),
            df=state.get("raw_df")
        )
        results = validation.get("validation_results", [])
        confirmed = sum(1 for v in results if v.get("verdict") == "CONFIRMED")
        tested = sum(1 for v in results
                     if v.get("statistics", {}).get("p_value") is not None)
        trail = log({"audit_trail": trail}, "Validator", "Validation complete",
                    f"{confirmed}/{len(results)} confirmed ({tested} statistically tested)")
        return {**state, "validation_results": validation,
                "current_node": "validator", "audit_trail": trail, "error": None}
    except Exception as e:
        trail = log(state, "Validator", "FAILED — skipping", str(e))
        return {**state, "validation_results": {}, "current_node": "validator",
                "audit_trail": trail, "error": None}


# ─────────────────────────────────────────────────
# NODE 8 — Report
# ─────────────────────────────────────────────────

def report_node(state: ADAState) -> ADAState:
    try:
        trail = log(state, "Report", "Generating final report...")
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # `or {}` rather than a .get default: a skipped stage leaves its key
        # present but None.
        validation = state.get("validation_results") or {}
        ml_results = state.get("ml_results") or {}
        summary = {
            "eda": (state.get("eda_report") or "")[:500],
            "cleaning": state.get("cleaning_strategy") or {},
            "ml": {
                "problem_type": ml_results.get("problem_type"),
                "model_results": ml_results.get("model_results"),
                "interpretation": ml_results.get("interpretation"),
            },
            "explanation": (state.get("explain_results") or {}).get("interpretation", {}),
            "hypothesis_validation": {
                "results": validation.get("validation_results", []),
                "summary": validation.get("overall_summary", ""),
                "alpha": validation.get("alpha"),
                "correction_method": validation.get("correction_method")
            }
        }

        final_report = call_llm(
            prompt=(
                "Write a professional data analysis report including hypothesis "
                "validation.\n\n"
                "The hypothesis verdicts were computed in code from real statistical "
                "tests — quote the test name, effect size and adjusted p-value when "
                "you discuss one, and do not assign a verdict of your own. "
                "NOT_SUPPORTED means the data did not provide evidence for the claim, "
                "not that the claim is false. SIGNIFICANT_BUT_TRIVIAL means the result "
                "was statistically significant but too small to matter practically.\n\n"
                f"{json.dumps(summary, default=str)}"
            ),
            system="You are a senior data scientist writing professional analysis reports.",
            max_tokens=1500
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(OUTPUT_DIR, f"ADA_v3_report_{timestamp}.txt")
        audit_path = os.path.join(OUTPUT_DIR, f"ADA_v3_audit_{timestamp}.json")

        trail = log({"audit_trail": trail}, "Report", "Pipeline complete!",
                    f"Report saved to {report_path}")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(final_report)
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(trail, f, indent=2)

        return {**state, "final_report": final_report, "end_time": end_time,
                "current_node": "report", "audit_trail": trail, "error": None}
    except Exception as e:
        trail = log(state, "Report", "FAILED", str(e))
        return {**state, "current_node": "report",
                "audit_trail": trail, "error": f"report: {str(e)}"}


# ─────────────────────────────────────────────────
# NODE 9 — Error Handler
# ─────────────────────────────────────────────────

def error_handler_node(state: ADAState) -> ADAState:
    """
    Decide what happens after a node fails, and record the decision in
    state["retry_target"] for route_after_error to act on:
      - critical node      -> error stays set, pipeline ends
      - retries remaining  -> retry_target = failed node, it runs again
      - retries exhausted  -> retry_target = None, pipeline skips ahead
    """
    error = state.get("error", "")
    current_node = state.get("current_node", "")

    # The counter belongs to one node: a fresh failure elsewhere starts at 0
    # rather than inheriting attempts spent on an earlier, recovered node.
    attempts = (state.get("retry_count", 0)
                if state.get("retry_target") == current_node else 0)

    trail = log(state, "ErrorHandler", f"Handling error in {current_node}", error)

    if current_node in CRITICAL_NODES:
        trail = log({"audit_trail": trail}, "ErrorHandler",
                    "Critical node failed — stopping pipeline")
        return {**state, "audit_trail": trail, "retry_target": None,
                "error": f"CRITICAL: {error}"}

    if attempts >= MAX_RETRIES:
        trail = log({"audit_trail": trail}, "ErrorHandler",
                    f"Max retries reached for {current_node} — skipping")
        return {**state, "audit_trail": trail, "retry_count": 0,
                "retry_target": None, "error": None}

    trail = log({"audit_trail": trail}, "ErrorHandler",
                f"Retrying {current_node}", f"Attempt {attempts + 1} of {MAX_RETRIES}")
    return {**state, "audit_trail": trail, "retry_count": attempts + 1,
            "retry_target": current_node, "error": None}


# ─────────────────────────────────────────────────
# CONDITIONAL EDGES
# ─────────────────────────────────────────────────

Route = Literal["error_handler", "load_data", "hypothesis", "eda", "cleaning",
                "ml", "explain", "validator", "report", "end"]


def check_error(state: ADAState) -> Route:
    """After a normal node: on to the next node, or to the error handler."""
    if state.get("error"):
        return "error_handler"
    return NEXT_NODE.get(state.get("current_node", ""), "end")


def route_after_error(state: ADAState) -> Route:
    """After the error handler: stop, re-run the failed node, or skip past it."""
    if state.get("error"):
        return "end"
    if state.get("retry_target"):
        return state["retry_target"]
    return NEXT_NODE.get(state.get("current_node", ""), "end")


# ─────────────────────────────────────────────────
# BUILD THE GRAPH
# ─────────────────────────────────────────────────

def build_ada_graph():
    graph = StateGraph(ADAState)

    graph.add_node("load_data", load_data_node)
    graph.add_node("hypothesis", hypothesis_node)
    graph.add_node("eda", eda_node)
    graph.add_node("cleaning", cleaning_node)
    graph.add_node("ml", ml_node)
    graph.add_node("explain", explain_node)
    graph.add_node("validator", validator_node)
    graph.add_node("report", report_node)
    graph.add_node("error_handler", error_handler_node)

    graph.set_entry_point("load_data")

    route_map = {
        "error_handler": "error_handler",
        "load_data": "load_data",
        "hypothesis": "hypothesis",
        "eda": "eda",
        "cleaning": "cleaning",
        "ml": "ml",
        "explain": "explain",
        "validator": "validator",
        "report": "report",
        "end": END
    }

    all_nodes = ["load_data", "hypothesis", "eda", "cleaning",
                 "ml", "explain", "validator", "report"]

    for node in all_nodes:
        graph.add_conditional_edges(node, check_error, route_map)

    graph.add_conditional_edges("error_handler", route_after_error, route_map)

    return graph.compile()


# ─────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────

def initial_state(filepath: str, target_col: Optional[str] = None) -> ADAState:
    return {
        "filepath": filepath,
        "target_col": target_col,
        "raw_df": None,
        "cleaned_df": None,
        "hypotheses": None,
        "eda_stats": None,
        "eda_report": None,
        "cleaning_strategy": None,
        "ml_results": None,
        "explain_results": None,
        "validation_results": None,
        "final_report": None,
        "current_node": None,
        "error": None,
        "retry_count": 0,
        "retry_target": None,
        "audit_trail": [],
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": None
    }


def run_ada(filepath: str, target_col: Optional[str] = None,
            on_step: Optional[Callable[[ADAState], None]] = None) -> ADAState:
    """
    Run the full pipeline and return the final state.

    on_step, if given, is called with the state after every node finishes —
    the Streamlit UI uses it to drive a real progress bar.
    """
    print("=" * 55)
    print("   ADA v3.0 — Hypothesis-Driven Analysis")
    print("   Powered by Claude AI")
    print("=" * 55)

    ada_graph = build_ada_graph()

    # 8 nodes, each of which may run up to 1 + MAX_RETRIES times with an
    # error-handler step between attempts. LangGraph's default limit of 25
    # steps is too low for a run that retries more than once or twice.
    config = {"recursion_limit": 8 * (1 + MAX_RETRIES) * 2 + 1}

    final_state = None
    for final_state in ada_graph.stream(initial_state(filepath, target_col),
                                        config=config, stream_mode="values"):
        if on_step is not None:
            on_step(final_state)

    print("\n" + "=" * 55)
    print("   ADA v3.0 Complete!")
    print("=" * 55)

    return final_state


# Backwards-compatible name from v2.0.
run_ada_v2 = run_ada
