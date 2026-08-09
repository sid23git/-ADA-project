# TODO v4.0 — refactor nodes into separate files under nodes/ folder

import pandas as pd
from datetime import datetime
from typing import Literal
from langgraph.graph import StateGraph, END
from utils.ada_state import ADAState
from utils.llm import call_llm
from agents.eda_agent import analyze_dataframe, run_eda_agent
from agents.cleaning_agent import run_cleaning_agent
from agents.ml_agent import run_ml_agent
from agents.explain_agent import run_explain_agent
from agents.hypothesis_agent import run_hypothesis_agent
from agents.validator_agent import run_validator_agent
from dotenv import load_dotenv
import json

load_dotenv()


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
        eda_report = run_eda_agent(state["filepath"])
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
        cleaned_df, strategy = run_cleaning_agent(df, eda_stats)
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
        ml_results = run_ml_agent(state["cleaned_df"], target_col=state.get("target_col"))
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
        trail = log(state, "Validator", "Testing hypotheses against findings...")
        validation = run_validator_agent(
            hypotheses=state.get("hypotheses", {}),
            eda_stats=state.get("eda_stats", {}),
            ml_results=state.get("ml_results", {}),
            explain_results=state.get("explain_results", {})
        )
        confirmed = sum(1 for v in validation.get("validation_results", [])
                       if v.get("verdict") == "CONFIRMED")
        total = len(validation.get("validation_results", []))
        trail = log({"audit_trail": trail}, "Validator", "Validation complete",
                    f"{confirmed}/{total} hypotheses confirmed")
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

        validation = state.get("validation_results", {})
        summary = {
            "eda": state.get("eda_report", "")[:500],
            "cleaning": state.get("cleaning_strategy", {}),
            "ml": state.get("ml_results", {}).get("interpretation", {}),
            "explanation": state.get("explain_results", {}).get("interpretation", {}),
            "hypothesis_validation": {
                "results": validation.get("validation_results", []),
                "summary": validation.get("overall_summary", "")
            }
        }

        final_report = call_llm(
            prompt=f"Write a professional data analysis report including hypothesis validation: {json.dumps(summary)}",
            system="You are a senior data scientist writing professional analysis reports.",
            max_tokens=1500
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"outputs/ADA_v3_report_{timestamp}.txt"
        audit_path = f"outputs/ADA_v3_audit_{timestamp}.json"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(final_report)
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(state.get("audit_trail", []), f, indent=2)

        trail = log({"audit_trail": trail}, "Report", "Pipeline complete!",
                    f"Report saved to {report_path}")

        return {**state, "final_report": final_report, "end_time": end_time,
                "current_node": "report", "audit_trail": trail, "error": None}
    except Exception as e:
        trail = log(state, "Report", "FAILED", str(e))
        return {**state, "current_node": "report",
                "audit_trail": trail, "error": f"report: {str(e)}"}


# ─────────────────────────────────────────────────
# NODE 9 — Error Handler
# ─
# ────────────────────────────────────────────────

def error_handler_node(state: ADAState) -> ADAState:
    error = state.get("error", "")
    retry_count = state.get("retry_count", 0)
    current_node = state.get("current_node", "")

    trail = log(state, "ErrorHandler", f"Handling error in {current_node}", error)

    if current_node in ["load_data"]:
        trail = log({"audit_trail": trail}, "ErrorHandler",
                    "Critical node failed — stopping pipeline")
        return {**state, "audit_trail": trail, "error": f"CRITICAL: {error}"}

    if retry_count >= 2:
        trail = log({"audit_trail": trail}, "ErrorHandler",
                    f"Max retries reached for {current_node} — skipping")
        return {**state, "audit_trail": trail, "retry_count": 0, "error": None}

    trail = log({"audit_trail": trail}, "ErrorHandler",
                f"Retrying {current_node}", f"Attempt {retry_count + 1} of 2")
    return {**state, "audit_trail": trail, "retry_count": retry_count + 1, "error": None}


# ─────────────────────────────────────────────────
# CONDITIONAL EDGES
# ─────────────────────────────────────────────────

def check_error(state: ADAState) -> Literal[
    "error_handler", "hypothesis", "eda", "cleaning",
    "ml", "explain", "validator", "report", "end"
]:
    if state.get("error"):
        return "error_handler"

    routing = {
        "load_data": "hypothesis",
        "hypothesis": "eda",
        "eda": "cleaning",
        "cleaning": "ml",
        "ml": "explain",
        "explain": "validator",
        "validator": "report",
        "report": "end"
    }
    return routing.get(state.get("current_node", ""), "end")


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

    graph.add_conditional_edges("error_handler", check_error, route_map)

    return graph.compile()


# ─────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────

def run_ada_v2(filepath: str, target_col: str = None) -> dict:
    print("=" * 55)
    print("   ADA v3.0 — Hypothesis-Driven Analysis")
    print("   Powered by Claude AI")
    print("=" * 55)

    ada_graph = build_ada_graph()

    initial_state: ADAState = {
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
        "audit_trail": [],
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": None
    }

    final_state = ada_graph.invoke(initial_state)

    print("\n" + "=" * 55)
    print("   ADA v3.0 Complete!")
    print("=" * 55)

    return final_state
