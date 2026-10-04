from typing import Any, Optional, TypedDict


class ADAState(TypedDict):
    """
    Updated state for ADA v3.0.
    Adds hypothesis and validation fields.
    """

    # ── Input ────────────────────────────────────────
    filepath: str
    target_col: Optional[str]

    # ── Data ─────────────────────────────────────────
    raw_df: Optional[Any]
    cleaned_df: Optional[Any]

    # ── Agent outputs ─────────────────────────────────
    hypotheses: Optional[dict]           # NEW — formed before EDA
    eda_stats: Optional[dict]
    eda_report: Optional[str]
    cleaning_strategy: Optional[dict]
    ml_results: Optional[dict]
    explain_results: Optional[dict]
    validation_results: Optional[dict]   # NEW — tested after explain
    final_report: Optional[str]

    # ── Pipeline control ──────────────────────────────
    current_node: Optional[str]
    error: Optional[str]
    retry_count: int
    retry_target: Optional[str]          # node the error handler is re-running
    audit_trail: list

    # ── Metadata ──────────────────────────────────────
    start_time: Optional[str]
    end_time: Optional[str]
