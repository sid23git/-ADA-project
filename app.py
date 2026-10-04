import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st

from ada_graph import run_ada

# node name -> (label shown when it finishes, progress % at that point)
STAGES = {
    "load_data": ("📂 Dataset loaded", 5),
    "hypothesis": ("🧠 Hypotheses generated", 15),
    "eda": ("🔍 EDA complete", 30),
    "cleaning": ("🧹 Data cleaned", 40),
    "ml": ("🤖 Models trained", 70),
    "explain": ("💡 SHAP explanations computed", 82),
    "validator": ("✅ Hypotheses validated", 90),
    "report": ("📄 Final report written", 99),
}

st.set_page_config(
    page_title="ADA v3.0 — Autonomous Data Analysis Agent",
    page_icon="🤖",
    layout="wide"
)

if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False

st.title("🤖 ADA v3.0 — Autonomous Data Analysis Agent")
st.markdown(
    "Powered by **LangGraph multi-agent architecture** and **Claude AI**. "
    "Upload any CSV and ADA autonomously analyzes, cleans, models, and explains it."
)
st.divider()

with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
    target_col_input = st.text_input(
        "Target column (optional)",
        placeholder="e.g. Survived, Price, Churn",
        help="Leave empty and ADA will use the last column"
    )
    run_button = st.button(
        "▶ Run ADA Pipeline",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None or st.session_state.pipeline_running
    )
    st.divider()
    st.markdown("**Pipeline stages**")
    st.markdown("1. 📂 Load Data")
    st.markdown("2. 🧠 Hypothesis Generation")
    st.markdown("3. 🔍 Exploratory Analysis")
    st.markdown("4. 🧹 Data Cleaning")
    st.markdown("5. 🤖 Model Training")
    st.markdown("6. 💡 SHAP Explanation")
    st.markdown("7. ✅ Hypothesis Validation")
    st.markdown("8. 📄 Final Report")
    st.divider()
    st.markdown("**v3.0 features**")
    st.markdown("✅ LangGraph state graph")
    st.markdown("✅ Hypothesis-driven analysis")
    st.markdown("✅ Self-correction loop")
    st.markdown("✅ Full audit trail")

if uploaded_file is None:
    st.info("👈 Upload a CSV file from the sidebar to get started.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🏥 Healthcare")
        st.markdown("Predict patient outcomes, identify risk factors")
    with col2:
        st.markdown("### 💰 Finance")
        st.markdown("Detect fraud, predict loan defaults")
    with col3:
        st.markdown("### 🛒 Retail")
        st.markdown("Predict churn, forecast sales")

elif not run_button:
    df_preview = pd.read_csv(uploaded_file)
    uploaded_file.seek(0)
    st.subheader("📋 Dataset Preview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df_preview.shape[0])
    col2.metric("Columns", df_preview.shape[1])
    col3.metric("Missing Values", int(df_preview.isnull().sum().sum()))
    col4.metric("Numeric Columns", len(df_preview.select_dtypes(include="number").columns))
    st.dataframe(df_preview.head(20), use_container_width=True)
    st.caption("Showing first 20 rows. Press 'Run ADA Pipeline' to start.")

else:
    temp_path = f"data/uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    os.makedirs("data", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    target = target_col_input.strip() if target_col_input.strip() else None
    progress = st.progress(0)
    status = st.status("🚀 ADA v3.0 pipeline starting...", expanded=True)

    try:
        st.session_state.pipeline_running = True

        def on_step(step_state):
            # Called by the graph after each node really finishes, so the bar
            # tracks the pipeline instead of racing ahead of it.
            trail = step_state.get("audit_trail") or []
            if not trail:
                return
            last = trail[-1]
            # After a failure current_node still names the failed node, so the
            # latest audit entry is what says whether this step succeeded.
            if step_state.get("error") or last["agent"] == "ErrorHandler":
                status.write(f"⚠️ [{last['agent']}] {last['action']}")
                return
            node = step_state.get("current_node")
            if node in STAGES:
                label, pct = STAGES[node]
                status.write(label)
                progress.progress(pct)

        with status:
            state = run_ada(filepath=temp_path, target_col=target, on_step=on_step)
            progress.progress(100)

        if state.get("error") and "CRITICAL" in str(state.get("error", "")):
            status.update(label="❌ Pipeline failed", state="error")
            st.error(f"Critical error: {state['error']}")
        else:
            status.update(label="✅ ADA v3.0 Pipeline Complete!", state="complete")

            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
                "🧠 Hypotheses", "📊 EDA", "🧹 Cleaning",
                "🤖 ML Results", "💡 Explanation",
                "✅ Validation", "📄 Final Report", "🔍 Audit Trail"
            ])

            # Tab 1 — Hypotheses
            with tab1:
                st.subheader("🧠 Hypotheses — Formed Before Analysis")
                st.markdown("Generated **before** any analysis — based only on column names and types.")
                hypotheses = state.get("hypotheses") or {}
                if not hypotheses or not hypotheses.get("hypotheses"):
                    st.warning("Hypothesis agent was skipped or failed.")
                else:
                    st.info(f"**Dataset identified as:** {hypotheses.get('dataset_type', 'N/A')}")
                    st.markdown(f"**Strategy:** {hypotheses.get('analysis_strategy', 'N/A')}")
                    for h in hypotheses.get("hypotheses", []):
                        icon = {"high": "🟢", "medium": "🟡"}.get(h.get("confidence"), "🔴")
                        with st.expander(f"{icon} {h['id']} — {h['hypothesis']}"):
                            st.markdown(f"**Confidence:** {h.get('confidence', 'N/A')}")
                            st.markdown(f"**Reasoning:** {h.get('reasoning', 'N/A')}")
                            st.markdown(f"**Expected evidence:** {h.get('expected_evidence', 'N/A')}")
                            spec = h.get("test_spec") or {}
                            if spec:
                                variables = ", ".join(
                                    f"`{spec[k]}`" for k in
                                    ("group_var", "outcome_var", "var_a", "var_b")
                                    if spec.get(k)
                                )
                                st.markdown(
                                    f"**Planned test:** `{spec.get('claim_type')}` "
                                    f"({spec.get('direction')}"
                                    + (f", focus `{spec['focus_level']}`"
                                       if spec.get("focus_level") else "")
                                    + f") on {variables}"
                                )
                    if hypotheses.get("unresolvable"):
                        st.warning(
                            "These hypotheses cannot be tested against this dataset:\n\n"
                            + "\n".join(f"- {n}" for n in hypotheses["unresolvable"])
                        )

            # Tab 2 — EDA
            with tab2:
                st.subheader("Exploratory Data Analysis")
                eda_stats = state.get("eda_stats") or {}
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Dataset Shape**")
                    shape = eda_stats.get("shape", {})
                    st.metric("Rows", shape.get("rows", "N/A"))
                    st.metric("Columns", shape.get("columns", "N/A"))
                with col2:
                    st.markdown("**Missing Values**")
                    missing = eda_stats.get("missing_values", {})
                    if missing:
                        st.dataframe(pd.DataFrame(list(missing.items()), columns=["Column", "Missing"]),
                                     use_container_width=True)
                    else:
                        st.success("No missing values found!")
                st.markdown("**AI EDA Report**")
                st.markdown(state.get("eda_report") or "EDA report not generated.")

            # Tab 3 — Cleaning
            with tab3:
                st.subheader("Data Cleaning")
                strategy = state.get("cleaning_strategy") or {}
                st.markdown("**AI Reasoning**")
                st.info(strategy.get("reasoning", "N/A"))
                col1, col2 = st.columns(2)
                raw_df = state.get("raw_df")
                clean_df = state.get("cleaned_df")
                with col1:
                    st.metric("Original Shape", str(raw_df.shape) if raw_df is not None else "N/A")
                with col2:
                    st.metric("Cleaned Shape", str(clean_df.shape) if clean_df is not None else "N/A")
                if strategy.get("columns_to_drop"):
                    st.markdown("**Columns Dropped**")
                    st.write(strategy.get("columns_to_drop"))
                if strategy.get("missing_value_strategies"):
                    st.markdown("**Imputation Strategies**")
                    st.dataframe(pd.DataFrame(list(strategy["missing_value_strategies"].items()),
                                              columns=["Column", "Strategy"]),
                                 use_container_width=True)

            # Tab 4 — ML
            with tab4:
                st.subheader("Machine Learning Results")
                ml = state.get("ml_results") or {}
                interp = ml.get("interpretation", {})
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Problem Type", (ml.get("problem_type") or "N/A").capitalize())
                    st.metric("Best Model", interp.get("best_model", "N/A"))
                with col2:
                    st.markdown("**AI Reasoning**")
                    st.info(interp.get("reasoning", "N/A"))
                model_results = ml.get("model_results", {})
                if model_results:
                    st.markdown("**Model Comparison**")
                    st.dataframe(pd.DataFrame(model_results).T, use_container_width=True)
                    if interp.get("selection_metric"):
                        st.caption(
                            f"Best model selected in code by highest `{interp['selection_metric']}` "
                            f"(cross-validated); the AI only explains the choice."
                        )
                if interp.get("concerns"):
                    st.warning(f"⚠️ {interp.get('concerns')}")
                if interp.get("recommendation"):
                    st.success(interp.get("recommendation"))

            # Tab 5 — Explanation
            with tab5:
                st.subheader("Model Explanation (SHAP)")
                explain = state.get("explain_results") or {}
                exp_interp = explain.get("interpretation", {})
                if not explain:
                    st.warning("Explanation agent was skipped.")
                else:
                    st.info(exp_interp.get("plain_english_summary", "N/A"))
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Top 3 Features**")
                        for i, feat in enumerate(exp_interp.get("top_3_features", []), 1):
                            st.markdown(f"**{i}. {feat.get('feature', '')}** — {feat.get('explanation', '')}")
                    with col2:
                        importance = explain.get("feature_importance", {})
                        if importance:
                            st.dataframe(pd.DataFrame(list(importance.items())[:10],
                                                      columns=["Feature", "SHAP"]),
                                         use_container_width=True)
                    plot_path = explain.get("shap_plot_path")
                    if plot_path and os.path.exists(plot_path):
                        st.image(plot_path, use_container_width=True)
                    if exp_interp.get("business_insight"):
                        st.success(exp_interp.get("business_insight"))

            # Tab 6 — Validation
            with tab6:
                st.subheader("✅ Hypothesis Validation")
                validation = state.get("validation_results") or {}
                if not validation or not validation.get("validation_results"):
                    st.warning("Validation agent was skipped.")
                else:
                    results = validation.get("validation_results", [])
                    st.caption(
                        f"Verdicts are computed in code from statistical tests run on the "
                        f"raw data (α={validation.get('alpha', 0.05)}, p-values adjusted "
                        f"with {validation.get('correction_method', 'Benjamini-Hochberg')}). "
                        f"The language model only writes the interpretation."
                    )

                    def _count(*verdicts):
                        return sum(1 for v in results if v.get("verdict") in verdicts)

                    cols = st.columns(5)
                    cols[0].metric("✅ Confirmed", _count("CONFIRMED"))
                    cols[1].metric("❌ Rejected", _count("REJECTED"))
                    cols[2].metric("🔍 Trivial effect", _count("SIGNIFICANT_BUT_TRIVIAL"))
                    cols[3].metric("➖ Not supported", _count("NOT_SUPPORTED"))
                    cols[4].metric("🚫 Not tested",
                                   _count("NOT_TESTABLE", "INSUFFICIENT_DATA"))
                    st.divider()

                    ICONS = {
                        "CONFIRMED": "✅", "REJECTED": "❌",
                        "SIGNIFICANT_BUT_TRIVIAL": "🔍", "NOT_SUPPORTED": "➖",
                        "INSUFFICIENT_DATA": "🚫", "NOT_TESTABLE": "🚫",
                    }
                    for v in results:
                        verdict = v.get("verdict", "")
                        icon = ICONS.get(verdict, "⚠️")
                        title = v.get("hypothesis", "")[:80]
                        with st.expander(f"{icon} {v.get('id')} — {verdict}: {title}..."):
                            if verdict == "CONFIRMED":
                                st.success(f"**Evidence:** {v.get('evidence', 'N/A')}")
                            elif verdict == "REJECTED":
                                st.error(f"**Evidence:** {v.get('evidence', 'N/A')}")
                            else:
                                st.warning(f"**Evidence:** {v.get('evidence', 'N/A')}")

                            stats_block = v.get("statistics") or {}
                            if stats_block.get("p_value") is not None:
                                rows = [
                                    ("Test", stats_block.get("test_used")),
                                    ("Statistic", f"{stats_block.get('statistic'):.4f}"
                                     if stats_block.get("statistic") is not None else "N/A"),
                                    ("p-value", f"{stats_block.get('p_value'):.5f}"),
                                    (f"Adjusted q ({stats_block.get('correction_method', 'BH')})",
                                     f"{stats_block.get('p_adjusted'):.5f}"
                                     if stats_block.get("p_adjusted") is not None else "N/A"),
                                ]
                                effect = stats_block.get("effect_value")
                                if effect is not None:
                                    ci = stats_block.get("effect_ci")
                                    ci_text = (f" [95% CI {ci[0]:.3f}, {ci[1]:.3f}]"
                                               if ci else "")
                                    rows.append((stats_block.get("effect_name", "Effect size"),
                                                 f"{effect:.4f}{ci_text}"))
                                rows.append(("Rows used",
                                             f"{stats_block.get('n_used')} of "
                                             f"{stats_block.get('n_total')}"))
                                if stats_block.get("assumption_note"):
                                    rows.append(("Assumptions", stats_block["assumption_note"]))
                                st.dataframe(
                                    pd.DataFrame(rows, columns=["", "Value"]),
                                    hide_index=True, use_container_width=True,
                                )
                                if stats_block.get("practically_significant") is False:
                                    st.caption(
                                        "⚠️ The effect is below the conventional threshold "
                                        "for practical significance."
                                    )

                            st.markdown(f"**Insight:** {v.get('insight', 'N/A')}")
                            if v.get("caveat"):
                                st.caption(f"Caveat: {v['caveat']}")

                    st.info(validation.get("overall_summary", ""))
                    if validation.get("most_surprising"):
                        st.success(validation.get("most_surprising"))

            # Tab 7 — Final Report
            with tab7:
                st.subheader("Final Report")
                report = state.get("final_report") or ""
                if report:
                    st.markdown(report)
                    st.download_button(
                        label="⬇️ Download Report",
                        data=report,
                        file_name=f"ADA_report_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )
                else:
                    st.warning("Report was not generated. Check the audit trail for details.")

            # Tab 8 — Audit Trail
            with tab8:
                st.subheader("🔍 Audit Trail")
                st.markdown("Every decision made by every agent — logged automatically.")
                audit = state.get("audit_trail", [])
                if audit:
                    st.dataframe(pd.DataFrame(audit), use_container_width=True)
                    st.download_button(
                        label="⬇️ Download Audit Trail",
                        data=json.dumps(audit, indent=2),
                        file_name=f"ADA_audit_{datetime.now().strftime('%Y%m%d')}.json",
                        mime="application/json"
                    )
                else:
                    st.info("No audit trail entries found.")

    except Exception as e:
        status.update(label="❌ Pipeline failed", state="error")
        st.error(f"Error: {str(e)}")
        st.exception(e)

    finally:
        st.session_state.pipeline_running = False
        if os.path.exists(temp_path):
            os.remove(temp_path)



