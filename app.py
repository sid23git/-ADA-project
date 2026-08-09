import streamlit as st
import pandas as pd
import os
from datetime import datetime
from ada_graph import run_ada_v2

# ─── Page Config ──────────────────────────────────────
st.set_page_config(
    page_title="ADA v2.0 — Autonomous Data Analysis Agent",
    page_icon="🤖",
    layout="wide"
)

# ─── Session State Init ───────────────────────────────
if "pipeline_running" not in st.session_state:
    st.session_state.pipeline_running = False

# ─── Header ───────────────────────────────────────────
st.title("🤖 ADA v2.0 — Autonomous Data Analysis Agent")
st.markdown(
    "Powered by **LangGraph multi-agent architecture**. "
    "Upload any CSV and ADA autonomously analyzes, cleans, "
    "models, and explains it — just like a junior data scientist."
)
st.divider()

# ─── Sidebar ──────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    uploaded_file = st.file_uploader(
        "Upload your CSV dataset",
        type=["csv"]
    )

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
    st.markdown("2. 🔍 Exploratory Analysis")
    st.markdown("3. 🧹 Data Cleaning")
    st.markdown("4. 🤖 Model Training")
    st.markdown("5. 💡 SHAP Explanation")
    st.markdown("6. 📄 Final Report")

    st.divider()
    st.markdown("**v2.0 upgrades**")
    st.markdown("✅ LangGraph state graph")
    st.markdown("✅ Shared agent state")
    st.markdown("✅ Self-correction loop")
    st.markdown("✅ Full audit trail")

# ─── Main Area ────────────────────────────────────────
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
    col4.metric("Numeric Columns",
                len(df_preview.select_dtypes(include="number").columns))

    st.dataframe(df_preview.head(20), use_container_width=True)
    st.caption("Showing first 20 rows. Press 'Run ADA Pipeline' to start.")

else:
    # Save uploaded file temporarily
    temp_path = f"data/uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    target = target_col_input.strip() if target_col_input.strip() else None

    progress = st.progress(0)
    status = st.status("🚀 ADA v2.0 pipeline starting...", expanded=True)

    try:
        st.session_state.pipeline_running = True

        with status:
            st.write("📂 Loading dataset into LangGraph state...")
            progress.progress(10)

            st.write("🔍 Running Exploratory Data Analysis...")
            progress.progress(25)

            st.write("🧹 Cleaning data autonomously...")
            progress.progress(45)

            st.write("🤖 Training and comparing ML models...")
            progress.progress(65)

            st.write("💡 Computing SHAP explanations...")
            progress.progress(80)

            st.write("📄 Generating final report...")
            progress.progress(90)

            # Run ADA v2.0
            state = run_ada_v2(
                filepath=temp_path,
                target_col=target
            )

            progress.progress(100)

        # Check for critical errors
        if state.get("error") and "CRITICAL" in str(state.get("error", "")):
            status.update(label="❌ Pipeline failed", state="error")
            st.error(f"Pipeline error: {state['error']}")
        else:
            status.update(
                label="✅ ADA v2.0 Pipeline Complete!",
                state="complete"
            )

            # ─── Results Tabs ─────────────────────────────
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
                "🧠 Hypotheses",
                "📊 EDA",
                "🧹 Cleaning",
                "🤖 ML Results",
                "💡 Explanation",
                "✅ Validation",
                "📄 Final Report",
                "🔍 Audit Trail"
            ])

            # Tab 1 — Hypotheses (NEW in v3.0)
            with tab1:
                st.subheader("🧠 Hypotheses — Formed Before Analysis")
                st.markdown(
                    "These hypotheses were generated **before** any "
                    "analysis ran — based only on column names and types. "
                    "This mimics how a real scientist thinks."
                )

                hypotheses = state.get("hypotheses" or {})

                if not hypotheses:
                    st.warning("Hypothesis agent was skipped.")
                else:
                    st.info(f"**Dataset identified as:** "
                            f"{hypotheses.get('dataset_type', 'N/A')}")

                    st.markdown(f"**Analysis strategy:** "
                                f"{hypotheses.get('analysis_strategy', 'N/A')}")

                    st.markdown("**Generated Hypotheses:**")
                    for h in hypotheses.get("hypotheses", []):
                        confidence_color = (
                            "🟢" if h.get("confidence") == "high"
                            else "🟡" if h.get("confidence") == "medium"
                            else "🔴"
                        )
                        with st.expander(
                                f"{confidence_color} {h['id']} — {h['hypothesis']}"
                        ):
                            st.markdown(f"**Confidence:** {h.get('confidence', 'N/A')}")
                            st.markdown(f"**Reasoning:** {h.get('reasoning', 'N/A')}")
                            st.markdown(f"**Expected evidence:** "
                                        f"{h.get('expected_evidence', 'N/A')}")
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
                        missing_df = pd.DataFrame(
                            list(missing.items()),
                            columns=["Column", "Missing Count"]
                        )
                        st.dataframe(missing_df, use_container_width=True)
                    else:
                        st.success("No missing values found!")

                st.markdown("**AI EDA Report**")
                st.markdown(state.get("eda_report", ""))

            # Tab 3 — Cleaning
            with tab3:
                st.subheader("Data Cleaning")
                strategy = state.get("cleaning_strategy") or {}

                st.markdown("**AI Reasoning**")
                st.info(strategy.get("reasoning", "N/A"))

                col1, col2 = st.columns(2)
                with col1:
                    raw_df = state.get("raw_df")
                    raw_str = f"{raw_df.shape}" if raw_df is not None else "N/A"
                    st.metric("Original Shape", raw_str)
                with col2:
                    clean_df = state.get("cleaned_df")
                    clean_str = f"{clean_df.shape}" if clean_df is not None else "N/A"
                    st.metric("Cleaned Shape", clean_str)

                if strategy.get("columns_to_drop"):
                    st.markdown("**Columns Dropped**")
                    st.write(strategy.get("columns_to_drop"))

                if strategy.get("missing_value_strategies"):
                    st.markdown("**Imputation Strategies Applied**")
                    strat_df = pd.DataFrame(
                        list(strategy["missing_value_strategies"].items()),
                        columns=["Column", "Strategy"]
                    )
                    st.dataframe(strat_df, use_container_width=True)

            # Tab 4 — ML Results
            with tab4:
                st.subheader("Machine Learning Results")
                ml = state.get("ml_results") or {}
                interp = ml.get("interpretation", {})

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Problem Type",
                              ml.get("problem_type", "N/A").capitalize())
                    st.metric("Best Model",
                              interp.get("best_model", "N/A"))
                with col2:
                    st.markdown("**AI Reasoning**")
                    st.info(interp.get("reasoning", "N/A"))

                st.markdown("**Model Comparison**")
                model_results = ml.get("model_results", {})
                if model_results:
                    model_df = pd.DataFrame(model_results).T
                    st.dataframe(model_df, use_container_width=True)

                if interp.get("concerns"):
                    st.warning(f"⚠️ {interp.get('concerns')}")

                if interp.get("recommendation"):
                    st.markdown("**Next Steps**")
                    st.success(interp.get("recommendation"))

            # Tab 5 — Explanation
            with tab5:
                st.subheader("Model Explanation (SHAP)")
                explain = state.get("explain_results" or {})
                exp_interp = explain.get("interpretation", {})

                if not explain:
                    st.warning("Explanation agent was skipped due to an error.")
                else:
                    st.markdown("**Plain English Summary**")
                    st.info(exp_interp.get("plain_english_summary", "N/A"))

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Top 3 Most Influential Features**")
                        top_feats = exp_interp.get("top_3_features", [])
                        for i, feat in enumerate(top_feats, 1):
                            st.markdown(
                                f"**{i}. {feat.get('feature', '')}** — "
                                f"{feat.get('explanation', '')}"
                            )
                    with col2:
                        st.markdown("**Feature Importance (SHAP)**")
                        importance = explain.get("feature_importance", {})
                        if importance:
                            imp_df = pd.DataFrame(
                                list(importance.items())[:10],
                                columns=["Feature", "SHAP Importance"]
                            )
                            st.dataframe(imp_df, use_container_width=True)

                    shap_path = "outputs/shap_summary.png"
                    if os.path.exists(shap_path):
                        st.markdown("**SHAP Summary Plot**")
                        st.image(shap_path, use_container_width=True)

                    if exp_interp.get("business_insight"):
                        st.markdown("**Business Insight**")
                        st.success(exp_interp.get("business_insight"))

            # Tab 6 — Validation (NEW in v3.0)
            with tab6:
                st.subheader("✅ Hypothesis Validation Results")
                st.markdown(
                    "Each hypothesis was tested against the actual "
                    "findings from EDA, ML, and SHAP analysis."
                )

                validation = state.get("validation_results") or {}

                if not validation:
                    st.warning("Validation agent was skipped.")
                else:
                    # Summary metrics
                    results = validation.get("validation_results", [])
                    confirmed = sum(1 for v in results
                                    if v.get("verdict") == "CONFIRMED")
                    rejected = sum(1 for v in results
                                   if v.get("verdict") == "REJECTED")
                    inconclusive = sum(1 for v in results
                                       if v.get("verdict") == "INCONCLUSIVE")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("✅ Confirmed", confirmed)
                    col2.metric("❌ Rejected", rejected)
                    col3.metric("⚠️ Inconclusive", inconclusive)

                    st.divider()

                    # Individual results
                    for v in results:
                        verdict = v.get("verdict", "")
                        icon = ("✅" if verdict == "CONFIRMED"
                                else "❌" if verdict == "REJECTED"
                        else "⚠️")
                        color = ("success" if verdict == "CONFIRMED"
                                 else "error" if verdict == "REJECTED"
                        else "warning")

                        with st.expander(
                                f"{icon} {v.get('id')} — {verdict}: "
                                f"{v.get('hypothesis', '')[:80]}..."
                        ):
                            if color == "success":
                                st.success(f"**Evidence:** {v.get('evidence', 'N/A')}")
                            elif color == "error":
                                st.error(f"**Evidence:** {v.get('evidence', 'N/A')}")
                            else:
                                st.warning(f"**Evidence:** {v.get('evidence', 'N/A')}")
                            st.markdown(f"**Insight:** {v.get('insight', 'N/A')}")

                    st.divider()
                    st.markdown("**Overall Summary**")
                    st.info(validation.get("overall_summary", "N/A"))

                    if validation.get("most_surprising"):
                        st.markdown("**Most Surprising Finding**")
                        st.success(validation.get("most_surprising"))

                    if validation.get("scientific_contribution"):
                        st.markdown("**Scientific Contribution**")
                        st.markdown(validation.get("scientific_contribution"))

            # Tab 7 — Final Report
            with tab7:
                st.subheader("Final Report")
                report = state.get("final_report", "")
                st.markdown(report)

                st.download_button(
                    label="⬇️ Download Report",
                    data=report,
                    file_name=f"ADA_report_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )

            # Tab 7 — Audit Trail (NEW in v2.0)
            with tab7:
                st.subheader("🔍 Audit Trail")
                st.markdown(
                    "Every decision made by every agent — "
                    "logged automatically by LangGraph state."
                )

                audit = state.get("audit_trail", [])
                if audit:
                    audit_df = pd.DataFrame(audit)
                    st.dataframe(audit_df, use_container_width=True)

                    st.download_button(
                        label="⬇️ Download Audit Trail (JSON)",
                        data=str(audit),
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
