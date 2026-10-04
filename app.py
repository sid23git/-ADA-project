"""ADA — Streamlit front end for the AI data-science team."""

import json
import os
import queue
import threading
import time

import altair as alt
import pandas as pd
import streamlit as st

from ada import llm
from ada.config import LEAD_MODEL, WORKER_MODEL, Budget, Settings
from ada.offline import ScriptedClaude
from ada.run import investigate
from evals.scenarios import SCENARIOS, load

st.set_page_config(page_title="ADA — AI Data Science Team", page_icon="🧪", layout="wide")

AGENT_ICONS = {"lead": "🧭", "data_quality": "🧹", "statistician": "📐", "ml_engineer": "🤖",
               "critic": "🕵️", "reporter": "📝", "verifier": "⚖️", "system": "⚙️"}
KIND_ICONS = {"plan": "planned", "start": "started", "tool_call": "→", "evidence": "📎",
              "submitted": "📤", "accepted": "✅", "rejected": "❌", "demoted": "⬇️",
              "error": "⚠️", "done": "📝", "revise": "✏️", "target": "🎯", "finish": "🏁"}
SAMPLE = "titanic (data/sample.csv)"


def _scenario_options():
    return [SAMPLE] + list(SCENARIOS)


def _load_choice(choice):
    if choice == SAMPLE:
        return (pd.read_csv("data/sample.csv"), "What determined who survived the Titanic, and how "
                "confident can we be?", "Survived")
    s = load(choice)
    return s.df, s.question, s.target


# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🧪 Investigation")
    source = st.radio("Data", ["Demo dataset", "Upload CSV"], horizontal=True)
    if source == "Upload CSV":
        upload = st.file_uploader("CSV file", type=["csv"])
        df = pd.read_csv(upload) if upload else None
        default_q, default_target = "", None
    else:
        choice = st.selectbox("Dataset", _scenario_options(),
                              help="The eval scenarios have planted ground truth — see evals/scenarios.py")
        df, default_q, default_target = _load_choice(choice)

    question = st.text_area("Business question", value=default_q, height=90,
                            placeholder="e.g. What drives churn, and who should we target?")
    target = None
    if df is not None:
        options = ["(let the lead decide)"] + list(df.columns)
        idx = options.index(default_target) if default_target in options else 0
        picked = st.selectbox("Target column", options, index=idx)
        target = None if picked == options[0] else picked

    has_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    mode = st.radio("Agents", ["Live — Claude", "Offline — scripted"], index=0 if has_key else 1,
                    help="Offline mode runs the real tools, ledger, verifier and orchestration with "
                         "rule-based stand-ins for the model. No API key, no cost, not AI.")
    max_cost = st.slider("Budget (USD)", 0.5, 10.0, 3.0, 0.5,
                         help="Hard cap. New LLM calls stop once it is spent; you still get a memo.")
    run = st.button("Run investigation", type="primary", use_container_width=True,
                    disabled=df is None or not question.strip())
    st.caption(f"Lead / critic / reporter: `{LEAD_MODEL}` · specialists: `{WORKER_MODEL}`")


st.title("ADA — an AI data-science team you can audit")
st.markdown(
    "A **lead** agent plans the investigation, **specialists** (data quality, statistics, ML) work "
    "in parallel with real analysis tools, a **critic** challenges every claim, and a **reporter** "
    "writes the memo. Every number must trace back to a computed result — code verifies it."
)

if df is not None and "result" not in st.session_state and not run:
    c = st.columns(4)
    c[0].metric("Rows", f"{len(df):,}")
    c[1].metric("Columns", df.shape[1])
    c[2].metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    c[3].metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
    st.dataframe(df.head(15), use_container_width=True)


# ── Running ─────────────────────────────────────────────────────────────────

def _feed_line(e: dict) -> str:
    icon = AGENT_ICONS.get(e["agent"], "·")
    kind = KIND_ICONS.get(e["kind"], e["kind"])
    return f"`{e['t']:6.1f}s` {icon} **{e['agent']}** {kind} — {e['message'][:160]}"


if run:
    st.session_state.pop("result", None)
    llm.set_client(ScriptedClaude() if mode.startswith("Offline") else None)
    events: queue.Queue = queue.Queue()
    box: dict = {}

    def worker():
        try:
            box["result"] = investigate(df, question.strip(), target=target,
                                        settings=Settings(budget=Budget(max_cost_usd=max_cost)),
                                        on_event=events.put)
        except Exception as exc:  # surfaced in the UI below
            box["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    status = st.status("Team at work…", expanded=True)
    metrics = st.empty()
    feed = status.empty()
    lines, counts = [], {"accepted": 0, "rejected": 0, "evidence": 0, "cost": 0.0}
    while thread.is_alive() or not events.empty():
        while not events.empty():
            e = events.get()
            if e["kind"] == "thought":
                continue
            lines.append(_feed_line(e))
            if e["kind"] in ("accepted", "rejected", "evidence"):
                counts[e["kind"]] += 1
        feed.markdown("\n\n".join(lines[-25:]))
        m = metrics.columns(3)
        m[0].metric("Evidence items", counts["evidence"])
        m[1].metric("Findings accepted", counts["accepted"])
        m[2].metric("Findings rejected", counts["rejected"])
        time.sleep(0.4)

    if "error" in box:
        status.update(label="Investigation failed", state="error")
        st.exception(box["error"])
        st.stop()
    status.update(label="Investigation complete", state="complete", expanded=False)
    st.session_state["result"] = box["result"]
    st.session_state["offline"] = mode.startswith("Offline")


# ── Results ─────────────────────────────────────────────────────────────────

result = st.session_state.get("result")
if result is None:
    st.stop()

ledger, summary = result.ledger, result.tracer.summary()
findings = ledger.findings()
accepted = [f for f in findings if f.status == "accepted"]

cols = st.columns(6)
cols[0].metric("Findings accepted", f"{len(accepted)} / {len(findings)}")
cols[1].metric("Memo grounding", f"{result.lint.get('grounding_rate', 0):.0%}",
               help="Share of numeric sentences whose numbers appear in the evidence they cite")
cols[2].metric("Evidence items", len(ledger.all_evidence()))
cols[3].metric("Hypothesis tests", len(ledger.hypothesis_tests()))
cols[4].metric("Cost" + (" (est.)" if st.session_state.get("offline") else ""),
               f"${summary['total_cost_usd']:.2f}")
cols[5].metric("Wall time", f"{summary['wall_seconds']:.0f}s")
if result.stop_reason:
    st.caption(f"Stopped because: {result.stop_reason}")

tabs = st.tabs(["📝 Memo", "🧾 Findings & review", "📎 Evidence ledger", "🧭 Agent timeline",
                "💰 Trace & cost"])

with tabs[0]:
    left, right = st.columns([3, 2])
    with left:
        st.markdown(result.memo)
        st.download_button("Download memo (.md)", result.memo, "ada_memo.md", "text/markdown")
    with right:
        st.subheader("Citations")
        for f in accepted:
            with st.expander(f"[{f.id}] {f.claim[:70]}…" if len(f.claim) > 70 else f"[{f.id}] {f.claim}"):
                st.markdown(f"**{f.kind}** · by `{f.agent}` · confidence {f.confidence}")
                st.markdown(f"**Critic:** {f.critic_reason}")
                st.markdown("**Evidence:** " + ", ".join(f"`{e}`" for e in f.evidence_ids))
        if result.lint.get("ungrounded"):
            st.warning("Sentences the grounding check could not verify:\n\n"
                       + "\n".join(f"- {s}" for s in result.lint["ungrounded"]))
        shap = os.path.join(result.run_dir, f"shap_{result.target}.png")
        if result.target and os.path.exists(shap):
            st.image(shap, caption="SHAP summary of the selected model")

with tabs[1]:
    status_filter = st.radio("Show", ["All", "Accepted", "Rejected"], horizontal=True)
    for f in findings:
        if status_filter != "All" and f.status != status_filter.lower():
            continue
        icon = {"accepted": "✅", "rejected": "❌"}.get(f.status, "⏳")
        with st.expander(f"{icon} {f.id} · {AGENT_ICONS.get(f.agent, '')} {f.agent} · {f.claim[:110]}"):
            st.markdown(f"**Claim:** {f.claim}")
            st.markdown(f"**Implication:** {f.implication}")
            st.markdown(f"**Review:** {f.critic_reason}")
            if f.checks:
                st.dataframe(pd.DataFrame([{
                    "check": c["check"], "result": "pass" if c["passed"] else
                    ("FAIL" if c["blocking"] else "warning"), "detail": c["detail"]} for c in f.checks]),
                    hide_index=True, use_container_width=True)

with tabs[2]:
    tests = ledger.hypothesis_tests()
    if tests:
        st.subheader("Hypothesis tests (corrected across the whole team)")
        st.dataframe(pd.DataFrame([{
            "id": e.id, "by": e.agent, "comparison": e.result["record"].get("comparison"),
            "test": e.result["record"].get("test_used"), "p": e.result.get("p_value"),
            "q (adjusted)": e.result.get("p_adjusted"),
            e.result["record"].get("effect_name") or "effect": e.result["record"].get("effect_value"),
            "verdict": e.result.get("verdict")} for e in tests]), hide_index=True, use_container_width=True)
    st.subheader("All evidence")
    ev_ids = [e.id for e in ledger.all_evidence()]
    if ev_ids:
        pick = st.selectbox("Evidence item", ev_ids,
                            format_func=lambda i: f"{i} · {ledger.evidence(i).tool} · {ledger.evidence(i).agent}")
        ev = ledger.evidence(pick)
        st.markdown(f"**Tool:** `{ev.tool}` · **agent:** `{ev.agent}` · **args:** `{json.dumps(ev.args)}`")
        if ev.tool == "run_python":
            st.code(ev.result.get("code", ""), language="python")
            st.code(ev.result.get("stdout", ""))
        else:
            st.json(ev.result, expanded=False)

with tabs[3]:
    for note in result.lead_notes:
        action = "stops" if note["done"] or not note["tasks"] else "dispatches"
        st.markdown(f"#### Round {note['round']} — lead {action}")
        st.markdown(f"> {note['reasoning']}")
        for t in note["tasks"]:
            st.markdown(f"- {AGENT_ICONS.get(t['specialist'], '')} **{t['specialist']}**: {t['objective']}")
    st.subheader("Specialist runs")
    st.dataframe(pd.DataFrame([{k: (", ".join(v) if isinstance(v, list) else v) for k, v in t.items()}
                               for t in result.task_log]), hide_index=True, use_container_width=True)
    with st.expander("Full event log"):
        st.dataframe(pd.DataFrame(result.tracer.events)[["t", "agent", "kind", "message"]],
                     hide_index=True, use_container_width=True)

with tabs[4]:
    by_agent = pd.DataFrame(summary["by_agent"]).T.reset_index(names="agent")
    c1, c2 = st.columns([2, 3])
    with c1:
        st.dataframe(by_agent, hide_index=True, use_container_width=True)
        st.caption(f"{summary['llm_calls']} LLM calls · {summary['tool_calls']} tool calls · "
                   f"{summary['input_tokens']:,} in / {summary['output_tokens']:,} out tokens · "
                   f"cache hit rate {summary['cache_hit_rate']:.0%}")
    with c2:
        spans = pd.DataFrame([s for s in result.tracer.to_dict()["spans"]])
        if not spans.empty:
            spans["end"] = spans["started"] + spans["duration_s"]
            chart = alt.Chart(spans).mark_bar(height=10).encode(
                x=alt.X("started:Q", title="seconds since start"), x2="end:Q",
                y=alt.Y("agent:N", title=None), color=alt.Color("kind:N", title="span"),
                tooltip=["agent", "kind", "name", "duration_s", "cost_usd"],
            ).properties(height=260, title="Where the time went (parallel specialists overlap)")
            st.altair_chart(chart, use_container_width=True)
    st.download_button("Download full trace (.json)", json.dumps(result.to_dict(), default=str, indent=2),
                       "ada_investigation.json", "application/json")
