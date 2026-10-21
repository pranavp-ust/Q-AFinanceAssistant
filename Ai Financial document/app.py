"""Streamlit UI - chat, human approval panel, agent trace and monitoring dashboard.

Run:  streamlit run app.py
"""
import tempfile
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import settings
from src.graph import ask, build_graph, review
from src.llm import llm_enabled
from src.monitoring import metrics
from src.tools import known_companies
from src.vectorstore import get_collection, ingest_pdf

st.set_page_config(page_title="AI Financial Document Q&A", page_icon="📊", layout="wide")


@st.cache_resource
def get_graph():
    # one compiled graph + in-memory checkpointer shared across reruns => conversation memory
    return build_graph()


graph = get_graph()
ss = st.session_state
ss.setdefault("thread_id", uuid.uuid4().hex[:8])
ss.setdefault("chat", [])        # [{"role", "content", "meta"}]
ss.setdefault("pending", None)   # result awaiting human review

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("📊 Financial Doc Q&A")
    st.caption(f"LLM: **{settings.llm_provider if llm_enabled() else 'OFF (rule-based fallbacks)'}** · "
               f"Embeddings: **{settings.embedding_provider}**")
    try:
        st.caption(f"Indexed chunks: **{get_collection().count()}**")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Vector store unavailable: {exc}")
    st.caption(f"Conversation id: `{ss.thread_id}`")
    if st.button("🆕 New conversation", use_container_width=True):
        ss.thread_id, ss.chat, ss.pending = uuid.uuid4().hex[:8], [], None
        st.rerun()

    with st.expander("📤 Add a report (PDF)"):
        up = st.file_uploader("PDF", type="pdf")
        co = st.text_input("Company name", placeholder="e.g. Nimbus Retail Ltd")
        fy = st.number_input("Fiscal year", 2000, 2100, 2024)
        if up and co and st.button("Index document"):
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / up.name
                p.write_bytes(up.getvalue())
                n = ingest_pdf(p, company=co, fiscal_year=int(fy))
            st.success(f"Indexed {n} chunks from {up.name}")

    st.subheader("Monitoring")
    snap = metrics.snapshot()
    c = snap["counters"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Requests", c.get("requests", 0))
    m2.metric("Reviews", c.get("sent_for_review", 0))
    m3.metric("Errors", c.get("node_errors", 0))
    m1.metric("LLM calls", c.get("llm_calls", 0))
    m2.metric("Retries", c.get("llm_retries", 0) + c.get("retrieval_retries", 0))
    m3.metric("Fallbacks", sum(v for k, v in c.items() if k.startswith("fallback:")))
    if snap["routes"]:
        st.caption("Route distribution")
        st.bar_chart(pd.Series(snap["routes"], name="questions"))
    if snap["avg_latency_ms"]:
        st.caption("Average latency per agent (ms)")
        st.dataframe(pd.Series(snap["avg_latency_ms"], name="ms").sort_values(ascending=False),
                     use_container_width=True)

# ------------------------------------------------------------------ main
st.title("AI Financial Document Q&A Assistant")
st.caption("Multi-agent RAG over annual reports & balance sheets · " + ", ".join(known_companies()))


def md(text: str) -> str:
    """Escape $ so Streamlit does not render '$281.4 million ... $2.73' as LaTeX."""
    return (text or "").replace("$", r"\$")


def render_meta(meta: dict):
    if not meta:
        return
    cols = st.columns(4)
    cols[0].caption(f"Route: **{meta.get('route')}**")
    cols[1].caption(f"Confidence: **{(meta.get('confidence') or 0):.2f}**")
    cols[2].caption(f"Companies: {', '.join(meta.get('companies') or []) or '-'}")
    cols[3].caption(f"FY: {meta.get('fiscal_year') or '-'}")
    if meta.get("human_decision"):
        st.caption(f"👤 Human review: **{meta['human_decision'].get('action')}**")
    with st.expander("Sources, agent trace & KPIs"):
        if meta.get("citations"):
            st.markdown("**Sources**")
            st.dataframe(pd.DataFrame(meta["citations"]), hide_index=True, use_container_width=True)
        if meta.get("trace"):
            st.markdown("**Agent trace**")
            st.dataframe(pd.DataFrame(meta["trace"]), hide_index=True, use_container_width=True)
        if meta.get("kpi_results"):
            st.markdown("**KPI extraction (document vs ledger)**")
            st.json(meta["kpi_results"], expanded=False)
        if meta.get("errors"):
            st.warning("; ".join(meta["errors"]))


for msg in ss.chat:
    with st.chat_message(msg["role"]):
        st.markdown(md(msg["content"]))
        if msg["role"] == "assistant":
            render_meta(msg.get("meta"))

# ------------------------------------------------------------------ human-in-the-loop panel
if ss.pending:
    res = ss.pending
    rv = res["review"]
    with st.container(border=True):
        st.subheader("👤 Human review required")
        for r in rv["reasons"]:
            st.warning(r)
        st.caption(f"Validation confidence: {(rv.get('confidence') or 0):.2f}")
        edited = st.text_area("Draft answer (edit if needed)", rv["draft_answer"], height=300)
        comment = st.text_input("Reviewer comment (optional)")
        a, e, r = st.columns(3)
        action = None
        if a.button("✅ Approve", use_container_width=True):
            action = "approve"
        if e.button("✏️ Approve with edits", use_container_width=True):
            action = "edit"
        if r.button("❌ Reject", use_container_width=True):
            action = "reject"
        if action:
            out = review(graph, ss.thread_id, action, edited if action == "edit" else "", comment)
            ss.pending = None
            ss.chat.append({"role": "assistant", "content": out["answer"], "meta": out})
            st.rerun()

question = st.chat_input("Ask about revenue, margins, risks, outlook…", disabled=bool(ss.pending))
if question:
    ss.chat.append({"role": "user", "content": question})
    with st.spinner("Agents working…"):
        res = ask(graph, ss.thread_id, question)
    if res["status"] == "awaiting_review":
        ss.pending = res
    else:
        ss.chat.append({"role": "assistant", "content": res["answer"], "meta": res})
    st.rerun()

if not ss.chat:
    st.markdown("**Try:**")
    for ex in ["What was Nimbus Retail's revenue in FY2024?", "And what about the previous year?",
               "What are the principal risks for Orion Software?",
               "Compare net margin and debt-to-equity across all companies in FY2024",
               "Give me a full analysis of Vertex Energy FY2024", "Should I buy Orion shares?"]:
        st.markdown(f"- {ex}")
