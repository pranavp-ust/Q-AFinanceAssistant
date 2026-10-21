"""Summary Agent - executive summary of a report (Overview / Performance / Risks / Outlook)."""
import re

from src import llm
from src.monitoring import agent_node, metrics
from src.prompts import SUMMARY_SYSTEM

from .common import body, format_context, sentences

SECTIONS = {
    "Financial performance": re.compile(r"\b(revenue|net income|gross profit|ebitda|free cash flow)\b", re.I),
    "Key risks": re.compile(r"(•|risk|could|may )", re.I),
    "Outlook": re.compile(r"\b(expects|outlook|guidance|strategy for the coming year)\b", re.I),
}


def extractive_summary(chunks: list[dict]) -> str:
    """Fallback: pick representative sentences per section, keeping citations."""
    out, used = [], set()
    for title, pat in SECTIONS.items():
        picked = []
        for i, c in enumerate(chunks[:10], start=1):
            for s in sentences(body(c)):
                if pat.search(s) and s not in used and len(picked) < 3:
                    picked.append(f"- {s.lstrip('• ').strip()} [{i}]")
                    used.add(s)
        if picked:
            out.append(f"**{title}**\n" + "\n".join(picked))
    return "\n\n".join(out) or "Not enough content was retrieved to summarise."


@agent_node("summary_agent")
def summary_agent(state):
    chunks = state.get("retrieved", [])
    q = state.get("standalone_question") or state["question"]
    if not chunks:
        return {"summary": "No relevant document passages were found to summarise.",
                "_trace_detail": "no context"}
    try:
        s = llm.chat(SUMMARY_SYSTEM, f"Request: {q}\n\nContext passages:\n{format_context(chunks, limit=10)}",
                     agent="summary_agent")
        src = "llm"
    except llm.LLMUnavailable:
        metrics.inc("fallback:summary")
        s, src = extractive_summary(chunks), "extractive-fallback"
    return {"summary": s, "_trace_detail": f"summary via {src}, {len(chunks)} chunks"}
