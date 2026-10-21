"""Retrieval Agent - RAG retrieval with relevance grading and a retry strategy,
plus grounded answer generation for the `qa` route.

Retry ladder (graph self-loop, max 3 attempts):
  1. company + fiscal-year filter
  2. company filter only (year may be mis-resolved)
  3. no filters (e.g. glossary / general questions)
"""
from src import llm
from src.config import settings
from src.monitoring import agent_node, metrics
from src.prompts import QA_SYSTEM
from src.tools import detect_metrics, search_documents

from .common import body, format_context, keywords, sentences

MAX_ATTEMPTS = 3
MIN_KEYWORD_COVERAGE = 0.5
SUMMARY_FACETS = ["revenue net income financial performance", "principal risks and uncertainties",
                  "outlook guidance strategy", "cash flow liquidity balance sheet"]


def _queries(state) -> list[str]:
    q = state.get("standalone_question") or state["question"]
    route = state.get("route")
    if route in ("summary", "full_analysis"):  # multi-query retrieval for broad coverage
        return [q] + SUMMARY_FACETS
    if route == "kpi":
        m, r = detect_metrics(q)
        return [q, " ".join(x.replace("_", " ") for x in m + r) or "income statement balance sheet"]
    return [q]


@agent_node("retrieval_agent")
def retrieval_agent(state):
    attempt = state.get("retrieval_attempts", 0) + 1
    companies = state.get("companies") or None
    year = state.get("fiscal_year")
    if attempt == 2:
        year = None
    elif attempt >= 3:
        companies, year = None, None

    try:
        seen, hits = set(), []
        k = settings.retrieval_top_k + (2 if state.get("route") in ("summary", "full_analysis") else 0)
        plan = [(query, companies, year) for query in _queries(state)]
        if not state.get("companies"):  # also search general docs (glossary has company "")
            plan.append((plan[0][0], [""], None))
        for query, cos, yr in plan:
            for h in search_documents(query, cos, yr, k=k):
                key = (h["source"], h["page"], h["text"][:60])
                if key not in seen:
                    seen.add(key)
                    hits.append(h)
    except Exception as exc:  # vector store missing / embedding failure -> degrade, don't crash
        metrics.inc("retrieval_failures")
        return {"retrieved": [], "retrieval_attempts": attempt, "retrieval_status": "failed",
                "errors": [f"retrieval_agent: {exc}"], "_trace_detail": f"retrieval failed: {exc}"}

    # relevance grading: close in vector space OR covers enough of the question's keywords
    company_words = {w for c in (state.get("companies") or []) for w in keywords(c)}
    qk = keywords(state.get("standalone_question") or state["question"]) - company_words
    for h in hits:
        h["keyword_coverage"] = round(len(qk & keywords(body(h))) / len(qk), 2) if qk else 0.0
    # no company in the question (e.g. "What is EBITDA?") -> prefer general docs such as the glossary
    prefer_general = not state.get("companies")
    relevant = sorted([h for h in hits if h["distance"] <= settings.max_relevant_distance
                       or h["keyword_coverage"] >= MIN_KEYWORD_COVERAGE],
                      key=lambda h: (-(prefer_general and not h.get("company")), -h.get("rrf", 0)))
    if not relevant and attempt < MAX_ATTEMPTS:
        metrics.inc("retrieval_retries")
        return {"retrieval_attempts": attempt, "retrieval_status": "retry", "retrieved": [],
                "_trace_detail": f"attempt {attempt}: 0 relevant of {len(hits)} -> broadening filters"}
    status = "ok" if relevant else "empty"
    best = relevant[0]["distance"] if relevant else None
    return {"retrieved": relevant[:10], "retrieval_attempts": attempt, "retrieval_status": status,
            "_trace_detail": f"attempt {attempt}: {len(relevant)} relevant chunks, best distance={best}, "
                             f"filters companies={companies} year={year}"}


def _extractive_answer(question: str, chunks: list[dict], companies: list[str]) -> str:
    """Fallback when the LLM is unavailable: best-overlapping sentences, with citations."""
    qk = keywords(question.split(" (company:")[0]) - {w for c in companies for w in keywords(c)}
    scored = []
    for i, c in enumerate(chunks[:6], start=1):
        for s in sentences(body(c)):
            overlap = len(qk & keywords(s))
            if overlap:
                scored.append((overlap, -i, s, i))
    if not scored:
        return "The documents do not contain this information."
    top = sorted(scored, reverse=True)[:3]
    return " ".join(f"{s} [{i}]" for _, _, s, i in top)


@agent_node("answer_agent")
def answer_agent(state):
    chunks = state.get("retrieved", [])
    q = state.get("standalone_question") or state["question"]
    if not chunks:
        return {"qa_answer": "I could not find relevant passages in the indexed financial documents.",
                "_trace_detail": "no context"}
    try:
        ans = llm.chat(QA_SYSTEM, f"Context passages:\n{format_context(chunks)}\n\nQuestion: {q}",
                       agent="answer_agent")
        src = "llm"
    except llm.LLMUnavailable:
        metrics.inc("fallback:answer")
        ans, src = _extractive_answer(q, chunks, state.get("companies") or []), "extractive-fallback"
    return {"qa_answer": ans, "_trace_detail": f"answer via {src}"}
