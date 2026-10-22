"""Compose + Answer Validation (guardrail) Agent.

compose_node   - merges outputs of the specialist agents into one draft answer.
validation     - groundedness check (LLM judge, heuristic fallback) + business rules
                 that decide whether a human must approve the answer.
"""
from src import llm
from src.config import settings
from src.monitoring import agent_node, metrics
from src.prompts import VALIDATION_SYSTEM

from .common import ADVICE_PATTERN, citations_for, format_context, numbers_in


@agent_node("compose")
def compose_node(state):
    route = state.get("route")
    parts = []
    if state.get("assumptions"):
        parts.append("_" + " ".join(state["assumptions"]) + "_")
    if route in ("qa", "advice"):
        parts.append(state.get("qa_answer", ""))
    if route == "summary":
        parts.append(state.get("summary", ""))
    if route == "full_analysis":
        parts.append("### Executive summary\n" + state.get("summary", "(summary unavailable)"))
    if route in ("kpi", "full_analysis"):
        if state.get("kpi_narrative"):
            parts.append(("### Key metrics\n" if route == "full_analysis" else "") + state["kpi_narrative"])
        if state.get("kpi_table"):
            parts.append(state["kpi_table"])
            kpi_results = state.get("kpi_results") or {}
            if all(r["verified_against_ledger"] for r in kpi_results.values()):
                parts.append("_Figures verified against the structured financial ledger; ratios computed by the "
                             "calculator tool._")
            else:
                parts.append("_Some figures are not in the structured ledger, so they come from the document "
                             "only and were not verified. Ratios computed by the calculator tool._")
    if state.get("errors"):
        parts.append("_Note: some components were unavailable, so this answer may be incomplete._")
    draft = "\n\n".join(p for p in parts if p)
    return {"draft_answer": draft, "citations": citations_for(state.get("retrieved", [])),
            "_trace_detail": f"composed {route} answer ({len(draft)} chars)"}


def heuristic_confidence(answer: str, context: str, has_chunks: bool) -> float:
    nums = numbers_in(answer)
    if not nums:
        return 0.75 if has_chunks else 0.3
    ctx = {n.rstrip("0").rstrip(".") for n in numbers_in(context)}
    supported = sum(1 for n in nums if n.rstrip("0").rstrip(".") in ctx)
    return round(supported / len(nums), 2)


@agent_node("validation_agent")
def validation_agent(state):
    answer = state.get("draft_answer", "")
    chunks = state.get("retrieved", [])
    context = format_context(chunks) + "\n\nKPI table:\n" + (state.get("kpi_table") or "")
    reasons = []

    try:
        verdict = llm.chat_json(VALIDATION_SYSTEM, f"CONTEXT:\n{context}\n\nANSWER:\n{answer}",
                                agent="validation_agent")
        confidence = float(verdict.get("confidence", 0.5))
        if verdict.get("unsupported_claims"):
            reasons.append("Unsupported claims: " + "; ".join(map(str, verdict["unsupported_claims"][:3])))
        src = "llm-judge"
    except (llm.LLMUnavailable, ValueError, TypeError):
        metrics.inc("fallback:validation")
        confidence, src = heuristic_confidence(answer, context, bool(chunks)), "heuristic"

    for company, r in (state.get("kpi_results") or {}).items():
        for d in r["discrepancies"]:
            reasons.append(f"{company}: document states {d['metric']} = {d['document_value']:,} but ledger has "
                           f"{d['ledger_value']:,} ({d['diff_pct']}% difference)")
        if not r["verified_against_ledger"]:
            reasons.append(f"{company}: figures could not be verified against the ledger")
    if ADVICE_PATTERN.search(state["question"]):
        reasons.append("Question requests investment advice - compliance review required")
    if state.get("route") == "full_analysis":
        reasons.append("Full analysis reports require analyst sign-off before release")
    if confidence < settings.review_confidence_threshold:
        reasons.append(f"Low validation confidence ({confidence:.2f} < {settings.review_confidence_threshold})")
    if state.get("errors"):
        reasons.append("Workflow degraded: " + "; ".join(state["errors"])[:200])

    needs_review = bool(reasons)
    if needs_review:
        metrics.inc("sent_for_review")
    return {"confidence": confidence, "needs_review": needs_review, "review_reasons": reasons,
            "_trace_detail": f"confidence={confidence:.2f} via {src}; review={needs_review}"}
