"""Human-in-the-loop approval, out-of-scope handling and finalisation (memory write)."""
from langgraph.types import interrupt

from src.monitoring import agent_node, metrics

from .common import ADVICE_PATTERN

DISCLAIMER = ("\n\n> _This assistant provides information from financial documents only and does not "
              "constitute investment advice._")


@agent_node("human_review")
def human_review(state):
    """Pauses the graph (LangGraph interrupt) until an analyst approves, edits or rejects."""
    decision = interrupt({
        "draft_answer": state.get("draft_answer", ""),
        "reasons": state.get("review_reasons", []),
        "confidence": state.get("confidence"),
        "question": state["question"],
    })
    action = (decision or {}).get("action", "approve")
    comment = (decision or {}).get("comment", "")
    metrics.inc(f"review:{action}")
    if action == "edit":
        final = decision.get("edited_answer") or state.get("draft_answer", "")
    elif action == "reject":
        final = ("This answer was withheld by the reviewing analyst."
                 + (f" Reviewer comment: {comment}" if comment else ""))
    else:
        final = state.get("draft_answer", "")
    if comment and action != "reject":
        final += f"\n\n_Reviewer note: {comment}_"
    return {"final_answer": final, "human_decision": {"action": action, "comment": comment},
            "_trace_detail": f"human decision={action}"}


@agent_node("out_of_scope")
def out_of_scope(state):
    msg = ("I can only answer questions about the indexed financial documents - for example revenue, "
           "margins, risks or outlook of Nimbus Retail, Orion Software or Vertex Energy. "
           "Could you rephrase your question?")
    return {"draft_answer": msg, "needs_review": False, "_trace_detail": "polite refusal"}


@agent_node("advice_agent")
def advice_agent(state):
    """Fixed reply for buy/sell questions. No retrieval, so no unrelated passages are quoted."""
    msg = ("I can't say whether to buy, sell or hold any shares. I can share the figures reported in the "
           "indexed financial documents, such as revenue, net income, EPS, debt or risks, if you want to "
           "compare them yourself.")
    return {"qa_answer": msg, "_trace_detail": "fixed investment-advice reply"}


@agent_node("finalize")
def finalize(state):
    final = state.get("final_answer") or state.get("draft_answer", "")
    if ADVICE_PATTERN.search(state["question"]) and DISCLAIMER not in final:
        final += DISCLAIMER
    metrics.inc("requests_completed")
    return {
        "final_answer": final,
        # memory: append this turn to the conversation history kept by the checkpointer
        "messages": [{"role": "user", "content": state["question"]},
                     {"role": "assistant", "content": final}],
        "_trace_detail": "answer delivered and saved to memory",
    }
