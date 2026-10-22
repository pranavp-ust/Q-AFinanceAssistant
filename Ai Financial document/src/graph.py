"""LangGraph workflow: wires the agents into sequential, router and parallel flows.

  START -> context_agent -> router_agent --(out_of_scope)--> out_of_scope -> finalize
                                  |
                                  v
                          retrieval_agent <--(retry: no relevant chunks)--+
                                  |---------------------------------------+
                  +---------------+-------------------+-----------------------+
                  | qa            | kpi               | summary               | full_analysis
                  v               v                   v                       v (PARALLEL fan-out)
             answer_agent     kpi_agent         summary_agent        kpi_agent || summary_agent
                  +---------------+-------------------+-----------------------+
                                  v  (fan-in)
                               compose -> validation_agent --(needs_review)--> human_review
                                                  |                                   |
                                                  +-----------> finalize <------------+ -> END
"""
import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from src.agents import (advice_agent, answer_agent, compose_node, context_agent, finalize, human_review,
                        kpi_agent, out_of_scope, retrieval_agent, router_agent, summary_agent, validation_agent)
from src.monitoring import log_event, metrics
from src.state import QAState


def after_router(state) -> str:
    route = state.get("route")
    if route == "out_of_scope":
        return "out_of_scope"
    if route == "advice":
        return "advice_agent"
    return "retrieval_agent"


def after_retrieval(state) -> list[str] | str:
    if state.get("retrieval_status") == "retry":
        return "retrieval_agent"
    route = state.get("route")
    if route == "full_analysis":
        return ["kpi_agent", "summary_agent"]       # parallel execution
    return {"qa": "answer_agent", "kpi": "kpi_agent", "summary": "summary_agent"}.get(route, "answer_agent")


def after_validation(state) -> str:
    return "human_review" if state.get("needs_review") else "finalize"


def build_graph(checkpointer=None):
    g = StateGraph(QAState)
    g.add_node("context_agent", context_agent)
    g.add_node("router_agent", router_agent)
    g.add_node("retrieval_agent", retrieval_agent)
    g.add_node("answer_agent", answer_agent)
    g.add_node("kpi_agent", kpi_agent)
    g.add_node("summary_agent", summary_agent)
    g.add_node("compose", compose_node)
    g.add_node("validation_agent", validation_agent)
    g.add_node("human_review", human_review)
    g.add_node("out_of_scope", out_of_scope)
    g.add_node("finalize", finalize)

    g.add_edge(START, "context_agent")                 # sequential
    g.add_edge("context_agent", "router_agent")
    g.add_node("advice_agent", advice_agent)
    g.add_conditional_edges("router_agent", after_router, ["out_of_scope", "advice_agent", "retrieval_agent"])
    g.add_edge("advice_agent", "compose")              # goes to review: advice rule in validation
    g.add_conditional_edges("retrieval_agent", after_retrieval,
                            ["retrieval_agent", "answer_agent", "kpi_agent", "summary_agent"])
    for n in ("answer_agent", "kpi_agent", "summary_agent"):
        g.add_edge(n, "compose")                       # fan-in
    g.add_edge("compose", "validation_agent")
    g.add_conditional_edges("validation_agent", after_validation, ["human_review", "finalize"])
    g.add_edge("human_review", "finalize")
    g.add_edge("out_of_scope", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


# ------------------------------------------------------------------ service API used by UI / CLI
TURN_RESET = {
    "standalone_question": "", "companies": [], "fiscal_year": None, "assumptions": [],
    "route": "", "route_reason": "", "retrieved": [], "retrieval_attempts": 0, "retrieval_status": "",
    "qa_answer": "", "kpi_results": {}, "kpi_table": "", "kpi_narrative": "", "summary": "",
    "draft_answer": "", "citations": [], "confidence": 0.0, "needs_review": False, "review_reasons": [],
    "human_decision": {}, "final_answer": "", "trace": None, "errors": None,
}


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}


def _result(graph, thread_id: str) -> dict:
    snap = graph.get_state(_config(thread_id))
    v = snap.values
    pending = [i for t in snap.tasks for i in (t.interrupts or [])]
    out = {k: v.get(k) for k in ("route", "route_reason", "standalone_question", "companies", "fiscal_year",
                                 "kpi_results", "kpi_table", "citations", "confidence", "review_reasons",
                                 "trace", "errors", "human_decision")}
    if pending:
        out.update(status="awaiting_review", review=pending[0].value, answer=v.get("draft_answer"))
    else:
        out.update(status="done", answer=v.get("final_answer"))
    return out


def ask(graph, thread_id: str, question: str) -> dict:
    turn_id = uuid.uuid4().hex[:8]
    metrics.inc("requests")
    log_event({"thread": thread_id, "turn": turn_id, "event": "question", "question": question})
    graph.invoke({**TURN_RESET, "question": question, "thread_id": thread_id, "turn_id": turn_id},
                 _config(thread_id))
    return _result(graph, thread_id)


def review(graph, thread_id: str, action: str, edited_answer: str = "", comment: str = "") -> dict:
    """Resume a paused workflow with the analyst's decision: approve | edit | reject."""
    graph.invoke(Command(resume={"action": action, "edited_answer": edited_answer, "comment": comment}),
                 _config(thread_id))
    return _result(graph, thread_id)


def history(graph, thread_id: str) -> list[dict]:
    return graph.get_state(_config(thread_id)).values.get("messages", [])
