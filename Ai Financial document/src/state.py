"""Shared LangGraph state passed between agents."""
import operator
from typing import Annotated, Any, Optional, TypedDict


def append_or_reset(left: list | None, right: list | None) -> list:
    """List reducer: parallel agents append; sending None at the start of a turn clears it."""
    if right is None:
        return []
    return (left or []) + right


class QAState(TypedDict, total=False):
    # identity
    thread_id: str
    turn_id: str
    # input
    question: str
    # memory (persisted across turns by the checkpointer)
    messages: Annotated[list[dict], operator.add]
    active_companies: list[str]
    active_year: Optional[int]
    # context agent
    standalone_question: str
    companies: list[str]
    fiscal_year: Optional[int]
    assumptions: list[str]
    # router agent
    route: str
    route_reason: str
    # retrieval agent
    retrieved: list[dict]
    retrieval_attempts: int
    retrieval_status: str  # ok | retry | empty | failed
    # specialist agents
    qa_answer: str
    kpi_results: dict[str, Any]
    kpi_table: str
    kpi_narrative: str
    summary: str
    # composition / validation
    draft_answer: str
    citations: list[dict]
    confidence: float
    needs_review: bool
    review_reasons: list[str]
    # human in the loop
    human_decision: dict
    final_answer: str
    # observability
    trace: Annotated[list[dict], append_or_reset]
    errors: Annotated[list[str], append_or_reset]
