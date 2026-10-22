"""Router Agent - decides which specialist path handles the question.

Routes: qa | kpi | summary | full_analysis | out_of_scope
LLM classification first; keyword rules as fallback (and as a sanity check).
"""
import re

from src import llm
from src.monitoring import agent_node, metrics
from src.prompts import ROUTER_SYSTEM
from src.tools import detect_metrics

from .common import ADVICE_PATTERN

ROUTES = {"qa", "kpi", "summary", "full_analysis", "out_of_scope"}
FINANCE_WORDS = re.compile(
    r"\b(revenue|profit|income|margin|ratio|debt|equity|asset|liabilit|cash|dividend|eps|earning|"
    r"risk|outlook|strategy|segment|report|balance sheet|financ|kpi|growth|ebitda|capex|guidance|"
    r"shareholder|invest|stock|share|fiscal|fy|quarter|annual)\w*", re.I)


def rule_route(question: str, has_company: bool) -> tuple[str, str]:
    q = question.lower()
    metrics_found, ratios = detect_metrics(q)
    if re.search(r"full analysis|complete analysis|detailed analysis|health check|financial health|"
                 r"analy[sz]e .*(company|performance)|summary (with|and) (key )?(metrics|kpis)|report card", q):
        return "full_analysis", "asks for narrative + metrics"
    if re.search(r"summar|overview|highlights|in brief|key takeaways", q):
        return "summary", "asks for a summary"
    if (metrics_found or ratios) and (has_company or "compare" in q) and not re.search(r"\b(why|explain|what is|what are|define|meaning)\b", q):
        return "kpi", f"numeric metrics requested: {metrics_found + ratios}"
    if has_company or FINANCE_WORDS.search(q):
        return "qa", "factual / explanatory question"
    return "out_of_scope", "no financial intent detected"


@agent_node("router_agent")
def router_agent(state):
    q = state.get("standalone_question") or state["question"]
    has_company = bool(state.get("companies"))
    if ADVICE_PATTERN.search(state["question"]):  # deterministic: never retrieve passages for buy/sell questions
        return {"route": "advice", "route_reason": "investment advice request",
                "_trace_detail": "route=advice via rules: investment advice request"}
    source = "llm"
    try:
        out = llm.chat_json(ROUTER_SYSTEM, f"Question: {q}\nCompanies detected: {state.get('companies')}",
                            agent="router_agent")
        route, reason = str(out.get("route", "")).strip(), out.get("reason", "")
        if route not in ROUTES:
            raise llm.LLMUnavailable(f"invalid route {route!r}")
    except llm.LLMUnavailable:
        route, reason = rule_route(q, has_company)
        source = "rules"
        metrics.inc("fallback:router")
    metrics.route(route)
    return {"route": route, "route_reason": reason,
            "_trace_detail": f"route={route} via {source}: {reason}"}
