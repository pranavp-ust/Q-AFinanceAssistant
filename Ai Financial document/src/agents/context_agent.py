"""Context (Memory) Agent - resolves follow-up questions using conversation memory.

"What about the previous year?" after a question on Nimbus FY2024 becomes
"What was Nimbus Retail Ltd's revenue in FY2023?".
Entity memory (active companies / year) lives in the checkpointed graph state.
"""
import re

from src import llm
from src.config import settings
from src.monitoring import agent_node
from src.prompts import CONTEXT_REWRITE_SYSTEM
from src.tools import known_companies, known_years

RELATIVE_PREV = re.compile(r"\b(last|previous|prior|preceding) (fiscal )?year\b", re.I)
# phrases that signal the question depends on the previous turn
FOLLOW_UP = re.compile(
    r"^(and|also|what about|how about|same|then)\b|\b(it|its|it's|they|their|them|this company|that company|"
    r"the company|same|previous|prior|last year|that year|this year|those|these)\b", re.I)
ALL_COMPANIES = re.compile(r"\b(all|every|each|across) (the )?(companies|firms)\b|\bcompare\b", re.I)


def extract_companies(text: str) -> list[str]:
    q = text.lower()
    found = []
    for c in known_companies():
        first = c.split()[0].lower()
        if c.lower() in q or re.search(rf"\b{re.escape(first)}\b", q):
            found.append(c)
    return found


def extract_year(text: str) -> int | None:
    m = re.findall(r"\b(?:fy\s?'?)?(20\d{2})\b", text, re.I)
    return int(m[-1]) if m else None


def _history_text(messages: list[dict]) -> str:
    window = messages[-2 * settings.memory_window_turns:]
    return "\n".join(f"{m['role'].upper()}: {m['content'][:400]}" for m in window)


@agent_node("context_agent")
def context_agent(state):
    q = state["question"].strip()
    history = state.get("messages", [])
    active_cos = state.get("active_companies") or []
    active_year = state.get("active_year")
    assumptions = []

    companies = extract_companies(q)
    year = extract_year(q)
    if RELATIVE_PREV.search(q) and active_year:
        year = active_year - 1
    if ALL_COMPANIES.search(q) and not companies:
        companies = known_companies()
    standalone = q
    used_memory = False

    if history:
        follow_up = bool(FOLLOW_UP.search(q))
        if not companies and active_cos and follow_up:
            companies, used_memory = list(active_cos), True
        if year is None and active_year and used_memory:
            year = active_year
        try:
            out = llm.chat_json(
                CONTEXT_REWRITE_SYSTEM.format(companies=", ".join(known_companies()), years=known_years(),
                                              company=", ".join(active_cos) or None, year=active_year),
                f"Conversation so far:\n{_history_text(history)}\n\nLatest question: {q}",
                agent="context_agent")
            standalone = out.get("standalone_question") or q
            if out.get("company") and out["company"] in known_companies() and not extract_companies(q):
                companies = [out["company"]]
            if isinstance(out.get("fiscal_year"), int) and extract_year(q) is None:
                year = out["fiscal_year"]
        except llm.LLMUnavailable:
            if used_memory:  # rule-based rewrite: make the inherited context explicit
                last_q = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
                standalone = (f"{q} (company: {', '.join(companies)}; fiscal year: FY{year}; "
                              f"previous question: {last_q})")

    if companies and year is None and known_years():
        year = max(known_years())
        assumptions.append(f"No fiscal year given - assumed latest available FY{year}.")

    return {
        "standalone_question": standalone,
        "companies": companies,
        "fiscal_year": year,
        "assumptions": assumptions,
        "active_companies": companies or active_cos,
        "active_year": year or active_year,
        "_trace_detail": f"companies={companies} year={year} memory_used={used_memory} q='{standalone}'",
    }
