"""KPI Extraction Agent.

1. Works out which metrics / ratios the question needs.
2. Extracts the values *stated in the documents* (LLM -> JSON; regex fallback).
3. Fetches the same figures from the structured ledger tool and computes ratios
   with the calculator tool (incl. YoY growth from the previous year).
4. Cross-checks document values against the ledger; any difference above
   KPI_DISCREPANCY_PCT is flagged and later sent for human review.
"""
import re

from src import llm
from src.config import settings
from src.monitoring import agent_node, metrics
from src.prompts import KPI_EXTRACT_SYSTEM, KPI_NARRATIVE_SYSTEM
from src.tools import (CORE_KPIS, CORE_RATIOS, PERCENT_RATIOS, compute_ratio, detect_metrics,
                       known_years, lookup_financials)
from src.vectorstore import search

from .common import body, format_context, money

LABELS = {
    "revenue": r"Revenue", "net_income": r"Net income", "gross_profit": r"Gross profit",
    "ebitda": r"EBITDA", "operating_income": r"operating income", "operating_expenses": r"Operating expenses",
    "cash_and_equivalents": r"Cash and cash equivalents at year end", "total_assets": r"Total assets",
    "total_liabilities": r"total liabilities", "shareholders_equity": r"shareholders' equity",
    "free_cash_flow": r"free cash flow of", "operating_cash_flow": r"Net cash from operating activities",
    "capital_expenditure": r"capital expenditure", "current_assets": r"Total current assets",
    "current_liabilities": r"total current liabilities", "interest_expense": r"Interest expense",
    "long_term_debt": r"long-term debt of", "eps": r"earnings per share", "dividend_per_share": r"dividend of",
}
RATIO_NEEDS_PREV = {"revenue_growth", "net_income_growth"}
PRETTY = {"eps": "EPS ($)", "dividend_per_share": "Dividend/share ($)", "roe": "ROE (%)", "roa": "ROA (%)",
          "debt_to_equity": "Debt-to-equity (x)", "current_ratio": "Current ratio (x)",
          "interest_coverage": "Interest coverage (x)"}


def pretty(key: str) -> str:
    if key in PRETTY:
        return PRETTY[key]
    label = key.replace("_", " ").capitalize()
    return f"{label} (%)" if key in PERCENT_RATIOS else f"{label} ($M)"


def regex_extract(chunks: list[dict], wanted: list[str], year: int) -> dict:
    """Fallback extraction from the narrative sentences of the reports."""
    text = " ".join(body(c) for c in chunks)
    out = {}
    for key in wanted:
        label = LABELS.get(key)
        if not label:
            continue
        pat = (rf"{label}(?: for FY{year})?(?: at year end)?(?: (?:was|were))?\s*(?:of\s*)?\$([\d,]+(?:\.\d+)?)"
               if key not in ("eps", "dividend_per_share") else rf"{label}[^$]{{0,20}}\$(\d+(?:\.\d+)?)")
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            out[key] = float(m.group(1).replace(",", ""))
    return out


def _targeted_chunks(company: str, year: int, wanted: list[str]) -> list[dict]:
    """Metric-focused search: the broad retrieval can miss the sentence that states a figure."""
    if not wanted or not year:
        return []
    found, seen = [], set()
    for key in wanted:  # one query per metric: a combined query dilutes the match
        try:
            hits = search(f"{key.replace('_', ' ')} for FY{year} was $ million", [company], year, k=3)
        except Exception:  # store missing / embedding failure -> rely on the broad retrieval only
            return found
        for h in hits:
            if h["text"] not in seen:
                seen.add(h["text"])
                found.append(h)
    return found


def _doc_values(question, company, year, wanted, chunks) -> tuple[dict, str]:
    own = [c for c in chunks if c.get("company") == company and c.get("fiscal_year") == year]
    if not own or not wanted:
        return {}, "none"
    try:
        out = llm.chat_json(KPI_EXTRACT_SYSTEM.format(metrics=wanted),
                            f"Company: {company}\nFiscal year: FY{year}\n\nContext:\n{format_context(own)}",
                            agent="kpi_agent")
        vals = {k: float(v) for k, v in out.items() if k in wanted and isinstance(v, (int, float))}
        return vals, "llm"
    except (llm.LLMUnavailable, ValueError, TypeError, AttributeError):
        metrics.inc("fallback:kpi_extract")
        return regex_extract(own, wanted, year), "regex"


def _fmt(key, v):
    if v is None:
        return "n/a"
    if key in PERCENT_RATIOS:
        return f"{v:.1f}%"
    if key in ("eps", "dividend_per_share", "current_ratio", "debt_to_equity", "interest_coverage"):
        return f"{v:.2f}"
    return f"{v:,.1f}"


def build_table(results: dict) -> str:
    companies = list(results)
    keys = []
    for r in results.values():
        for k in list(r["metrics"]) + list(r["ratios"]):
            if k not in keys:
                keys.append(k)
    head = "| KPI | " + " | ".join(f"{c} FY{results[c]['fiscal_year']}" for c in companies) + " |"
    sep = "|---|" + "---:|" * len(companies)
    rows = []
    for k in keys:
        cells = []
        for c in companies:
            r = results[c]
            v = r["metrics"].get(k, r["ratios"].get(k))
            flag = " ⚠" if any(d["metric"] == k for d in r["discrepancies"]) else ""
            cells.append(_fmt(k, v) + flag)
        rows.append(f"| {pretty(k)} | " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *rows])


@agent_node("kpi_agent")
def kpi_agent(state):
    q = state.get("standalone_question") or state["question"]
    companies = state.get("companies") or []
    year = state.get("fiscal_year") or (max(known_years()) if known_years() else None)
    if not companies:
        return {"kpi_results": {}, "kpi_table": "",
                "kpi_narrative": "Please mention which company you are asking about.",
                "_trace_detail": "no company resolved"}

    raw, ratios = detect_metrics(q)
    if state.get("route") == "full_analysis" or not (raw or ratios):
        raw, ratios = list(CORE_KPIS), list(CORE_RATIOS)
    chunks = state.get("retrieved", [])
    results, sources = {}, set()

    for company in companies:
        ledger = lookup_financials(company, year)               # tool: structured DB
        prev = lookup_financials(company, year - 1) if year else {}
        seen = {c["text"] for c in chunks}
        extra = [c for c in _targeted_chunks(company, year, raw) if c["text"] not in seen]
        doc_vals, src = _doc_values(q, company, year, raw, chunks + extra)  # LLM / regex extraction
        sources.add(src)
        values = {k: ledger.get(k, doc_vals.get(k)) for k in raw}
        ratio_vals = {}
        basis = ledger or doc_vals
        for r in ratios:                                         # tool: ratio calculator
            if r in RATIO_NEEDS_PREV and not prev:
                ratio_vals[r] = None
                continue
            ratio_vals[r] = compute_ratio(r, basis, prev)
        discrepancies = []
        for k, dv in doc_vals.items():                           # cross-check doc vs ledger
            lv = ledger.get(k)
            if lv:
                diff = abs(dv - lv) / abs(lv) * 100
                if diff > settings.kpi_discrepancy_pct:
                    discrepancies.append({"metric": k, "document_value": dv, "ledger_value": lv,
                                          "diff_pct": round(diff, 2)})
        results[company] = {"fiscal_year": year, "metrics": values, "ratios": ratio_vals,
                            "document_values": doc_vals, "discrepancies": discrepancies,
                            "verified_against_ledger": bool(ledger)}

    table = build_table(results)
    try:
        narrative = llm.chat(KPI_NARRATIVE_SYSTEM, f"Question: {q}\n\nKPI table:\n{table}", agent="kpi_agent")
    except llm.LLMUnavailable:
        metrics.inc("fallback:kpi_narrative")
        parts = []
        for c, r in results.items():
            shown = {**r["metrics"], **r["ratios"]}
            parts.append(f"{c} (FY{r['fiscal_year']}): " + ", ".join(
                f"{pretty(k).split(' (')[0]} {money(v) if k in r['metrics'] and k not in ('eps', 'dividend_per_share') else _fmt(k, v)}"
                for k, v in shown.items()) + ".")
        narrative = " ".join(parts)
    n_disc = sum(len(r["discrepancies"]) for r in results.values())
    return {"kpi_results": results, "kpi_table": table, "kpi_narrative": narrative,
            "_trace_detail": f"{len(companies)} companies, metrics={raw}, ratios={ratios}, "
                             f"extraction={sorted(sources)}, discrepancies={n_disc}"}
