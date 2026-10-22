"""Tools the agents can call.

* search_documents   - semantic search over the report vector store (RAG)
* lookup_financials  - structured "ledger" database of reported figures (CSV via pandas)
* compute_ratio      - financial ratio calculator (margins, leverage, liquidity, returns, growth)
* safe_calculate     - arithmetic expression evaluator (AST-based, no eval())

Each tool is a plain function, registered in TOOL_REGISTRY with a description, and every
call is logged so the trace shows which tools an agent used.
"""
import ast
import operator
import re
from functools import lru_cache

import pandas as pd

from src import vectorstore
from src.config import settings
from src.monitoring import log_event, metrics

# ---------------------------------------------------------------- metric vocabulary
METRIC_ALIASES = {
    "revenue": ["revenue", "sales", "turnover", "top line", "top-line"],
    "gross_profit": ["gross profit"],
    "operating_income": ["operating income", "ebit", "operating profit"],
    "ebitda": ["ebitda"],
    "net_income": ["net income", "net profit", "profit after tax", "pat", "earnings", "bottom line"],
    "operating_expenses": ["operating expenses", "opex"],
    "interest_expense": ["interest expense"],
    "total_assets": ["total assets"],
    "total_liabilities": ["total liabilities"],
    "current_assets": ["current assets"],
    "current_liabilities": ["current liabilities"],
    "long_term_debt": ["long-term debt", "long term debt", "debt"],
    "shareholders_equity": ["shareholders' equity", "shareholders equity", "equity", "net worth"],
    "cash_and_equivalents": ["cash and cash equivalents", "cash balance", "cash"],
    "operating_cash_flow": ["operating cash flow", "cash from operations", "cash from operating activities"],
    "capital_expenditure": ["capital expenditure", "capex"],
    "free_cash_flow": ["free cash flow", "fcf"],
    "eps": ["earnings per share", "eps"],
    "dividend_per_share": ["dividend per share", "dividend"],
}
RATIO_ALIASES = {
    "gross_margin": ["gross margin"],
    "operating_margin": ["operating margin", "ebit margin"],
    "net_margin": ["net margin", "net profit margin", "profit margin"],
    "current_ratio": ["current ratio", "liquidity"],
    "debt_to_equity": ["debt to equity", "debt-to-equity", "leverage", "d/e"],
    "roe": ["return on equity", "roe"],
    "roa": ["return on assets", "roa"],
    "interest_coverage": ["interest coverage"],
    "revenue_growth": ["revenue growth", "sales growth", "grow", "growth"],
    "net_income_growth": ["profit growth", "net income growth", "earnings growth"],
}
CORE_KPIS = ["revenue", "gross_profit", "ebitda", "net_income", "total_assets",
             "total_liabilities", "shareholders_equity", "free_cash_flow", "eps"]
CORE_RATIOS = ["gross_margin", "net_margin", "current_ratio", "debt_to_equity", "roe", "revenue_growth"]
PERCENT_RATIOS = {"gross_margin", "operating_margin", "net_margin", "roe", "roa",
                  "revenue_growth", "net_income_growth"}


def _match_terms(question: str, aliases: dict) -> list[str]:
    q = question.lower()
    found = []
    # longer aliases first so "free cash flow" wins over "cash"
    pairs = sorted(((a, k) for k, al in aliases.items() for a in al), key=lambda p: -len(p[0]))
    for alias, key in pairs:
        pattern = r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])"
        if re.search(pattern, q) and key not in found:
            found.append(key)
            q = re.sub(pattern, " ", q)
    return found


def detect_metrics(question: str) -> tuple[list[str], list[str]]:
    """Return (raw metrics, ratios) mentioned in a question."""
    ratios = _match_terms(question, RATIO_ALIASES)
    q = question.lower()
    for r in ratios:  # avoid "net margin" also matching raw metric "net income"-ish words
        for a in RATIO_ALIASES[r]:
            q = q.replace(a, " ")
    return _match_terms(q, METRIC_ALIASES), ratios


# ---------------------------------------------------------------- structured ledger
@lru_cache(maxsize=1)
def _ledger() -> pd.DataFrame:
    if not settings.financials_csv.exists():
        raise FileNotFoundError("financials.csv missing - run `python data/generate_data.py`")
    return pd.read_csv(settings.financials_csv)


def known_companies() -> list[str]:
    """Ledger companies plus any company with an indexed report (so uploaded reports are recognised)."""
    try:
        ledger = _ledger()["company"].unique().tolist()
    except FileNotFoundError:
        ledger = list(vectorstore.SLUG_TO_COMPANY.values())
    return sorted(set(ledger) | set(vectorstore.indexed_companies()))


def known_years() -> list[int]:
    try:
        return sorted(int(y) for y in _ledger()["fiscal_year"].unique())
    except FileNotFoundError:
        return []


def _log_tool(name: str, args: dict, ok: bool):
    metrics.inc(f"tool:{name}")
    log_event({"tool": name, "args": args, "ok": ok})


def lookup_financials(company: str, fiscal_year: int, metrics_wanted: list[str] | None = None) -> dict:
    """Reported figures for one company-year from the structured ledger ($ million unless per-share)."""
    df = _ledger()
    row = df[(df["company"] == company) & (df["fiscal_year"] == int(fiscal_year))]
    ok = not row.empty
    _log_tool("lookup_financials", {"company": company, "fiscal_year": fiscal_year}, ok)
    if not ok:
        return {}
    rec = row.iloc[0].to_dict()
    if metrics_wanted:
        rec = {k: rec[k] for k in metrics_wanted if k in rec}
    return rec


def compute_ratio(name: str, current: dict, previous: dict | None = None) -> float | None:
    """Financial ratio calculator. Percent ratios are returned as percentages."""
    _log_tool("compute_ratio", {"name": name}, True)

    def div(a, b):
        return None if b in (0, None) or a is None else a / b

    c, p = current, previous or {}
    formulas = {
        "gross_margin": lambda: div(c.get("gross_profit"), c.get("revenue")),
        "operating_margin": lambda: div(c.get("operating_income"), c.get("revenue")),
        "net_margin": lambda: div(c.get("net_income"), c.get("revenue")),
        "current_ratio": lambda: div(c.get("current_assets"), c.get("current_liabilities")),
        "debt_to_equity": lambda: div(c.get("total_liabilities"), c.get("shareholders_equity")),
        "roe": lambda: div(c.get("net_income"), c.get("shareholders_equity")),
        "roa": lambda: div(c.get("net_income"), c.get("total_assets")),
        "interest_coverage": lambda: div(c.get("operating_income"), c.get("interest_expense")),
        "revenue_growth": lambda: (div(c.get("revenue") - p["revenue"], p["revenue"])
                                   if p.get("revenue") and c.get("revenue") is not None else None),
        "net_income_growth": lambda: (div(c.get("net_income") - p["net_income"], abs(p["net_income"]))
                                      if p.get("net_income") and c.get("net_income") is not None else None),
    }
    if name not in formulas:
        raise ValueError(f"Unknown ratio {name}")
    v = formulas[name]()
    if v is None:
        return None
    return round(v * 100, 2) if name in PERCENT_RATIOS else round(v, 2)


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def safe_calculate(expression: str) -> float:
    """Evaluate +-*/** arithmetic safely, e.g. '(1234.5-1100)/1100*100'."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"Unsupported expression: {expression}")
    result = ev(ast.parse(expression.replace(",", ""), mode="eval"))
    _log_tool("safe_calculate", {"expression": expression}, True)
    return result


def search_documents(query: str, companies: list[str] | None = None, fiscal_year: int | None = None,
                     k: int | None = None) -> list[dict]:
    try:
        hits = vectorstore.search(query, companies, fiscal_year, k)
        _log_tool("search_documents", {"query": query, "companies": companies, "fiscal_year": fiscal_year}, True)
        return hits
    except Exception:
        _log_tool("search_documents", {"query": query}, False)
        raise


TOOL_REGISTRY = {
    "search_documents": (search_documents, "Semantic search over annual reports, balance sheets and glossary."),
    "lookup_financials": (lookup_financials, "Exact reported figures for a company and fiscal year."),
    "compute_ratio": (compute_ratio, "Compute margins, leverage, liquidity, returns and growth."),
    "safe_calculate": (safe_calculate, "Evaluate an arithmetic expression."),
}
