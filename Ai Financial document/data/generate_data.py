"""Generate synthetic (fictional) financial data for the capstone.

Outputs
-------
data/structured/financials.csv          - the "ledger": exact figures per company & fiscal year
data/documents/<slug>_FY<year>_annual_report.pdf   - narrative annual reports (RAG source)
data/documents/<slug>_FY<year>_balance_sheet.pdf   - standalone balance sheets
data/documents/financial_kpi_glossary.pdf          - KPI definitions

All companies and numbers are fictional. The PDFs and the CSV are generated from the
same numbers, so the KPI agent can cross-check text it extracted against the ledger.
A small, deliberate error is injected into one report (Vertex FY2024 revenue in the
chairman's letter) so the demo can show discrepancy detection -> human review.

Usage:  python data/generate_data.py
"""
import csv
import random
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).absolute().parent
DOCS = ROOT / "documents"
STRUCT = ROOT / "structured"
YEARS = [2022, 2023, 2024]

COMPANIES = [
    {
        "name": "Nimbus Retail Ltd",
        "slug": "nimbus_retail",
        "sector": "Retail",
        "base_revenue": 2400.0,
        "growth": (0.06, 0.11),
        "gross_margin": (0.31, 0.34),
        "opex_ratio": (0.18, 0.20),
        "segments": ["Grocery", "Apparel", "Online Marketplace"],
        "risks": [
            "Rising logistics and fuel costs could compress gross margins.",
            "Intense price competition from discount and online-only retailers.",
            "Consumer demand is sensitive to inflation and household disposable income.",
            "Dependence on third-party suppliers for private-label products.",
        ],
        "strategy": "expanding the online marketplace, opening 40 small-format stores and investing in supply-chain automation",
    },
    {
        "name": "Orion Software Inc",
        "slug": "orion_software",
        "sector": "Enterprise Software",
        "base_revenue": 860.0,
        "growth": (0.14, 0.22),
        "gross_margin": (0.71, 0.76),
        "opex_ratio": (0.44, 0.50),
        "segments": ["Cloud Subscriptions", "Professional Services", "On-premise Licences"],
        "risks": [
            "A cyber-security incident could damage customer trust and trigger penalties.",
            "Customer churn may rise if competitors bundle similar features for free.",
            "Talent shortages may increase engineering salary costs.",
            "Foreign-exchange movements affect reported revenue from overseas customers.",
        ],
        "strategy": "migrating licence customers to cloud subscriptions and launching an AI analytics add-on",
    },
    {
        "name": "Vertex Energy plc",
        "slug": "vertex_energy",
        "sector": "Renewable Energy",
        "base_revenue": 1500.0,
        "growth": (0.03, 0.09),
        "gross_margin": (0.38, 0.42),
        "opex_ratio": (0.12, 0.14),
        "segments": ["Solar Generation", "Wind Generation", "Energy Storage"],
        "risks": [
            "Changes in government renewable-energy subsidies could reduce project returns.",
            "Higher interest rates increase the cost of financing new capacity.",
            "Weather variability affects generation volumes and revenue.",
            "Supply-chain constraints on solar panels and turbine components.",
        ],
        "strategy": "adding 600 MW of solar capacity, scaling battery storage and refinancing long-term debt",
    },
]


def r1(x: float) -> float:
    return round(x, 1)


def build_financials() -> list[dict]:
    rows = []
    for c in COMPANIES:
        rng = random.Random(c["slug"])
        revenue = c["base_revenue"]
        equity = revenue * rng.uniform(0.45, 0.6)
        lt_debt = revenue * rng.uniform(0.25, 0.5)
        for i, year in enumerate(YEARS):
            if i:
                revenue *= 1 + rng.uniform(*c["growth"])
            gross_margin = rng.uniform(*c["gross_margin"])
            cost_of_revenue = revenue * (1 - gross_margin)
            gross_profit = revenue - cost_of_revenue
            opex = revenue * rng.uniform(*c["opex_ratio"])
            operating_income = gross_profit - opex
            depreciation = revenue * rng.uniform(0.03, 0.06)
            ebitda = operating_income + depreciation
            interest = lt_debt * rng.uniform(0.05, 0.07)
            pbt = operating_income - interest
            tax = pbt * 0.25
            net_income = pbt - tax
            equity += net_income * 0.6  # 40% paid out as dividends
            lt_debt *= rng.uniform(0.9, 1.08)
            current_liabilities = revenue * rng.uniform(0.14, 0.2)
            total_liabilities = current_liabilities + lt_debt + revenue * rng.uniform(0.03, 0.06)
            total_assets = total_liabilities + equity
            current_assets = current_liabilities * rng.uniform(1.2, 2.1)
            cash = current_assets * rng.uniform(0.3, 0.5)
            ocf = net_income + depreciation + revenue * rng.uniform(-0.01, 0.03)
            capex = revenue * rng.uniform(0.04, 0.09)
            shares = 100.0 + i * 1.5  # millions
            row = {
                "company": c["name"],
                "fiscal_year": year,
                "sector": c["sector"],
                "revenue": revenue,
                "cost_of_revenue": cost_of_revenue,
                "gross_profit": gross_profit,
                "operating_expenses": opex,
                "operating_income": operating_income,
                "depreciation": depreciation,
                "ebitda": ebitda,
                "interest_expense": interest,
                "tax_expense": tax,
                "net_income": net_income,
                "total_assets": total_assets,
                "current_assets": current_assets,
                "cash_and_equivalents": cash,
                "current_liabilities": current_liabilities,
                "total_liabilities": total_liabilities,
                "long_term_debt": lt_debt,
                "shareholders_equity": equity,
                "operating_cash_flow": ocf,
                "capital_expenditure": capex,
                "free_cash_flow": ocf - capex,
                "shares_outstanding_mn": shares,
                "eps": net_income / shares,
                "dividend_per_share": net_income * 0.4 / shares,
            }
            rows.append({k: (r1(v) if isinstance(v, float) and k not in ("eps", "dividend_per_share") else v)
                         for k, v in row.items()})
            rows[-1]["eps"] = round(row["eps"], 2)
            rows[-1]["dividend_per_share"] = round(row["dividend_per_share"], 2)
            # keep identities exact after rounding
            rr = rows[-1]
            rr["gross_profit"] = r1(rr["revenue"] - rr["cost_of_revenue"])
            rr["total_assets"] = r1(rr["total_liabilities"] + rr["shareholders_equity"])
            rr["free_cash_flow"] = r1(rr["operating_cash_flow"] - rr["capital_expenditure"])
    return rows


def money(x: float) -> str:
    return f"${x:,.1f} million"


def pct(new: float, old: float) -> str:
    return f"{(new - old) / abs(old) * 100:.1f}%"


def _table(data, col_widths=None):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b57")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    return t


def statement_rows(row, prev, fields):
    head = ["Line item ($ million)", f"FY{row['fiscal_year']}"] + ([f"FY{prev['fiscal_year']}"] if prev else [])
    data = [head]
    for label, key in fields:
        line = [label, f"{row[key]:,.1f}"]
        if prev:
            line.append(f"{prev[key]:,.1f}")
        data.append(line)
    return data


INCOME_FIELDS = [
    ("Revenue", "revenue"), ("Cost of revenue", "cost_of_revenue"), ("Gross profit", "gross_profit"),
    ("Operating expenses", "operating_expenses"), ("Operating income (EBIT)", "operating_income"),
    ("Depreciation & amortisation", "depreciation"), ("EBITDA", "ebitda"),
    ("Interest expense", "interest_expense"), ("Tax expense", "tax_expense"), ("Net income", "net_income"),
]
BALANCE_FIELDS = [
    ("Cash and cash equivalents", "cash_and_equivalents"), ("Total current assets", "current_assets"),
    ("Total assets", "total_assets"), ("Total current liabilities", "current_liabilities"),
    ("Long-term debt", "long_term_debt"), ("Total liabilities", "total_liabilities"),
    ("Shareholders' equity", "shareholders_equity"),
]
CASH_FIELDS = [
    ("Net cash from operating activities", "operating_cash_flow"),
    ("Capital expenditure", "capital_expenditure"), ("Free cash flow", "free_cash_flow"),
]


def write_annual_report(c, row, prev, inject_error=False):
    styles = getSampleStyleSheet()
    h1, h2, body = styles["Title"], styles["Heading2"], styles["BodyText"]
    y = row["fiscal_year"]
    path = DOCS / f"{c['slug']}_FY{y}_annual_report.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4, title=f"{c['name']} Annual Report FY{y}")
    s = []
    P = lambda t: s.append(Paragraph(t, body))  # noqa: E731

    s += [Paragraph(f"{c['name']}<br/>Annual Report FY{y}", h1),
          Paragraph(f"Sector: {c['sector']}. All figures in US dollars. Fiscal year ends 31 March {y}. "
                    "This is a synthetic document created for an educational capstone project.", body),
          Spacer(1, 12)]

    letter_revenue = row["revenue"] * (1.08 if inject_error else 1.0)
    s.append(Paragraph("Chairman's Letter to Shareholders", h2))
    growth_txt = (f", up {pct(row['revenue'], prev['revenue'])} from {money(prev['revenue'])} in FY{y-1}"
                  if prev else "")
    P(f"Dear shareholders, FY{y} was a year of disciplined execution for {c['name']}. "
      f"Revenue for FY{y} was {money(letter_revenue)}{growth_txt if not inject_error else ''}. "
      f"Net income for FY{y} was {money(row['net_income'])} and earnings per share were ${row['eps']:.2f}. "
      f"The Board has recommended a dividend of ${row['dividend_per_share']:.2f} per share.")
    P(f"Our strategy for the coming year focuses on {c['strategy']}. We remain committed to "
      "transparent reporting, prudent capital allocation and sustainable long-term value creation.")

    s.append(Paragraph("Management Discussion and Analysis", h2))
    P(f"Gross profit for FY{y} was {money(row['gross_profit'])}, giving a gross margin of "
      f"{row['gross_profit'] / row['revenue'] * 100:.1f}%. Operating expenses were {money(row['operating_expenses'])} "
      f"and operating income was {money(row['operating_income'])}. EBITDA for FY{y} was {money(row['ebitda'])}.")
    if prev:
        direction = "improved" if row["net_income"] > prev["net_income"] else "declined"
        P(f"Net income {direction} from {money(prev['net_income'])} in FY{y-1} to {money(row['net_income'])} in FY{y}, "
          f"a change of {pct(row['net_income'], prev['net_income'])}. Interest expense was {money(row['interest_expense'])}, "
          f"reflecting the level of long-term debt of {money(row['long_term_debt'])}.")
    rng = random.Random(f"{c['slug']}{y}")
    shares = [rng.uniform(0.2, 0.5) for _ in c["segments"]]
    total = sum(shares)
    seg_txt = "; ".join(f"{seg} contributed {money(row['revenue'] * sh / total)}"
                        for seg, sh in zip(c["segments"], shares))
    P(f"Segment performance: {seg_txt}.")

    s.append(Paragraph("Liquidity and Capital Resources", h2))
    P(f"Cash and cash equivalents at year end were {money(row['cash_and_equivalents'])}. "
      f"Net cash from operating activities was {money(row['operating_cash_flow'])} and capital expenditure was "
      f"{money(row['capital_expenditure'])}, resulting in free cash flow of {money(row['free_cash_flow'])}. "
      f"Total current assets were {money(row['current_assets'])} against total current liabilities of "
      f"{money(row['current_liabilities'])}.")
    P(f"Total assets were {money(row['total_assets'])}, total liabilities were {money(row['total_liabilities'])} "
      f"and shareholders' equity was {money(row['shareholders_equity'])}.")

    s.append(Paragraph("Principal Risks and Uncertainties", h2))
    for risk in c["risks"]:
        P(f"&bull; {risk}")

    s.append(Paragraph("Outlook", h2))
    lo, hi = c["growth"]
    P(f"For FY{y+1}, management expects revenue growth of {lo*100:.0f}% to {hi*100:.0f}%, supported by "
      f"{c['strategy']}. Guidance assumes stable macroeconomic conditions.")

    s.append(PageBreak())
    s.append(Paragraph("Consolidated Income Statement", h2))
    s.append(_table(statement_rows(row, prev, INCOME_FIELDS)))
    s.append(Paragraph("Consolidated Balance Sheet", h2))
    s.append(_table(statement_rows(row, prev, BALANCE_FIELDS)))
    s.append(Paragraph("Consolidated Cash Flow Statement", h2))
    s.append(_table(statement_rows(row, prev, CASH_FIELDS)))
    doc.build(s)
    return path


def write_balance_sheet(c, row):
    styles = getSampleStyleSheet()
    y = row["fiscal_year"]
    path = DOCS / f"{c['slug']}_FY{y}_balance_sheet.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4)
    s = [Paragraph(f"{c['name']} - Balance Sheet as at 31 March {y}", styles["Title"]),
         Paragraph("Synthetic document for educational use.", styles["BodyText"]), Spacer(1, 10),
         _table(statement_rows(row, None, BALANCE_FIELDS)), Spacer(1, 10),
         Paragraph(f"Total assets of {money(row['total_assets'])} equal total liabilities of "
                   f"{money(row['total_liabilities'])} plus shareholders' equity of "
                   f"{money(row['shareholders_equity'])}.", styles["BodyText"])]
    doc.build(s)


GLOSSARY = [
    ("Revenue", "Total income from selling goods and services before any costs are deducted."),
    ("Gross margin", "Gross profit divided by revenue. Shows how much of each dollar of sales remains after direct costs."),
    ("EBITDA", "Earnings before interest, tax, depreciation and amortisation. Approximates operating cash generation."),
    ("Operating margin", "Operating income (EBIT) divided by revenue."),
    ("Net profit margin", "Net income divided by revenue."),
    ("Current ratio", "Current assets divided by current liabilities. Above 1.0 means short-term obligations are covered."),
    ("Debt-to-equity ratio", "Total liabilities divided by shareholders' equity. Measures financial leverage."),
    ("Return on equity (ROE)", "Net income divided by shareholders' equity."),
    ("Return on assets (ROA)", "Net income divided by total assets."),
    ("Free cash flow", "Operating cash flow minus capital expenditure. Cash available to repay debt or pay dividends."),
    ("Earnings per share (EPS)", "Net income divided by the number of shares outstanding."),
    ("Interest coverage", "Operating income divided by interest expense. Below 1.5 is usually considered risky."),
]


def write_glossary():
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(DOCS / "financial_kpi_glossary.pdf"), pagesize=A4)
    s = [Paragraph("Financial KPI Glossary", styles["Title"])]
    for term, desc in GLOSSARY:
        s += [Paragraph(term, styles["Heading3"]), Paragraph(desc, styles["BodyText"])]
    doc.build(s)


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    STRUCT.mkdir(parents=True, exist_ok=True)
    rows = build_financials()
    with open(STRUCT / "financials.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    by_key = {(r["company"], r["fiscal_year"]): r for r in rows}
    for c in COMPANIES:
        for y in YEARS:
            row, prev = by_key[(c["name"], y)], by_key.get((c["name"], y - 1))
            inject = c["slug"] == "vertex_energy" and y == 2024
            write_annual_report(c, row, prev, inject_error=inject)
            write_balance_sheet(c, row)
    write_glossary()
    print(f"Wrote {len(rows)} ledger rows to {STRUCT / 'financials.csv'}")
    print(f"Wrote {len(list(DOCS.glob('*.pdf')))} PDFs to {DOCS}")


if __name__ == "__main__":
    main()
