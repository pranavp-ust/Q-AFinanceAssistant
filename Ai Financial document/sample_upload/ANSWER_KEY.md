# Answer key: Lakeview_Foods_FY2024_annual_report.pdf

Fictional company. It is **not** in `financials.csv`, so the app cannot verify its figures against the ledger.

## How to test

1. Run `streamlit run app.py` from the `Ai Financial document` folder.
2. In the sidebar, open **Add a report (PDF)**.
3. Upload `Lakeview_Foods_FY2024_annual_report.pdf`.
4. Company name: `Lakeview Foods Ltd` (must match exactly, since questions are matched on it).
5. Fiscal year: `2024`.
6. Click **Index document**, then ask the questions below.

## Expected answers (LLM off)

| Question | Expected answer | Expected review |
|---|---|---|
| What was Lakeview Foods' revenue in FY2024? | $500.0 million | Yes, "could not be verified against the ledger" |
| What was Lakeview Foods' net income in FY2024? | $40.0 million | Yes, same reason |
| What was the EBITDA of Lakeview Foods in FY2024? | $100.0 million | Yes |
| What were the principal risks for Lakeview Foods? | Wheat and packaging costs, supermarket price pressure, food-safety regulation | No |
| What is the FY2025 revenue growth guidance for Lakeview Foods? | 8% to 12% | No |
| What were the cash and free cash flow for Lakeview Foods in FY2024? | Cash $60.0 million, free cash flow $60.0 million | Yes |
| What is Lakeview Foods' gross margin? | 40.0% | Yes |
| What was Lakeview Foods' revenue in FY2023? | $450.0 million | Not in the 2024 report's text in full, but the income statement table lists it |
| Should I buy Lakeview Foods shares? | Compliance review, not a recommendation | Yes (investment-advice rule) |

The review flag is expected for KPI questions, since the figures can't be checked against the ledger. A pure text question, such as the risks or the outlook, should not need review.

## Other reference figures

- Revenue growth FY2024 vs FY2023: (500 - 450) / 450 = 11.1%
- Net income growth: (40 - 34) / 34 = 17.6%
- Net profit margin: 40 / 500 = 8.0%
- Operating income: $80.0 million
- Interest expense: $10.0 million
- Total assets: $600.0 million
- Total liabilities: $350.0 million
- Shareholders' equity: $250.0 million
- Long-term debt: $200.0 million
- Current assets / current liabilities: 150 / 100 = 1.5x
- Dividend per share: $0.80
- EPS: $2.00

## Things to check

- **Accuracy:** do the answers match the figures above?
- **Citations:** each answer should cite a chunk, usually from the annual report.
- **Monitoring panel:** the fallback counters should rise, since the LLM is off.
- **Scope:** ask about Nimbus Retail or Orion Software afterwards. Their answers should not contain Lakeview figures.
- **Known weakness:** the regex fallback looks for phrases such as "Revenue for FY2024 was $X". Ask a question worded differently (e.g. "How much did the company sell?") to see whether the routing and extraction still work.
