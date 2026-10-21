"""Offline evaluation: KPI accuracy, routing accuracy and retrieval hit-rate.

Ground truth comes from data/structured/financials.csv, so the numbers are exact.
Usage:  python evaluate.py        -> prints a table and writes logs/eval_report.md
"""
import time

import pandas as pd

from src.config import settings
from src.graph import ask, build_graph, review

graph = build_graph()
ledger = pd.read_csv(settings.financials_csv)
cases = []
for _, r in ledger[ledger["fiscal_year"] >= 2023].iterrows():
    c, y = r["company"], int(r["fiscal_year"])
    cases += [
        {"q": f"What was {c}'s revenue in FY{y}?", "route": "kpi", "expect": f"{r['revenue']:,.1f}"},
        {"q": f"What was the net income of {c} in FY{y}?", "route": "kpi", "expect": f"{r['net_income']:,.1f}"},
        {"q": f"What was {c}'s free cash flow in FY{y}?", "route": "kpi", "expect": f"{r['free_cash_flow']:,.1f}"},
        {"q": f"What are the principal risks for {c} in FY{y}?", "route": "qa",
         "source": f"FY{y}_annual_report", "company": c},
        {"q": f"What is the outlook for {c} after FY{y}?", "route": "qa",
         "source": f"FY{y}_annual_report", "company": c},
    ]
cases += [
    {"q": "What is free cash flow?", "route": "qa", "source": "glossary"},
    {"q": "Summarize Orion Software's FY2024 annual report", "route": "summary"},
    {"q": "Give me a full analysis of Nimbus Retail FY2024", "route": "full_analysis"},
    {"q": "Who won the football world cup?", "route": "out_of_scope"},
]

rows = []
for i, case in enumerate(cases):
    t0 = time.perf_counter()
    res = ask(graph, f"eval-{i}", case["q"])
    reviewed = res["status"] == "awaiting_review"
    if reviewed:
        res = review(graph, f"eval-{i}", "approve")
    ms = (time.perf_counter() - t0) * 1000
    row = {"question": case["q"], "route_ok": res["route"] == case["route"], "reviewed": reviewed,
           "latency_ms": round(ms)}
    if "expect" in case:
        row["answer_ok"] = case["expect"] in (res["answer"] or "")
    if "source" in case:
        cites = res.get("citations") or []
        row["retrieval_hit"] = any(case["source"] in (c["source"] or "") and
                                   (c["company"] == case.get("company", c["company"])) for c in cites[:3])
    rows.append(row)

df = pd.DataFrame(rows)
summary = {
    "questions": len(df),
    "routing_accuracy": f"{df['route_ok'].mean():.0%}",
    "kpi_answer_accuracy": f"{df['answer_ok'].dropna().astype(bool).mean():.0%}",
    "retrieval_hit_rate@3": f"{df['retrieval_hit'].dropna().astype(bool).mean():.0%}",
    "sent_for_human_review": int(df["reviewed"].sum()),
    "avg_latency_ms": int(df["latency_ms"].mean()),
}
print(df.to_string(index=False, max_colwidth=60))
print("\nSUMMARY")
for k, v in summary.items():
    print(f"  {k:<24} {v}")
report = settings.logs_dir / "eval_report.md"
report.write_text("# Evaluation report\n\n" + "\n".join(f"- **{k}**: {v}" for k, v in summary.items())
                  + "\n\n```\n" + df.to_string(index=False) + "\n```\n", encoding="utf-8")
print(f"\nReport written to {report}")
