"""End-to-end workflow tests (offline mode: every agent uses its fallback)."""
from src.graph import ask, history, review
from src.tools import lookup_financials


def agents(res):
    return [t["agent"] for t in res["trace"]]


def test_kpi_route_uses_ledger_values(graph):
    res = ask(graph, "t-kpi", "What was Nimbus Retail's revenue in FY2024?")
    assert res["status"] == "done" and res["route"] == "kpi"
    expected = lookup_financials("Nimbus Retail Ltd", 2024)["revenue"]
    assert f"{expected:,.1f}" in res["answer"]
    assert agents(res) == ["context_agent", "router_agent", "retrieval_agent", "kpi_agent",
                           "compose", "validation_agent", "finalize"]


def test_memory_resolves_follow_up(graph):
    ask(graph, "t-mem", "What was Nimbus Retail's revenue in FY2024?")
    res = ask(graph, "t-mem", "And what about the previous year?")
    assert res["companies"] == ["Nimbus Retail Ltd"] and res["fiscal_year"] == 2023
    assert f"{lookup_financials('Nimbus Retail Ltd', 2023)['revenue']:,.1f}" in res["answer"]
    assert len(history(graph, "t-mem")) == 4


def test_memory_not_applied_to_unrelated_question(graph):
    ask(graph, "t-mem2", "What was Nimbus Retail's revenue in FY2024?")
    res = ask(graph, "t-mem2", "What's the weather in Paris?")
    assert res["route"] == "out_of_scope" and res["companies"] == []


def test_full_analysis_runs_kpi_and_summary_in_parallel_and_needs_review(graph):
    res = ask(graph, "t-full", "Give me a full analysis of Vertex Energy FY2024")
    assert res["status"] == "awaiting_review"
    assert {"kpi_agent", "summary_agent"} <= set(agents(res))
    # injected error in the Vertex FY2024 chairman's letter must be caught
    assert any("revenue" in r and "ledger" in r for r in res["review"]["reasons"])
    done = review(graph, "t-full", "edit", edited_answer="Corrected analysis.", comment="fixed revenue")
    assert done["status"] == "done"
    assert done["answer"].startswith("Corrected analysis.")
    assert done["human_decision"]["action"] == "edit"


def test_reject_withholds_answer(graph):
    res = ask(graph, "t-rej", "Should I buy Orion Software shares?")
    assert res["status"] == "awaiting_review"
    assert any("investment advice" in r for r in res["review"]["reasons"])
    done = review(graph, "t-rej", "reject", comment="no advice")
    assert "withheld" in done["answer"]


def test_comparison_across_companies(graph):
    from src.tools import known_companies
    res = ask(graph, "t-cmp", "Compare net margin across all companies in FY2024")
    assert res["route"] == "kpi"
    # includes any uploaded reports, not only the ledger companies
    assert len(res["kpi_results"]) == len(known_companies())
    assert {"Nimbus Retail Ltd", "Orion Software Inc", "Vertex Energy plc"} <= set(res["kpi_results"])


def test_retrieval_failure_degrades_gracefully(graph, monkeypatch):
    import sys
    ra = sys.modules["src.agents.retrieval_agent"]  # the package re-exports a function of the same name

    def boom(*a, **k):
        raise RuntimeError("vector store offline")
    monkeypatch.setattr(ra, "search_documents", boom)
    res = ask(graph, "t-err", "What was Orion Software's net income in FY2024?")
    # KPI agent still answers from the structured ledger; the degraded run goes to review
    assert res["status"] == "awaiting_review"
    assert any("degraded" in r for r in res["review"]["reasons"])
    assert f"{lookup_financials('Orion Software Inc', 2024)['net_income']:,.1f}" in res["answer"]


def test_advice_question_gets_fixed_reply_without_retrieval(graph):
    res = ask(graph, "t-adv", "Should I buy Lakeview Foods shares?")
    assert res["route"] == "advice"
    assert "can't say whether to buy" in res["answer"]
    assert "retrieval_agent" not in agents(res)
    assert any("investment advice" in r for r in res["review"]["reasons"])
