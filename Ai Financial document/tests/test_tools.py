import pytest

from src.tools import compute_ratio, detect_metrics, lookup_financials, safe_calculate, search_documents


def test_lookup_financials_returns_ledger_row():
    row = lookup_financials("Nimbus Retail Ltd", 2024)
    assert row["revenue"] > 0
    assert row["total_assets"] == pytest.approx(row["total_liabilities"] + row["shareholders_equity"], abs=0.11)


def test_lookup_unknown_company_is_empty():
    assert lookup_financials("Unknown Corp", 2024) == {}


def test_ratios():
    cur = {"revenue": 200.0, "net_income": 20.0, "gross_profit": 80.0, "current_assets": 150.0,
           "current_liabilities": 100.0, "total_liabilities": 300.0, "shareholders_equity": 200.0}
    assert compute_ratio("net_margin", cur) == 10.0
    assert compute_ratio("gross_margin", cur) == 40.0
    assert compute_ratio("current_ratio", cur) == 1.5
    assert compute_ratio("debt_to_equity", cur) == 1.5
    assert compute_ratio("revenue_growth", cur, {"revenue": 160.0}) == 25.0
    assert compute_ratio("revenue_growth", cur, {}) is None


def test_safe_calculate_blocks_code():
    assert safe_calculate("(1,250 - 1,000) / 1,000 * 100") == 25.0
    with pytest.raises(ValueError):
        safe_calculate("__import__('os').system('dir')")


def test_detect_metrics():
    m, r = detect_metrics("What was the free cash flow and net margin?")
    assert "free_cash_flow" in m and "cash_and_equivalents" not in m
    assert r == ["net_margin"]


def test_search_filters_by_company_and_year():
    hits = search_documents("revenue", ["Orion Software Inc"], 2023)
    assert hits and all(h["company"] == "Orion Software Inc" and h["fiscal_year"] == 2023 for h in hits)
