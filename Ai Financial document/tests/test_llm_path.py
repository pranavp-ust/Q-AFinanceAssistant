"""Exercise the LLM code path (prompts, JSON parsing, retries) with a fake chat model."""
import json
from types import SimpleNamespace

import pytest

import src.llm as llm
from src.graph import ask


class FakeModel:
    def __init__(self, fail_first: int = 0, bad_json_once: bool = False):
        self.fail_first, self.bad_json_once, self.calls = fail_first, bad_json_once, 0

    def invoke(self, messages):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise TimeoutError("simulated API timeout")
        system = messages[0].content
        if "Router Agent" in system:
            if self.bad_json_once:
                self.bad_json_once = False
                return SimpleNamespace(content="Sure! The route is kpi.")
            out = {"route": "kpi", "reason": "asks for revenue"}
        elif "KPI Extraction Agent" in system:
            out = {"revenue": 1804.5}
        elif "Answer Validation" in system:
            out = {"grounded": True, "confidence": 0.92, "unsupported_claims": []}
        elif "Context Agent" in system:
            out = {"standalone_question": "What was Vertex Energy plc's revenue in FY2024?",
                   "company": "Vertex Energy plc", "fiscal_year": 2024}
        else:
            return SimpleNamespace(content="Vertex Energy reported revenue of $1,670.8 million in FY2024 [1].")
        return SimpleNamespace(content="```json\n" + json.dumps(out) + "\n```")


@pytest.fixture
def fake(monkeypatch):
    def install(**kw):
        m = FakeModel(**kw)
        monkeypatch.setattr(llm, "_model", lambda name=None: m)
        monkeypatch.setattr(llm.settings.__class__, "llm_max_retries", 3, raising=False)
        return m
    return install


def test_llm_route_and_discrepancy_detection(graph, fake):
    fake()
    res = ask(graph, "llm-1", "What was Vertex Energy's revenue in FY2024?")
    assert res["route"] == "kpi"
    trace = {t["agent"]: t["detail"] for t in res["trace"]}
    assert "via llm" in trace["router_agent"]
    assert "extraction=['llm']" in trace["kpi_agent"]
    # LLM read 1804.5 from the letter; ledger says otherwise -> human review
    assert res["status"] == "awaiting_review"


def test_json_repair(graph, fake):
    fake(bad_json_once=True)
    res = ask(graph, "llm-2", "What was Vertex Energy's revenue in FY2024?")
    assert "via llm" in {t["agent"]: t["detail"] for t in res["trace"]}["router_agent"]


def test_retry_then_success(fake):
    m = fake(fail_first=1)
    assert "revenue" in llm.chat("generic system", "hi")
    assert m.calls == 2
