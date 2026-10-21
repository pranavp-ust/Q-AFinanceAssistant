# Demo & Presentation Guide (30 minutes)

Follows the "Recommended Demo Flow" in the capstone guide. Every member presents one part. That matters because "equal participation" is worth 5 marks.

## Suggested team split (4-5 members)
| Member | Owns (build + present) | Files |
|---|---|---|
| 1 | Problem statement, synthetic data, RAG ingestion | `data/generate_data.py`, `src/vectorstore.py`, `ingest.py` |
| 2 | Architecture, LangGraph workflow (sequential / parallel / router) | `src/graph.py`, `src/state.py`, `docs/ARCHITECTURE.md` |
| 3 | Agents & prompts (context, router, retrieval, summary) | `src/agents/*`, `src/prompts.py` |
| 4 | KPI agent, tools, validation & human-in-the-loop | `src/agents/kpi_agent.py`, `src/tools.py`, `src/agents/validation_agent.py`, `src/agents/human_review.py` |
| 5 | Robustness, logging/monitoring, UI, evaluation & tests | `src/llm.py`, `src/monitoring.py`, `app.py`, `evaluate.py`, `tests/` |

## Timeline
| # | Section | Min | Content |
|---|---|---|---|
| 1 | Problem statement | 3 | Annual reports run 100+ pages, and analysts spend hours finding KPIs and checking numbers. Goal: instant, cited, verified answers with a human in control |
| 2 | Architecture | 5 | System diagram + agent table from `ARCHITECTURE.md`. Explain the tech stack: LangGraph, OpenAI/Gemini, ChromaDB, LangChain splitter, Streamlit |
| 3 | Workflow | 5 | Show the LangGraph diagram: sequential spine, router, parallel fan-out for `full_analysis`, retrieval retry loop, HITL interrupt |
| 4 | Live demo | 7 | Script below |
| 5 | Challenges | 3 | See list below |
| 6 | Q&A | 7 | See prepared answers below |

## Live demo script (7 min)
Before you start: run `python ingest.py`, then `streamlit run app.py`, and open the sidebar so the monitoring panel is visible.

1. **KPI + citation**: "What was Nimbus Retail's revenue in FY2024?" → open *Sources, agent trace & KPIs* and show the route, tools used and latency per agent.
2. **Memory**: "And what about the previous year?" → resolves to FY2023 without restating the company.
3. **QA with RAG**: "What are the principal risks for Orion Software?" → cited answer.
4. **Comparison + calculator tool**: "Compare net margin and debt-to-equity across all companies in FY2024".
5. **Parallel + HITL (the highlight)**: "Give me a full analysis of Vertex Energy FY2024".
   * Trace shows `kpi_agent` and `summary_agent` running in parallel.
   * The review panel shows *"document states revenue = 1,804.5 but ledger has 1,670.8 (8.0% difference)"*.
   * Click **Approve with edits**, correct the revenue line, and add a comment.
6. **Guardrails**: "Should I buy Orion shares?" → compliance review → **Reject**. Then "What's the weather in Paris?" → out of scope.
7. **Robustness**: stop and restart the app with `LLM_PROVIDER=none`, then ask question 1 again. It still answers, through the fallbacks. Point at the *Fallbacks* metric.
8. Show `logs/agent_trace.jsonl` and the `python evaluate.py` summary.

## Challenges faced (talking points)
* **Hallucinated numbers.** The fix was the ledger tool + calculator, plus a cross-check that routes mismatches to a human.
* **Follow-up questions.** Pronouns and "previous year" needed a context agent. Naive memory leaked old companies into unrelated questions, so inheritance only triggers on follow-up cues.
* **Exact financial terms vs semantic search.** Pure vector search missed "EBITDA"/"free cash flow" chunks. Hybrid BM25 + vector with RRF fixed it.
* **Corporate network.** The embedding model download failed with SSL errors behind the proxy. We added an offline hashing embedding and `pip-system-certs`.
* **Windows path length.** reportlab failed in deep folders. We keep the project in a short path.
* **Parallel state merges.** Two agents writing the same key in one super-step conflicts. We used separate keys plus list reducers for the trace.
* **HITL inside a node wrapper.** Our error-catching wrapper swallowed LangGraph's interrupt exception. It now re-raises `GraphBubbleUp`.

## Q&A preparation
| Likely question | Answer |
|---|---|
| Why LangGraph over CrewAI? | Explicit state machine: conditional edges, parallel super-steps, checkpointed `interrupt()` for HITL, and replayable state. Orchestration is the core of the rubric |
| How do you prevent hallucination? | Grounding prompt + citations, validation agent, numbers from ledger/calculator, discrepancy → human |
| How is memory implemented? | Checkpointer per thread_id; `messages` reducer; entity memory; LLM rewrite to a standalone question |
| What if the LLM API is down? | Retries → fallback model → rule-based fallback per agent. The demo works fully offline |
| How do you measure RAG quality? | `evaluate.py`: retrieval hit-rate@3, KPI accuracy against ground truth, routing accuracy, latency |
| How would you scale to real filings? | Swap to a managed vector DB, table-aware PDF parsing (e.g. unstructured / LlamaParse), XBRL as the ledger, SqliteSaver/PostgresSaver checkpointer, auth + audit log for reviewers |
| Why chunk size 600? | Paragraph-sized chunks keep one topic each. Smaller chunks lost the "Revenue ... was $X" context; larger ones diluted retrieval |
| Where does parallelism help? | Full analysis: KPI extraction and summarisation are independent, so they run concurrently and the latency is max(a, b) rather than a + b |
