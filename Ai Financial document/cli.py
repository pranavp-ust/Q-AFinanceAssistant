"""Terminal chat for the Financial Document Q&A Assistant.

Usage:
  python cli.py                    interactive chat (human review prompts in the terminal)
  python cli.py --auto-approve     approve every review automatically (demo / scripted runs)
  python cli.py -q "question"      ask one question and exit
Commands inside the chat:  /new  (new conversation)   /metrics   /trace   /quit
"""
import argparse
import json
import uuid

from src.graph import ask, build_graph, review
from src.llm import llm_enabled
from src.monitoring import metrics


def print_result(res: dict, show_trace: bool = False):
    print(f"\n[route: {res.get('route')} | confidence: {res.get('confidence') or 0:.2f} | "
          f"companies: {res.get('companies')} | FY: {res.get('fiscal_year')}]")
    print(res.get("answer") or "")
    cites = res.get("citations") or []
    if cites:
        print("\nSources: " + "; ".join(f"[{c['n']}] {c['source']} p.{c['page']}" for c in cites[:5]))
    if show_trace:
        for t in res.get("trace") or []:
            print(f"  - {t['agent']:<16} {t['status']:<6} {t['ms']:>7.1f} ms  {t['detail']}")


def handle(graph, thread, question, auto_approve, show_trace):
    res = ask(graph, thread, question)
    while res["status"] == "awaiting_review":
        rv = res["review"]
        print("\n" + "=" * 70 + "\nHUMAN REVIEW REQUIRED")
        for r in rv["reasons"]:
            print(f"  * {r}")
        print("-" * 70 + "\nDRAFT ANSWER:\n" + rv["draft_answer"] + "\n" + "=" * 70)
        if auto_approve:
            action, edited, comment = "approve", "", "auto-approved (demo mode)"
            print("[auto-approve]")
        else:
            choice = input("Approve [a] / Edit [e] / Reject [r]? ").strip().lower()[:1] or "a"
            action = {"a": "approve", "e": "edit", "r": "reject"}.get(choice, "approve")
            edited = input("Enter corrected answer:\n") if action == "edit" else ""
            comment = input("Reviewer comment (optional): ").strip()
        res = review(graph, thread, action, edited, comment)
    print_result(res, show_trace)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--question")
    ap.add_argument("--auto-approve", action="store_true")
    ap.add_argument("--trace", action="store_true", help="print agent trace after every answer")
    args = ap.parse_args()

    graph = build_graph()
    thread = uuid.uuid4().hex[:8]
    print(f"Financial Document Q&A Assistant  (LLM: {'on' if llm_enabled() else 'OFF - rule-based fallbacks'})")
    if args.question:
        handle(graph, thread, args.question, args.auto_approve, True)
        return
    print("Ask about Nimbus Retail, Orion Software or Vertex Energy. /new /metrics /trace /quit")
    show_trace = args.trace
    while True:
        try:
            q = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q == "/quit":
            break
        if q == "/new":
            thread = uuid.uuid4().hex[:8]
            print("Started a new conversation.")
            continue
        if q == "/metrics":
            print(json.dumps(metrics.snapshot(), indent=2))
            continue
        if q == "/trace":
            show_trace = not show_trace
            print(f"trace {'on' if show_trace else 'off'}")
            continue
        handle(graph, thread, q, args.auto_approve, show_trace)


if __name__ == "__main__":
    main()
