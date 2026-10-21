"""Logging and monitoring.

* Every agent decision is written as one JSON line to logs/agent_trace.jsonl
* Human-readable logs go to logs/app.log and the console
* An in-process Metrics object counts requests, routes, retries, fallbacks,
  approvals and latency; the Streamlit sidebar and `python cli.py --metrics` show it.
"""
import functools
import json
import logging
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

from langgraph.errors import GraphBubbleUp

from src.config import settings

_fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
logger = logging.getLogger("findoc")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(settings.logs_dir / "app.log", encoding="utf-8")
    fh.setFormatter(_fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(_fmt)
    sh.setLevel(logging.WARNING)
    logger.addHandler(fh)
    logger.addHandler(sh)

_trace_lock = threading.Lock()
TRACE_FILE = settings.logs_dir / "agent_trace.jsonl"


def log_event(event: dict) -> None:
    event = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), **event}
    with _trace_lock, open(TRACE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, default=str) + "\n")


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self.counters: Counter = Counter()
        self.routes: Counter = Counter()
        self.latency_ms: dict[str, list[float]] = defaultdict(list)

    def inc(self, name: str, n: int = 1):
        with self._lock:
            self.counters[name] += n

    def route(self, route: str):
        with self._lock:
            self.routes[route] += 1

    def observe(self, node: str, ms: float):
        with self._lock:
            self.latency_ms[node].append(ms)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "counters": dict(self.counters),
                "routes": dict(self.routes),
                "avg_latency_ms": {k: round(sum(v) / len(v), 1) for k, v in self.latency_ms.items() if v},
            }


metrics = Metrics()


def agent_node(name: str):
    """Decorator for LangGraph nodes: timing, structured trace, metrics and graceful failure.

    If a node raises, the error is logged and the node returns `{"errors": [...]}` instead of
    crashing the whole graph. Downstream nodes check what is missing and degrade gracefully.
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(state):
            start = time.perf_counter()
            status, update = "ok", {}
            try:
                update = fn(state) or {}
            except GraphBubbleUp:  # human-in-the-loop interrupt: must reach LangGraph
                log_event({"thread": state.get("thread_id"), "turn": state.get("turn_id"),
                           "agent": name, "status": "interrupted"})
                raise
            except Exception as exc:  # noqa: BLE001 - we want to catch everything at node boundary
                status = "error"
                logger.exception("node %s failed", name)
                metrics.inc("node_errors")
                update = {"errors": [f"{name}: {type(exc).__name__}: {exc}"]}
            ms = (time.perf_counter() - start) * 1000
            metrics.observe(name, ms)
            entry = {"agent": name, "status": status, "ms": round(ms, 1),
                     "detail": update.pop("_trace_detail", "")}
            log_event({"thread": state.get("thread_id"), "turn": state.get("turn_id"), **entry})
            update["trace"] = [entry]
            return update
        return wrapper
    return deco
