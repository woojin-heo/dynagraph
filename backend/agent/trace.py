"""
Structured tracing for agent execution debugging.

Provides:
- TraceCollector: per-conversation trace event store
- @traced decorator: automatic function entry/exit recording via contextvars
- File logging to logs/agent.jsonl (JSON Lines)

Usage:
    from .trace import TraceCollector, traced, get_current_trace, record

    # In ConversationAgent.run():
    token = set_current_trace(self.trace)
    try: ...
    finally: reset_current_trace(token)

    # In any instrumented function:
    @traced("planning")
    def planning_agent(state, ...): ...

    # Ad-hoc recording inside a function:
    record("llm_call", prompt_vars={...}, response="...")
"""
import contextvars
import functools
import json
import logging
import os
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Callable, Dict, List, Optional


_TRACE_MAX_STR = 2000

_LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
_LOG_FILE = os.path.join(_LOG_DIR, "agent.jsonl")

_logger = logging.getLogger("dynagraph.agent")
_logger.setLevel(logging.DEBUG)
_logger.propagate = False

if not _logger.handlers:
    os.makedirs(_LOG_DIR, exist_ok=True)
    _fh = RotatingFileHandler(_LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5)
    _fh.setLevel(logging.DEBUG)

    class _JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            data = record.__dict__.get("trace_data", {})
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "conversation_id": data.get("conversation_id", ""),
                "event": data.get("event_type", record.getMessage()),
                "data": {k: v for k, v in data.items() if k not in ("conversation_id", "event_type")},
            }
            return json.dumps(entry, default=str, ensure_ascii=False)

    _fh.setFormatter(_JsonFormatter())
    _logger.addHandler(_fh)


def _truncate(value: Any, limit: int = _TRACE_MAX_STR) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"... [truncated, total {len(value)} chars]"
    if isinstance(value, dict):
        return {k: _truncate(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [_truncate(v, limit) for v in value]
    return value


class TraceCollector:
    """Collects structured trace events for one conversation."""

    def __init__(self, conversation_id: str, log_to_file: bool = True):
        self.conversation_id = conversation_id
        self.events: List[Dict[str, Any]] = []
        self._log_to_file = log_to_file

    def record(self, event_type: str, **data: Any) -> None:
        event = {
            "event_type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "conversation_id": self.conversation_id,
            **_truncate(data),
        }
        self.events.append(event)
        if self._log_to_file:
            _logger.info(event_type, extra={"trace_data": event})

    def get_trace(self, event_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if event_types is None:
            return list(self.events)
        types_set = set(event_types)
        return [e for e in self.events if e["event_type"] in types_set]

    def get_summary(self) -> Dict[str, Any]:
        turns: Dict[int, Dict[str, Any]] = {}
        current_turn = 0
        for ev in self.events:
            et = ev["event_type"]
            if et == "turn_start":
                current_turn = ev.get("turn", current_turn + 1)
                turns[current_turn] = {"turn": current_turn, "user_message": ev.get("user_message", ""), "steps": [], "errors": []}
            bucket = turns.get(current_turn)
            if bucket is None:
                turns[current_turn] = {"turn": current_turn, "steps": [], "errors": []}
                bucket = turns[current_turn]
            if et == "planning_end":
                bucket["plan"] = ev.get("plan", "")
                bucket["actions"] = ev.get("actions", [])
            elif et == "action_end":
                bucket["steps"].append({
                    "action_type": ev.get("action_type", ""),
                    "kind": ev.get("kind", ""),
                    "duration_ms": ev.get("duration_ms"),
                    "result_preview": _truncate(ev.get("result", ""), 300),
                })
            elif et == "error":
                bucket["errors"].append(ev.get("error", ""))
            elif et == "turn_complete":
                bucket["total_duration_ms"] = ev.get("duration_ms")
                bucket["final_result_preview"] = _truncate(ev.get("result", ""), 500)
        return {"conversation_id": self.conversation_id, "turns": list(turns.values())}


# ---------------------------------------------------------------------------
# Context variable: lets any function access the current trace without params
# ---------------------------------------------------------------------------

_current_trace: contextvars.ContextVar[Optional[TraceCollector]] = contextvars.ContextVar(
    "current_trace", default=None
)


def set_current_trace(tc: TraceCollector) -> contextvars.Token:
    return _current_trace.set(tc)


def reset_current_trace(token: contextvars.Token) -> None:
    try:
        _current_trace.reset(token)
    except ValueError:
        # Streaming responses can finalize generators in a different context.
        # Avoid crashing request handling when contextvars token reset is not valid.
        _current_trace.set(None)


def get_current_trace() -> Optional[TraceCollector]:
    return _current_trace.get()


def record(event_type: str, **data: Any) -> None:
    """Record an event on the current trace (no-op if no trace is active)."""
    tc = _current_trace.get()
    if tc is not None:
        tc.record(event_type, **data)


# ---------------------------------------------------------------------------
# @traced decorator
# ---------------------------------------------------------------------------

def traced(event_type: str) -> Callable:
    """Decorator that records {event_type}_start / {event_type}_end automatically."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tc = _current_trace.get()
            if tc is None:
                return fn(*args, **kwargs)
            tc.record(f"{event_type}_start", function=fn.__name__)
            t0 = time.time()
            try:
                result = fn(*args, **kwargs)
                duration_ms = round((time.time() - t0) * 1000, 1)
                tc.record(f"{event_type}_end", function=fn.__name__, duration_ms=duration_ms)
                return result
            except Exception as exc:
                duration_ms = round((time.time() - t0) * 1000, 1)
                tc.record("error", function=fn.__name__, error=str(exc), duration_ms=duration_ms)
                raise
        return wrapper
    return decorator
