"""
Human-in-the-loop (HITL) helpers for execution streams.

- build_paused_payload: build the status='paused' event with pending nodes/actions for review.
- process_execution_steps: consume stream_execution/resume_execution steps, handle __interrupt__
  and yield running / paused / complete events. Used by both run() and resume() in runtime.
"""
import re
from typing import Dict, Any, List, Generator, Callable, Optional


def _format_sql_for_display(raw: str) -> str:
    """Insert newlines before major SQL keywords for readability."""
    if not raw or not raw.strip():
        return raw
    s = raw.strip()
    for keyword in ("FROM", "WHERE", "GROUP BY", "ORDER BY", "LEFT JOIN", "INNER JOIN", "RIGHT JOIN", "JOIN", "HAVING", "LIMIT", "OFFSET", "UNION", "EXCEPT", "INTERSECT"):
        s = re.sub(rf"\s+({keyword})\s+", r"\n\1 ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+(AND)\s+", r"\n  \1 ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+(OR)\s+", r"\n  \1 ", s, flags=re.IGNORECASE)
    return s.strip()


def build_paused_payload(
    graph,
    thread_id: str,
    current_state: Dict[str, Any],
    user_message: str,
    hitl_before: List[str],
) -> Dict[str, Any]:
    """Build the payload for status='paused' (pending nodes, actions, plan, etc.)."""
    graph_state = graph.get_state({"configurable": {"thread_id": thread_id}}) if graph else None
    next_nodes = getattr(graph_state, "next", None) or () if graph_state else ()
    if isinstance(next_nodes, (list, tuple)):
        next_nodes = list(next_nodes)
    else:
        next_nodes = []
    if not next_nodes and graph_state and getattr(graph_state, "tasks", None):
        next_nodes = [
            t.get("id") if isinstance(t, dict) else getattr(t, "id", getattr(t, "name", None))
            for t in graph_state.tasks
        ]
        next_nodes = [n for n in next_nodes if n]
    if not next_nodes and hitl_before:
        next_nodes = list(hitl_before)
    actions_list = current_state.get("actions", [])
    action_by_type = {a.get("action_type"): a for a in actions_list}
    previous_results = current_state.get("previous_results", {})

    pending_actions = []
    for n in next_nodes:
        action = action_by_type.get(n) or {"action_type": n}
        action = dict(action)
        params = dict(action.get("params") or {})
        if n == "SQL_EXECUTION":
            raw = (previous_results.get("SQL_GENERATION") or params.get("query") or "").strip()
            params["query"] = _format_sql_for_display(raw)
        action["params"] = params
        pending_actions.append(action)

    completed_nodes = [k for k in previous_results if not k.startswith("__")]
    return {
        "phase": "execution",
        "status": "paused",
        "message": "Execution paused for human review. Call agent.resume() to continue.",
        "paused_before_nodes": next_nodes,
        "pending_actions": pending_actions,
        "completed_nodes": completed_nodes,
        "user_message": user_message,
        "plan": current_state.get("plan", ""),
    }


def process_execution_steps(
    step_iterator,
    current_state: Dict[str, Any],
    user_message: str,
    get_current_state_fn: Callable,
    graph,
    thread_id: str,
    hitl_before: List[str],
    set_pending_message: Callable[[str], None],
    collect_sources_fn: Callable[[Dict[str, Any]], List[str]],
    format_references_fn: Callable[[List[str]], str] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Consume execution steps (from stream_execution or resume_execution), handle
    __interrupt__ / paused / complete. Yields running steps, then either
    status='paused' or status='complete' (with result/all_results).
    Caller appends to conversation_previous_results and messages when status='complete'.
    """
    saw_interrupt = False
    for step in step_iterator:
        if step.get("node") == "__interrupt__":
            saw_interrupt = True
            current = get_current_state_fn(graph, thread_id)
            if current:
                current_state.update(current)
            yield {
                "phase": "execution",
                "node": "__interrupt__",
                "output": step.get("output"),
                "status": "running",
            }
            continue
        if step.get("status") == "complete":
            if saw_interrupt:
                set_pending_message(user_message)
                yield build_paused_payload(
                    graph, thread_id, current_state, user_message, hitl_before
                )
                return
            # LangGraph may not yield __interrupt__; check graph state for pending tasks
            if graph and not saw_interrupt:
                try:
                    graph_state = graph.get_state({"configurable": {"thread_id": thread_id}})
                    if graph_state and getattr(graph_state, "tasks", None):
                        current = get_current_state_fn(graph, thread_id)
                        if current:
                            current_state.update(current)
                        set_pending_message(user_message)
                        yield build_paused_payload(
                            graph, thread_id, current_state, user_message, hitl_before
                        )
                        return
                except Exception:
                    pass
            final_state = get_current_state_fn(graph, thread_id)
            if final_state:
                current_state.update(final_state)
            prev = current_state.get("previous_results", {})
            result_text = prev.get("RESPONSE_GENERATION", "")
            viz_image = prev.get("__VISUALIZATION_IMAGE__", "")
            if viz_image:
                result_text = result_text.rstrip() + "\n\n" + viz_image
            sources = collect_sources_fn(prev)
            if sources and result_text and "references:" not in result_text:
                if format_references_fn:
                    result_text = result_text.rstrip() + format_references_fn(sources)
                else:
                    result_text = result_text.rstrip() + "\n\nreferences:\n" + "\n".join(f"- {s}" for s in sources)
            safe_prev = {k: v for k, v in prev.items() if not k.startswith("__")}
            yield {
                "phase": "execution",
                "status": "complete",
                "result": result_text,
                "all_results": safe_prev,
            }
            return
        current = get_current_state_fn(graph, thread_id)
        if current:
            current_state.update(current)
        raw_output = step.get("output")
        if isinstance(raw_output, dict) and "previous_results" in raw_output:
            raw_output = {
                **raw_output,
                "previous_results": {
                    k: v for k, v in raw_output["previous_results"].items()
                    if not k.startswith("__")
                },
            }
        yield {
            "phase": "execution",
            "node": step.get("node"),
            "output": raw_output,
            "status": "running",
        }
    # Fallback: no __interrupt__ but graph may be paused (e.g. different LangGraph version)
    if graph and not saw_interrupt:
        try:
            graph_state = graph.get_state({"configurable": {"thread_id": thread_id}})
            if graph_state and getattr(graph_state, "tasks", None):
                set_pending_message(user_message)
                yield build_paused_payload(
                    graph, thread_id, current_state, user_message, hitl_before
                )
        except Exception:
            pass
