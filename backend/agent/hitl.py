"""
Human-in-the-loop (HITL) helpers for execution streams.

- build_paused_payload: build the status='paused' event with pending nodes/actions for review.
- process_execution_steps: consume stream_execution/resume_execution steps, handle __interrupt__
  and yield running / paused / complete events. Used by both run() and resume() in runtime.
"""
from typing import Dict, Any, List, Generator, Callable, Optional


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
    pending_actions = [action_by_type.get(n) or {"action_type": n} for n in next_nodes]
    completed_nodes = list(current_state.get("previous_results", {}).keys())
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
    sources_from_search_fn: Callable[[str], List[str]],
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
            final_state = get_current_state_fn(graph, thread_id)
            if final_state:
                current_state.update(final_state)
            prev = current_state.get("previous_results", {})
            result_text = prev.get("RESPONSE_GENERATION", "")
            search_result = prev.get("SEARCH_DOCUMENT", "")
            sources = sources_from_search_fn(search_result)
            if sources and result_text and "references:" not in result_text:
                result_text = result_text.rstrip() + "\n\nreferences: " + ", ".join(sources)
            yield {
                "phase": "execution",
                "status": "complete",
                "result": result_text,
                "all_results": prev,
            }
            return
        current = get_current_state_fn(graph, thread_id)
        if current:
            current_state.update(current)
        yield {
            "phase": "execution",
            "node": step.get("node"),
            "output": step.get("output"),
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
