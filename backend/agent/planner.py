"""
Planning agent module.

Responsible for converting user requests into a structured action plans.
"""
from typing import Optional, Dict, Any, List, Union
import json
from .state import AgentState
from .prompt_lib import PLANNING_PROMPT, TOOLS_DESCRIPTION, get_available_documents

try:
    from backend.db import get_available_tables
except ImportError:
    from db import get_available_tables
from .runtime import LLM

def _format_conversation_previous_results(
    results: Union[List[Dict[str, Any]], Dict[str, Any]]
) -> str:
    """Format conversation_previous_results for the planner prompt (truncate long values).

    Preferred format (plan shape + results per action, per turn):
      [{"turn": 1, "plan": "...", "need_clarification": false, "actions": [{action_type, description, ..., "result": "..."}, ...]}, ...]
    """
    if not results:
        return "(None)"
    lines = []
    if isinstance(results, list):
        for entry in results:
            if not isinstance(entry, dict) or "turn" not in entry:
                continue
            turn_num = entry["turn"]
            lines.append(f"--- Turn {turn_num} ---")
            # New format: plan + actions with result
            if "plan" in entry:
                plan = str(entry.get("plan", ""))
                if plan:
                    lines.append(f"  plan: {plan[:400]}{'...' if len(plan) > 400 else ''}")
                if entry.get("need_clarification"):
                    lines.append("  (clarification was requested)")
            if "actions" in entry and isinstance(entry["actions"], list):
                for action in entry["actions"]:
                    if not isinstance(action, dict):
                        continue
                    atype = action.get("action_type", "?")
                    desc = action.get("description", "")
                    if desc and len(desc) > 120:
                        desc = desc[:120] + "..."
                    result = action.get("result", "")
                    if len(result) > 400:
                        result = result[:400] + "... [truncated]"
                    lines.append(f"  {atype}: {desc}")
                    if result:
                        lines.append(f"    result: {result}")
            # Legacy format: contents (action_type -> result)
            elif "contents" in entry and isinstance(entry["contents"], dict):
                for action_type, value in entry["contents"].items():
                    s = str(value)
                    if len(s) > 500:
                        s = s[:500] + "... [truncated]"
                    lines.append(f"  {action_type}: {s}")
            lines.append("")
        if lines and lines[-1] == "":
            lines.pop()
    else:
        # Legacy dict format (action_type -> value or list of {turn, result})
        for k, v in results.items():
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict) and "turn" in item and "result" in item:
                        s = str(item["result"])
                        if len(s) > 300:
                            s = s[:300] + "... [truncated]"
                        lines.append(f"- {k} (Turn {item['turn']}): {s}")
                    else:
                        lines.append(f"- {k}: {str(item)[:300]}")
            else:
                s = str(v)
                if len(s) > 500:
                    s = s[:500] + "... [truncated]"
                lines.append(f"- {k}: {s}")
    return "\n".join(lines)


def planning_agent(state: AgentState,
                   llm=LLM,
                   config: Optional[Dict[str, Any]] = None,
                   conversation_previous_results: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None) -> AgentState:
    """
    Planning agent function.
    """
    messages = state.get("messages", [])
    recent_messages = messages[-10:] # last 10 messages
    previous_results = state.get("previous_results", {})
    conv_prev = conversation_previous_results or {}
    
    # Get available documents for context
    available_docs = get_available_documents()
    if not available_docs:
        available_docs = "(No documents currently stored in vector database)"

    # Get available tables (SQL DB) for context
    available_tables = get_available_tables()
    if not available_tables:
        available_tables = "(No table definition file found. Add backend/db/tables.yaml to describe SQL tables.)"

    # Action planning
    planning_chain = PLANNING_PROMPT | llm
    planning_result = planning_chain.invoke({
        "user_request": messages[-1].content if messages else None,
        "conversation_history": recent_messages,
        "previous_results": previous_results,
        "conversation_previous_results": _format_conversation_previous_results(conv_prev),
        "tools_description": TOOLS_DESCRIPTION,
        "available_documents": available_docs,
        "available_tables": available_tables,
    })

    planning_response = json.loads(planning_result.content)
    need_clarification = planning_response.get("need_clarification", False)
    actions = planning_response.get("actions", [])

    # Ensure RESPONSE_GENERATION is always last when we have a non-clarification plan
    if not need_clarification and actions:
        has_response_gen = any(
            a.get("action_type") == "RESPONSE_GENERATION"
            for a in actions
            if isinstance(a, dict)
        )
        if not has_response_gen:
            max_order = max(
                (a.get("execution_order", 0) for a in actions if isinstance(a, dict)),
                default=0,
            )
            last_action_types = [
                a.get("action_type")
                for a in actions
                if isinstance(a, dict) and a.get("execution_order") == max_order
            ]
            actions = list(actions) + [
                {
                    "action_type": "RESPONSE_GENERATION",
                    "description": "Synthesize previous results and provide final response to the user",
                    "dependencies": last_action_types or [],
                    "execution_order": max_order + 1,
                }
            ]

    return {
        "need_clarification": need_clarification,
        "plan": planning_response.get("plan", ""),
        "actions": actions,
    }