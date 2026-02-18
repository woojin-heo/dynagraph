"""
Planning agent module.

Responsible for converting user requests into a structured action plans.
"""
from typing import Optional, Dict, Any
import json
from .state import AgentState
from .prompt_lib import PLANNING_PROMPT, TOOLS_DESCRIPTION, get_available_documents
from .runtime import LLM

def _format_conversation_previous_results(results: Dict[str, Any]) -> str:
    """Format conversation_previous_results for the planner prompt (truncate long values).
    
    Results can be either:
    - Old format: {"SEARCH_DOCUMENT": "result"} (for backward compatibility)
    - New format: {"SEARCH_DOCUMENT": [{"turn": 1, "result": "..."}, {"turn": 2, "result": "..."}]}
    """
    if not results:
        return "(None)"
    lines = []
    for k, v in results.items():
        if isinstance(v, list):
            # New format: list of turn results
            turn_results = []
            for item in v:
                if isinstance(item, dict) and "turn" in item and "result" in item:
                    turn_num = item["turn"]
                    result = str(item["result"])
                    if len(result) > 300:
                        result = result[:300] + "... [truncated]"
                    turn_results.append(f"  Turn {turn_num}: {result}")
                else:
                    # Fallback for unexpected format
                    turn_results.append(f"  {str(item)[:300]}")
            if turn_results:
                lines.append(f"- {k} (from {len(v)} turn(s)):")
                lines.extend(turn_results)
        else:
            # Old format: direct value (for backward compatibility)
            s = str(v)
            if len(s) > 500:
                s = s[:500] + "... [truncated]"
            lines.append(f"- {k}: {s}")
    return "\n".join(lines)


def planning_agent(state: AgentState,
                   llm=LLM,
                   config: Optional[Dict[str, Any]] = None,
                   conversation_previous_results: Optional[Dict[str, Any]] = None) -> AgentState:
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

    # Action planning
    planning_chain = PLANNING_PROMPT | llm
    planning_result = planning_chain.invoke({
        "user_request": messages[-1].content if messages else None,
        "conversation_history": recent_messages,
        "previous_results": previous_results,
        "conversation_previous_results": _format_conversation_previous_results(conv_prev),
        "tools_description": TOOLS_DESCRIPTION,
        "available_documents": available_docs,
    })

    planning_response = json.loads(planning_result.content)
    return {
        "need_clarification": planning_response.get("need_clarification", False),
        "plan": planning_response.get("plan", ""),
        "actions": planning_response.get("actions", []),
    }