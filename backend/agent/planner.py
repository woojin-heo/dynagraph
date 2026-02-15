"""
Planning agent module.

Responsible for converting user requests into a structured action plans.
"""
from typing import Optional, Dict, Any
import json
from .state import AgentState
from .prompt_lib import PLANNING_PROMPT, TOOLS_DESCRIPTION
from .runtime import LLM

def planning_agent(state: AgentState,
                   llm=LLM,
                   config: Optional[Dict[str, Any]] = None) -> AgentState:
    """
    Planning agent function.
    """
    messages = state.get("messages", [])
    recent_messages = messages[-10:] # last 10 messages
    previous_results = state.get("previous_results", {})

    # Action planning
    planning_chain = PLANNING_PROMPT | llm
    planning_result = planning_chain.invoke({
        "user_request": messages[-1].content if messages else None,
        "conversation_history": recent_messages,
        "previous_results": previous_results,
        "tools_description": TOOLS_DESCRIPTION,
    })

    planning_response = json.loads(planning_result.content)
    return {
        "need_clarification": planning_response.get("need_clarification", False),
        "plan": planning_response.get("plan", ""),
        "actions": planning_response.get("actions", []),
    }