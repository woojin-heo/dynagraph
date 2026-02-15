from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()
LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# These imports come after LLM definition to avoid circular import issues
from .state import AgentState
from .planner import planning_agent
from .graph import create_execution_graph


def run_agent(user_message: str) -> Dict[str, Any]:
    """
    Run the full agent pipeline:
    1. Planning phase: analyze user request and create action plan
    2. Execution phase: execute each action in the plan
    
    Args:
        user_message: The user's input message.
    
    Returns:
        Final state with all results.
    """
    # Initial state
    state: AgentState = {
        "messages": [HumanMessage(content=user_message)],
        "previous_results": {},
        "need_clarification": False,
        "plan": "",
        "actions": [],
    }
    
    # Phase 1: Planning
    planner_result = planning_agent(state)
    state.update(planner_result)
    
    # Check if clarification is needed
    if state.get("need_clarification"):
        return {
            "status": "need_clarification",
            "plan": state.get("plan", ""),
            "state": state,
        }
    
    # Phase 2: Execution
    actions = state.get("actions", [])
    if not actions:
        return {
            "status": "no_actions",
            "plan": state.get("plan", ""),
            "state": state,
        }
    
    # Create and run execution graph
    execution_graph = create_execution_graph(actions)
    if execution_graph:
        final_state = execution_graph.invoke(state)
        state.update(final_state)
    
    return {
        "status": "complete",
        "plan": state.get("plan", ""),
        "result": state.get("previous_results", {}).get("RESPONSE_GENERATION", ""),
        "state": state,
    }