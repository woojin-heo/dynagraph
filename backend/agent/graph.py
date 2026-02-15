from functools import partial
from typing import Dict, Any, List
from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .prompt_lib import ACTION_PROMPT
from .runtime import LLM
from .tools.general_tools import TOOLS_MAP


def action_executor(action: Dict[str, Any], state: AgentState, llm=LLM) -> Dict[str, Any]:
    """
    Execute an action based on its type.
    - LLM based actions: prompt | llm chain execution
    - Tool based actions: call the tool function
    
    Returns state updates (previous_results will be updated with this action's result)
    """
    action_type = action.get('action_type', 'unknown')
    description = action.get('description', '')
    params = action.get('params', {})

    conversation_history = state.get('messages', [])[-10:] # last 10 messages
    previous_results = state.get('previous_results', {})
    user_request = conversation_history[-1].content if conversation_history else ''

    result = None

    # LLM based actions: use prompt
    prompt = ACTION_PROMPT.get(action_type)
    if prompt:
        chain = prompt | llm
        llm_result = chain.invoke({
            "conversation_history": conversation_history,
            "previous_results": previous_results,
            "user_request": user_request,
            "description": description
        })
        result = llm_result.content

    # Tool based actions: call the tool function
    tool = TOOLS_MAP.get(action_type)
    if tool:
        result = tool.invoke(params)

    if result is None:
        result = f"Action {action_type} not found"

    # Update previous_results with this action's result
    updated_results = {**previous_results, action_type: result}
    
    return {
        "previous_results": updated_results,
    }


def create_execution_graph(actions: List[Dict[str, Any]], enable_hitl: bool = True):
    """
    Create an execution graph from the planned actions.
    
    Args:
        actions: A list of actions from the planner.
        enable_hitl: Whether to enable human in the loop (not implemented yet).
    
    Returns:
        A compiled StateGraph ready to be invoked.
    """
    graph = StateGraph(AgentState)
    
    if not actions:
        return None
    
    # Sort actions by execution_order
    sorted_actions = sorted(actions, key=lambda x: x.get('execution_order', 0))

    # Add nodes for each action
    for action in sorted_actions:
        node_name = action.get('action_type', 'unknown')
        # partial: action is fixed, state will be passed by LangGraph at runtime
        graph.add_node(node_name, partial(action_executor, action))

    # Add edges between nodes (sequential for now)
    # TODO: Handle parallel execution for same execution_order
    node_names = [a.get('action_type', 'unknown') for a in sorted_actions]
    
    # START -> first node
    graph.add_edge(START, node_names[0])
    
    # Connect nodes sequentially
    for i in range(len(node_names) - 1):
        graph.add_edge(node_names[i], node_names[i + 1])
    
    # Last node -> END
    graph.add_edge(node_names[-1], END)

    return graph.compile()
