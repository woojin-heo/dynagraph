from langgraph.graph import StateGraph
from .state import AgentState
from .prompt_lib import ACTION_PROMPT
from .runtime import LLM
from .tools import tavily_search, wikipedia_search

# tool mapping
ACTION_TOOLS = {
    "SEARCH_TAVILY": tavily_search,
    "SEARCH_WIKIPEDIA": wikipedia_search,
    "SEARCH_DOCUMENT": None,
}

def action_executor(action, state: AgentState, llm=LLM):
    """
    Execute an action based on its type.
    - LLM based actions: prompt | llm chain execution
    - Tool based actions: call the tool function
    """
    action_type = action.get('action_type', 'unknown')
    description = action.get('description', '')
    conversation_history = state.get('messages', [])[-10:] # last 10 messages
    previous_results = state.get('previous_results', {})
    user_request = conversation_history[-1].content if conversation_history else ''

    # LLM prompt for the action
    prompt = ACTION_PROMPT.get(action_type)
    if prompt:
        chain = prompt | llm
        result = chain.invoke({
            "conversation_history": conversation_history,
            "previous_results": previous_results,
            "user_request": user_request,
            "description": description
        })
        return {"action_type": action_type, "result": result.content}

    tool = ACTION_TOOLS.get(action_type)
    if tool:
        result = tool(description)
        return {"action_type": action_type, "result": result}

    return {"action_type": action_type, "result": "Action not found"}

def create_graph(actions, enable_htil=True):
    """
    Create a graph from the actions.
    Args:
        actions: A list of actions.
        enable_htil: Whether to enable human in the loop.
    Returns:
        A graph.
    """
    graph = StateGraph(AgentState)
    
    # sort actions by execution_order
    actions = sorted(actions, key=lambda x: x.get('execution_order', 0))

    # add nodes for each action
    for action in actions:
        node_name = f"action_{action.get('action_type', 'unknown')}"
        graph.add_node(node_name, action_executor(action))

    # add edges between nodes
    for i in range(len(actions) - 1):
        graph.add_edge(actions[i]['action_type'], actions[i + 1]['action_type'])

    return graph