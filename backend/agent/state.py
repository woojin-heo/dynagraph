from typing import TypedDict, List, Dict, Any, Annotated, NotRequired
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


def merge_results(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reducer for merging previous_results from parallel nodes.
    
    When multiple nodes run in parallel, each returns its result.
    This reducer combines them into a single dict.
    
    Example:
        left:  {"SEARCH_TAVILY": "tavily result"}
        right: {"SEARCH_WIKIPEDIA": "wiki result"}
        result: {"SEARCH_TAVILY": "tavily result", "SEARCH_WIKIPEDIA": "wiki result"}
    """
    if left is None:
        return right or {}
    if right is None:
        return left
    return {**left, **right}


class AgentState(TypedDict):
    # messages: conversation history (using add_messages reducer)
    messages: Annotated[List[BaseMessage], add_messages]
    
    # previous_results: each action's result (merged when running in parallel)
    previous_results: Annotated[Dict[str, Any], merge_results]
    
    # the following fields do not need reducers (single value, overwrite)
    need_clarification: bool
    plan: str
    actions: List[Dict[str, Any]]
    # HITL: param overrides per action_type (set on resume from human_input)
    human_param_overrides: NotRequired[Dict[str, Dict[str, Any]]]
