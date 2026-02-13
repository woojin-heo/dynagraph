from typing import TypedDict, List, Dict, Any
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: List[BaseMessage]
    previous_results: Dict[str, Any]
    need_clarification: bool
    plan: str
    actions: List[Dict[str, Any]]
