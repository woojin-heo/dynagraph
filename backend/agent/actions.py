"""
Unified Action Registry

All actions (LLM-based and tool-based) are registered here.
This makes it easy to:
1. See all available actions in one place
2. Debug missing actions
3. Add new actions consistently
"""
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any
from langchain_core.prompts import ChatPromptTemplate

from .tools import tavily_search, wikipedia_search, search_document


@dataclass
class ActionDefinition:
    """Definition of an action."""
    action_type: str
    kind: str  # "llm" or "tool"
    prompt: Optional[ChatPromptTemplate] = None  # for LLM actions
    tool: Optional[Callable] = None  # for tool actions
    description: str = ""  # human-readable description


# =============================================================================
# Prompt definitions (kept here for single-file visibility)
# =============================================================================

REASONING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a reasoning agent. Analyze the current context and provide logical analysis.

Previous results from other actions:
{previous_results}

Conversation history:
{conversation_history}

Provide clear, structured reasoning about the task at hand."""),
    ("human", "{description}"),
])

CONTEXT_REFERENCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a context extraction agent. Extract and organize relevant information.

Conversation history:
{conversation_history}

Previous results from other actions:
{previous_results}

Extract the most relevant context needed to address the user's request."""),
    ("human", "{description}"),
])

RESPONSE_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a response generation agent. Create a final response for the user.

User's original request: {user_request}

Previous results from other actions:
{previous_results}

Generate a clear, well-structured response that directly addresses the user's original request.

References rule: Add "references: source1, source2" at the end of your response ONLY when SEARCH_DOCUMENT was executed in this turn (i.e. "Previous results from other actions" above contains SEARCH_DOCUMENT output with <Document source="..."> tags). List the document source names from those tags. Do NOT add a references line when the response is based only on CONTEXT_REFERENCE, REASONING, or other actions—even if the content originally came from documents in a previous turn. In those cases, omit the references line entirely."""),
    ("human", "{description}"),
])


# =============================================================================
# Action Registry
# =============================================================================

ACTION_REGISTRY: Dict[str, ActionDefinition] = {
    # LLM-based actions
    "REASONING": ActionDefinition(
        action_type="REASONING",
        kind="llm",
        prompt=REASONING_PROMPT,
        description="Analyze and reason about the current context",
    ),
    "CONTEXT_REFERENCE": ActionDefinition(
        action_type="CONTEXT_REFERENCE",
        kind="llm",
        prompt=CONTEXT_REFERENCE_PROMPT,
        description="Extract relevant context from conversation",
    ),
    "RESPONSE_GENERATION": ActionDefinition(
        action_type="RESPONSE_GENERATION",
        kind="llm",
        prompt=RESPONSE_GENERATION_PROMPT,
        description="Generate final response for user",
    ),
    
    # Tool-based actions
    "SEARCH_TAVILY": ActionDefinition(
        action_type="SEARCH_TAVILY",
        kind="tool",
        tool=tavily_search,
        description="Search the web using Tavily",
    ),
    "SEARCH_WIKIPEDIA": ActionDefinition(
        action_type="SEARCH_WIKIPEDIA",
        kind="tool",
        tool=wikipedia_search,
        description="Search Wikipedia for information",
    ),
    "SEARCH_DOCUMENT": ActionDefinition(
        action_type="SEARCH_DOCUMENT",
        kind="tool",
        tool=search_document,
        description="Search internal documents (RAG)",
    ),
}


def get_action(action_type: str) -> Optional[ActionDefinition]:
    """Get action definition by type."""
    return ACTION_REGISTRY.get(action_type)


def get_all_actions() -> Dict[str, ActionDefinition]:
    """Get all registered actions."""
    return ACTION_REGISTRY


def list_actions_by_kind(kind: str) -> list:
    """List all actions of a specific kind ('llm' or 'tool')."""
    return [
        action for action in ACTION_REGISTRY.values() 
        if action.kind == kind
    ]


def get_tools_for_planner() -> list:
    """Get tool-based actions for planner prompt (schema generation)."""
    return [
        action.tool for action in ACTION_REGISTRY.values()
        if action.kind == "tool" and action.tool is not None
    ]
