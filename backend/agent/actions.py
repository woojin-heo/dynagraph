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

from .tools import search_tavily, search_wikipedia, search_document, sql_execution, visualization_execution


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

References rule: Do NOT add a "references:" line yourself. References (exact URLs or source names from document tags) are appended automatically when SEARCH_DOCUMENT, SEARCH_TAVILY, or SEARCH_WIKIPEDIA was used in this turn."""),
    ("human", "{description}"),
])

SQL_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a SQL generation agent. Generate a PostgreSQL-compatible SQL query.

Below are the exact CREATE TABLE definitions of every table and column available in the database.
ONLY these tables and columns exist — do NOT invent, guess, or assume any column that is not listed here.

{db_schema}

Previous results from other actions:
{previous_results}

Rules:
1. Use ONLY the table and column names that appear in the CREATE TABLE statements above.
2. If a column you want does not exist in the schema, do NOT include it — adapt the query to use what is available.
3. Output a single executable PostgreSQL SQL statement. No markdown, no code fences, no explanation."""),
    ("human", "{description}"),
])

VISUALIZATION_CODE_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a Python visualization code generation agent.
Generate matplotlib code that creates a chart/graph from the data in previous results.

Previous results from other actions:
{previous_results}

Conversation history:
{conversation_history}

Rules:
1. matplotlib.pyplot is pre-imported as `plt`. pandas as `pd`, numpy as `np` are also available.
2. Parse/define the data directly in Python from the previous results shown above.
3. Set a clear title, axis labels, and legend where appropriate.
4. Use a clean, professional style (e.g. plt.style.use('seaborn-v0_8-whitegrid') or similar).
5. Use plt.figure(figsize=(10, 6)) for a good default size.
6. For Korean text in labels/title use: plt.rcParams['font.family'] = 'AppleGothic'  (or fall back to 'NanumGothic' on Linux).
7. Call plt.tight_layout() at the end.
8. Do NOT call plt.show() or plt.savefig() — the figure is captured automatically.
9. Output ONLY executable Python code. No markdown code fences, no explanation."""),
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
        tool=search_tavily,
        description="Search the web using Tavily",
    ),
    "SEARCH_WIKIPEDIA": ActionDefinition(
        action_type="SEARCH_WIKIPEDIA",
        kind="tool",
        tool=search_wikipedia,
        description="Search Wikipedia for information",
    ),
    "SEARCH_DOCUMENT": ActionDefinition(
        action_type="SEARCH_DOCUMENT",
        kind="tool",
        tool=search_document,
        description="Search internal documents (RAG)",
    ),
    "SQL_GENERATION": ActionDefinition(
        action_type="SQL_GENERATION",
        kind="llm",
        prompt=SQL_GENERATION_PROMPT,
        description="Generate SQL query to retrieve data from the database",
    ),
    "SQL_EXECUTION": ActionDefinition(
        action_type="SQL_EXECUTION",
        kind="tool",
        tool=sql_execution,
        description="Execute SQL query and return the result",
    ),
    "VISUALIZATION_CODE_GENERATION": ActionDefinition(
        action_type="VISUALIZATION_CODE_GENERATION",
        kind="llm",
        prompt=VISUALIZATION_CODE_GENERATION_PROMPT,
        description="Generate Python matplotlib code to visualize data",
    ),
    "VISUALIZATION_EXECUTION": ActionDefinition(
        action_type="VISUALIZATION_EXECUTION",
        kind="tool",
        tool=visualization_execution,
        description="Execute visualization code and return chart image",
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
