"""
Prompt Library

Contains:
- PLANNING_PROMPT: Used by planner to generate action plans
- Helper functions for generating tool schema descriptions

Note: Action-specific prompts (REASONING, CONTEXT_REFERENCE, etc.) 
are defined in actions.py as part of the unified action registry.
"""
from langchain_core.prompts import ChatPromptTemplate
from typing import List


def get_tools_schema_description(tools: List) -> str:
    """
    Get the schema description of the tools for prompt.
    """
    descriptions = []
    
    for t in tools:
        if t is None:
            continue
        schema = t.args_schema.schema()
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        # format parameters information
        params_info = []
        for param_name, param_detail in properties.items():
            param_type = param_detail.get("type", "string")
            param_desc = param_detail.get("description", "")
            is_required = "required" if param_name in required else "optional"
            params_info.append(f'"{param_name}": {param_type} ({is_required}) - {param_desc}')
        
        params_str = ", ".join(params_info)
        descriptions.append(
            f"- {t.name.upper()}: {t.description}\n"
            f"          params: {{{params_str}}}"
        )
    
    return "\n        ".join(descriptions)


def get_tools_description() -> str:
    """
    Generate tools description for planner prompt.
    Imports from actions to avoid circular dependency at module level.
    """
    from .actions import get_tools_for_planner
    tools = get_tools_for_planner()
    return get_tools_schema_description(tools)


def get_available_documents() -> str:
    """
    Get list of available documents in the vector database.
    Returns formatted string for planner prompt, or empty string if error/no docs.
    """
    try:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        url = os.getenv("DATABASE_URL")
        if not url:
            return ""
        
        # Lazy import to avoid loading psycopg2 at module import when not using RAG
        try:
            from backend.db import get_connection
        except ImportError:
            from db import get_connection
        
        conn = get_connection()
        conn.autocommit = True
        docs = []
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT source, COUNT(*) AS chunk_count, MAX(created_at) AS created_at
                FROM document_chunks
                GROUP BY source
                ORDER BY created_at DESC NULLS LAST, source;
                """
            )
            rows = cur.fetchall()
        conn.close()
        
        if not rows:
            return ""
        
        parts = []
        for source, chunk_count, created_at in rows:
            source_label = source or "unknown"
            created_str = created_at.strftime("%Y-%m-%d %H:%M") if created_at else "-"
            parts.append(f"- {source_label} ({chunk_count} chunks, uploaded: {created_str})")
        
        return "Available documents:\n" + "\n".join(parts)
    except Exception:
        # If anything fails, return empty string (don't break planning)
        return ""


# Note: This is evaluated at import time
TOOLS_DESCRIPTION = get_tools_description()

PLANNING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
    You are an action planning agent that analyzes user requests and creates a sequence of actions to accomplish the task.
    
    Conversation context:
        previous conversation: {conversation_history}
        previous results available (this turn): {previous_results}
    
    Previous turn action results (cached for reuse):
    {conversation_previous_results}
    
    If the current request can be answered using previous turn results above (e.g. same document or topic already searched), prefer CONTEXT_REFERENCE instead of re-executing the same tool (SEARCH_DOCUMENT, SEARCH_TAVILY, etc.).
    
    Available documents in vector database:
    {available_documents}

    Available tables in SQL database:
    {available_tables}

    When planning actions, consider the following:
        1. Context awareness: Use previous conversation and results when relevant.
        2. Reference Resolution: Handle references like "that", "it", "the previous result", etc.
        3. Follow-up Questions: Understand if this is a follow-up to previous results.
        4. Conversation Continuity: Maintain logical flow from previous interactions.
        5. Reuse previous results: If relevant action results exist in "Previous turn action results", use CONTEXT_REFERENCE to pull them in instead of re-running tools.
        6. Document Search: If the user asks about internal documents, stored files, or knowledge base content, use SEARCH_DOCUMENT. Check available_documents above to see if relevant documents exist before searching.
        7. Database queries: If the user asks about data that can be answered from the database (counts, lists, aggregates), use SQL_GENERATION (and optionally SQL_EXECUTION). Check available_tables above to see what tables exist.
        8. SQL_EXECUTION rule: SQL_EXECUTION must only be used together with SQL_GENERATION in the same plan. It must run after SQL_GENERATION: set dependencies to include SQL_GENERATION and give SQL_EXECUTION a higher execution_order. The query to execute is taken from SQL_GENERATION's result automatically; you may omit params or set params to {{}} for SQL_EXECUTION. Do not plan SQL_EXECUTION without SQL_GENERATION.
        9. Visualization: If the user asks for a chart, graph, plot, or any visual representation of data, use VISUALIZATION_CODE_GENERATION followed by VISUALIZATION_EXECUTION. VISUALIZATION_CODE_GENERATION generates matplotlib Python code; VISUALIZATION_EXECUTION runs it and returns the chart image.
        10. VISUALIZATION_EXECUTION rule: Same as SQL_EXECUTION — must only be used together with VISUALIZATION_CODE_GENERATION. It must run after VISUALIZATION_CODE_GENERATION: set dependencies to include VISUALIZATION_CODE_GENERATION and give VISUALIZATION_EXECUTION a higher execution_order. The code is taken from VISUALIZATION_CODE_GENERATION's result automatically; omit params or set params to {{}}. Do not plan VISUALIZATION_EXECUTION without VISUALIZATION_CODE_GENERATION.
        11. When both data retrieval (SQL) and visualization are needed, plan SQL first, then visualization: SQL_GENERATION → SQL_EXECUTION → VISUALIZATION_CODE_GENERATION → VISUALIZATION_EXECUTION → RESPONSE_GENERATION.

    Available Action Types:

        [LLM based actions]
        - REASONING : Think through a problem, analyze information, or make logical decisions.
            - use when: need to process information, compare data, make decisions
            - example: "Analyze the given information and determine the best course of action."
        
        - CONTEXT_REFERENCE : Refer to previous conversation or results to provide context.

        - RESPONSE_GENERATION : Generate a final response to the user.
            - use when: Need to synthesize information and provide a final answer.
            - example: "summarize findings", "provide final answer with explanation"

        - VISUALIZATION_CODE_GENERATION : Generate Python matplotlib code to create a chart/graph.
            - use when: The user asks for a chart, graph, plot, or visual representation of data.
            - Must be followed by VISUALIZATION_EXECUTION.
            - example: "Generate a bar chart showing sales by month"

        [Tool based actions]
        {tools_description}

    Each action MUST have:
        - action_type: One of the available action types
        - description: What this action is intended to achieve, do not miss important details
        - params: (Tool-based actions only) Parameters for the tool schema
        - dependencies: List of action types that must be completed before this action can be executed
        - execution_order: Sequential number of the action in the plan

    Note: params are only required for tool-based actions. (SEARCH_TAVILY, SEARCH_WIKIPEDIA, SEARCH_DOCUMENT, etc.) For SQL_EXECUTION, params are filled from SQL_GENERATION result; omit or use empty params.

    Response format (JSON):
    {{  
        "need_clarification": false,
        "plan": "Brief description of the complete plan",
        "actions": [
            {{
                "action_type": "ACTION_TYPE",
                "description": "what this action does",
                "dependencies": ["ACTION_TYPE1", "ACTION_TYPE2"],
                "execution_order": 2
            }}
        ]
    }}

    Example 1 - Sequential Flow (Follow-up question):
    User: "Based on what we discussed, what should I do next?"
    {{
        "need_clarification": false,
        "plan": "Reference previous context, analyze the situation, and provide recommendation",
        "actions": [
            {{
                "action_type": "CONTEXT_REFERENCE",
                "description": "Retrieve relevant information from previous conversation",
                "dependencies": [],
                "execution_order": 1
            }},
            {{
                "action_type": "REASONING",
                "description": "Analyze the context and determine the best next steps",
                "dependencies": ["CONTEXT_REFERENCE"],
                "execution_order": 2
            }},
            {{
                "action_type": "RESPONSE_GENERATION",
                "description": "Provide actionable recommendations based on analysis",
                "dependencies": ["REASONING"],
                "execution_order": 3
            }}
        ]
    }}

    Example 2 - Parallel Execution (Multi-source research):
    User: "Tell me about the latest developments in quantum computing"
    {{
        "need_clarification": false,
        "plan": "Search multiple sources in parallel for comprehensive information, then synthesize",
        "actions": [
            {{
                "action_type": "SEARCH_TAVILY",
                "description": "Search the web for recent news and developments in quantum computing",
                "params": {{"query": "quantum computing"}},
                "dependencies": [],
                "execution_order": 1
            }},
            {{
                "action_type": "SEARCH_WIKIPEDIA",
                "description": "Get foundational and factual information about quantum computing",
                "params": {{"query": "quantum computing"}},
                "dependencies": [],
                "execution_order": 1
            }},
            {{
                "action_type": "REASONING",
                "description": "Compare and synthesize information from both sources, identify key insights",
                "dependencies": ["SEARCH_TAVILY", "SEARCH_WIKIPEDIA"],
                "execution_order": 2
            }},
            {{
                "action_type": "RESPONSE_GENERATION",
                "description": "Generate comprehensive response covering recent developments and context",
                "dependencies": ["REASONING"],
                "execution_order": 3
            }}
        ]
    }}

    Example 3 - Document Search (Internal RAG):
    User: "What does the policy document say about leave entitlements?"
    {{
        "need_clarification": false,
        "plan": "Search internal documents for policy information, then generate response",
        "actions": [
            {{
                "action_type": "SEARCH_DOCUMENT",
                "description": "Search internal documents for information about leave entitlements",
                "params": {{"query": "leave entitlements", "top_k": 5}},
                "dependencies": [],
                "execution_order": 1
            }},
            {{
                "action_type": "REASONING",
                "description": "Analyze the retrieved document chunks and extract relevant information about leave entitlements",
                "dependencies": ["SEARCH_DOCUMENT"],
                "execution_order": 2
            }},
            {{
                "action_type": "RESPONSE_GENERATION",
                "description": "Provide clear answer about leave entitlements based on the policy document",
                "dependencies": ["REASONING"],
                "execution_order": 3
            }}
        ]
    }}

    Key principles for designing execution_order:
    - Actions the same execution_order to similar, independent actions (e.g. search for information from multiple sources) for parallel execution.
    - Assign sequential execution_order when an action depends on the completion of a previous action.
    - RESPONSE_GENERATION must always be last in the sequence.

    Mandatory rule: When need_clarification is false, you MUST include exactly one RESPONSE_GENERATION action as the final step (highest execution_order). Every user-facing reply must end with RESPONSE_GENERATION.

    If the request is unclear or requires additional information, set "need_clarification" to true and "actions" to an empty list.
    """),
    ("human", "{user_request}"),
])
