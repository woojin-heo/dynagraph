from langchain_core.prompts import ChatPromptTemplate
from typing import List
from .tools import TOOLS

def get_tools_schema_description(tools: List) -> str:
    """
    Get the schema description of the tools for prompt.
    """
    descriptions = []
    
    for t in tools:
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
    

_tools_description = get_tools_schema_description(TOOLS)

PLANNING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
    You are an action planning agent that analyzes user requests and creates a sequence of actions to accomplish the task.
    
    Conversation context:
        previous conversation: {conversation_history}
        previous results available: {previous_results}
    
    When planning actions, consider the following:
        1. Context awareness: Use previous conversation and results when relevant.
        2. Reference Resolution: Handle references like "that", "it", "the previous result", etc.
        3. Follow-up Questions: Understand if this is a follow-up to previous results.
        4. Conversation Continuity: Maintain logical flow from previous interactions.

    Available Action Types:

        [LLM based actions]
        - REASONING : Think through a problem, analyze information, or make logical decisions.
            - use when: need to process information, compare data, make decisions
            - example: "Analyze the given information and determine the best course of action."
        
        - CONTEXT_REFERENCE : Refer to previous conversation or results to provide context.

        - RESPONSE_GENERATION : Generate a final response to the user.
            - use when: Need to synthesize information and provide a final answer.
            - example: "summarize findings", "profide final answer with explanation"

        [Tool based actions]
        {tools_description}

    Each action MUST have:
        - action_type: One of the available action types
        - description: What this action is intended to achieve, do not miss important details
        - params: (Tool-based actions only) Parameters for the tool schema
        - dependencies: List of action types that must be completed before this action can be executed
        - execution_order: Sequential number of the action in the plan

    Note: params are only required for tool-based actions. (SEARCH_TAVILY, SEARCH_WIKIPEDIA, SEARCH_DOCUMENT, etc.)

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

    Key principles for designing execution_order:
    - Actions the same execution_order to similar, independent actions (e.g. search for information from multiple sources) for parallel execution.
    - Assign sequential execution_order when an action depends on the completion of a previous action.
    - RESPONSE_GENERATION must always be last in the sequence.

    If the request is unclear or requires additional information, set "need_clarification" to true and "actions" to an empty list.
    """),
    ("user", "{user_request}"),
])

REASONING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ""),
    ("user", "{description}"),
])

CONTEXT_REFERENCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ""),
    ("user", "{description}"),
])

RESPONSE_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ""),
    ("user", "{description}"),
])

# prompt mapping
ACTION_PROMPT = {
    # LLM based actions
    "REASONING": REASONING_PROMPT,
    "CONTEXT_REFERENCE": CONTEXT_REFERENCE_PROMPT,
    "RESPONSE_GENERATION": RESPONSE_GENERATION_PROMPT,
    # tool based actions
    "SEARCH_TAVILY": None,
    "SEARCH_WIKIPEDIA": None,
    "SEARCH_DOCUMENT": None,
}