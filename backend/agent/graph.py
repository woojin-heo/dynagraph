import re
import time
from functools import partial
from itertools import groupby
from typing import Dict, Any, List, Optional, Generator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState
from .actions import get_action
from .runtime import LLM, get_llm_for_action
from .trace import record


def _extract_sql_query(text: str) -> str:
    """Extract executable SQL from SQL_GENERATION output (strip markdown, code fences, leading/trailing junk)."""
    if not text or not text.strip():
        return ""
    s = text.strip()
    # Remove ```sql ... ``` or ``` ... ```
    m = re.search(r"```(?:\w*)\s*([\s\S]*?)```", s)
    if m:
        s = m.group(1).strip()
    return s.strip()


try:
    from backend.db import get_schema_for_prompt
except ImportError:
    from db import get_schema_for_prompt


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
    overrides = state.get('human_param_overrides', {}).get(action_type, {})
    params = {**params, **overrides}

    conversation_history = state.get('messages', [])[-10:]
    previous_results = state.get('previous_results', {})
    if action_type == 'SQL_EXECUTION':
        override_query = overrides.get('query', '').strip()
        if override_query:
            raw = override_query
        else:
            raw = (previous_results.get('SQL_GENERATION') or params.get('query') or '').strip()
        query = _extract_sql_query(raw)
        params = {**params, 'query': query}
    user_request = conversation_history[-1].content if conversation_history else ''

    action_def = get_action(action_type)
    kind = action_def.kind if action_def else "unknown"

    record("action_start",
           action_type=action_type, kind=kind,
           description=description, params=params,
           has_overrides=bool(overrides))

    t0 = time.time()

    if action_def is None:
        result = f"Action '{action_type}' not found in ACTION_REGISTRY"
    elif action_def.kind == "llm":
        if action_def.prompt is None:
            result = f"Action '{action_type}' is LLM-based but has no prompt defined"
        else:
            invoke_vars = {
                "conversation_history": conversation_history,
                "previous_results": previous_results,
                "user_request": user_request,
                "description": description,
            }
            if action_type == "SQL_GENERATION":
                try:
                    invoke_vars["db_schema"] = get_schema_for_prompt()
                except Exception:
                    invoke_vars["db_schema"] = ""

            record("llm_call",
                   action_type=action_type,
                   prompt_vars={k: v for k, v in invoke_vars.items() if k != "conversation_history"},
                   message_count=len(conversation_history))

            effective_llm = get_llm_for_action(action_type)
            chain = action_def.prompt | effective_llm
            llm_result = chain.invoke(invoke_vars)
            result = llm_result.content

            record("llm_response",
                   action_type=action_type,
                   response=result)

            if action_type == "SQL_GENERATION":
                result = _extract_sql_query(result) or result
    elif action_def.kind == "tool":
        if action_def.tool is None:
            result = f"Action '{action_type}' is tool-based but tool not implemented"
        else:
            record("tool_call",
                   action_type=action_type,
                   tool_name=action_def.tool.name if hasattr(action_def.tool, 'name') else action_type,
                   params=params)

            result = action_def.tool.invoke(params)

            record("tool_response",
                   action_type=action_type,
                   result=str(result))
    else:
        result = f"Action '{action_type}' has unknown kind: {action_def.kind}"

    duration_ms = round((time.time() - t0) * 1000, 1)
    record("action_end",
           action_type=action_type, kind=kind,
           duration_ms=duration_ms,
           result=str(result))

    updated_results = {**previous_results, action_type: result}
    
    return {
        "previous_results": updated_results,
    }


def create_execution_graph(
    actions: List[Dict[str, Any]], 
    enable_hitl: bool = False,
    hitl_before: Optional[List[str]] = None,
) -> tuple:
    """
    Create an execution graph from the planned actions.
    
    Supports parallel execution: actions with the same execution_order
    will be executed in parallel (fan-out), then merged (fan-in).
    
    Example:
        execution_order=1: [SEARCH_TAVILY, SEARCH_WIKIPEDIA]  <- parallel
        execution_order=2: [REASONING]
        execution_order=3: [RESPONSE_GENERATION]
        
        Creates graph:
                 ┌─→ SEARCH_TAVILY ──┐
        START ──→│                   │──→ REASONING ──→ RESPONSE_GENERATION ──→ END
                 └─→ SEARCH_WIKIPEDIA─┘
    
    Args:
        actions: A list of actions from the planner.
        enable_hitl: Whether to enable human in the loop interrupts.
        hitl_before: List of action types to pause BEFORE execution (for human review).
    
    Returns:
        Tuple of (compiled_graph, checkpointer).
    """
    graph = StateGraph(AgentState)
    
    if not actions:
        return None, None

    # Enforce: SQL_EXECUTION only after SQL_GENERATION (drop SQL_EXECUTION if no SQL_GENERATION in plan)
    action_types = {a.get('action_type') for a in actions}
    if 'SQL_EXECUTION' in action_types and 'SQL_GENERATION' not in action_types:
        actions = [a for a in actions if a.get('action_type') != 'SQL_EXECUTION']
    if not actions:
        return None, None

    # Sort actions by execution_order
    sorted_actions = sorted(actions, key=lambda x: x.get('execution_order', 0))

    # Add nodes for each action
    for action in sorted_actions:
        node_name = action.get('action_type', 'unknown')
        graph.add_node(node_name, partial(action_executor, action))

    # Group actions by execution_order for parallel execution
    execution_groups = []
    for order, group in groupby(sorted_actions, key=lambda x: x.get('execution_order', 0)):
        group_nodes = [a.get('action_type', 'unknown') for a in group]
        execution_groups.append(group_nodes)
    
    # Build edges between groups
    for i, current_group in enumerate(execution_groups):
        if i == 0:
            # START -> all nodes in first group (fan-out if multiple)
            for node in current_group:
                graph.add_edge(START, node)
        
        if i < len(execution_groups) - 1:
            # Current group -> next group (fan-in / fan-out)
            next_group = execution_groups[i + 1]
            for curr_node in current_group:
                for next_node in next_group:
                    graph.add_edge(curr_node, next_node)
        else:
            # Last group -> END
            for node in current_group:
                graph.add_edge(node, END)

    # Always use checkpointer for state tracking and debugging
    checkpointer = MemorySaver()

    node_names_in_graph = {a.get("action_type", "unknown") for a in sorted_actions}
    if enable_hitl and hitl_before:
        # Only interrupt before nodes that exist in this run's graph (e.g. CONTEXT_REFERENCE-only turn has no SEARCH_DOCUMENT)
        interrupt_nodes = [n for n in hitl_before if n in node_names_in_graph]
        if interrupt_nodes:
            compiled = graph.compile(
                checkpointer=checkpointer,
                interrupt_before=interrupt_nodes,
            )
        else:
            compiled = graph.compile(checkpointer=checkpointer)
    else:
        compiled = graph.compile(checkpointer=checkpointer)

    return compiled, checkpointer


def get_graph_mermaid(
    actions: List[Dict[str, Any]],
    hitl_before: Optional[List[str]] = None,
) -> Optional[str]:
    """
    Build the same execution graph as create_execution_graph and return LangGraph's
    Mermaid diagram string (get_graph().draw_mermaid()) if the API is available.
    Returns None if the graph has no actions or if draw_mermaid is not available.
    """
    if not actions:
        return None
    try:
        compiled, _ = create_execution_graph(
            actions,
            enable_hitl=bool(hitl_before),
            hitl_before=hitl_before or [],
        )
        if compiled is None:
            return None
        g = compiled.get_graph()
        if g is not None and hasattr(g, "draw_mermaid"):
            return g.draw_mermaid()
    except Exception:
        pass
    return None


def stream_execution(
    graph, 
    state: AgentState, 
    thread_id: str = "default"
) -> Generator[Dict[str, Any], None, None]:
    """
    Stream execution results for each node.
    
    Args:
        graph: Compiled execution graph.
        state: Initial state.
        thread_id: Thread ID for checkpointing.
    
    Yields:
        Dict with node_name and result for each step.
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    for step in graph.stream(state, config):
        # step is like {"NODE_NAME": {"previous_results": {...}}}
        for node_name, node_output in step.items():
            yield {
                "node": node_name,
                "output": node_output,
                "status": "running",
            }
    
    yield {"status": "complete"}


def resume_execution(
    graph, 
    thread_id: str,
    human_input: Optional[Dict[str, Any]] = None
) -> Generator[Dict[str, Any], None, None]:
    """
    Resume execution after human review.
    
    Args:
        graph: Compiled execution graph (with checkpointer).
        thread_id: Thread ID to resume.
    human_input: Optional modifications from human review.
        Use {"param_overrides": {"ACTION_TYPE": {"param_key": "value"}}} to override
        params for pending nodes (e.g. {"param_overrides": {"SEARCH_TAVILY": {"query": "new query"}}}).

    Yields:
        Dict with node_name and result for each step.
    """
    config = {"configurable": {"thread_id": thread_id}}

    if human_input:
        param_overrides = human_input.get("param_overrides", human_input)
        if param_overrides:
            graph.update_state(config, {"human_param_overrides": param_overrides})

    # Resume with None to continue from checkpoint
    for step in graph.stream(None, config):
        for node_name, node_output in step.items():
            yield {
                "node": node_name,
                "output": node_output,
                "status": "running",
            }
    
    yield {"status": "complete"}


def get_current_state(graph, thread_id: str) -> Optional[AgentState]:
    """
    Get the current state from a paused graph.
    
    Args:
        graph: Compiled execution graph (with checkpointer).
        thread_id: Thread ID.
    
    Returns:
        Current state or None if not found.
    """
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    if snapshot:
        return snapshot.values
    return None
