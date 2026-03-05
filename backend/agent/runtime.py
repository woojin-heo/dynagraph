import re
import time
from typing import Dict, Any, Optional, Generator, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from dotenv import load_dotenv
import uuid

load_dotenv()

from .trace import TraceCollector, set_current_trace, reset_current_trace


def _sources_from_document_tags(text: str) -> List[str]:
    """Extract unique source values from <Document source="..."> tags (URLs for Tavily, paths for RAG)."""
    if not text or "Error" in text or "No matching" in text:
        return []
    names = re.findall(r'source="([^"]+)"', text)
    return list(dict.fromkeys(names))  # preserve order, dedupe


def _collect_references(previous_results: Dict[str, Any]) -> List[str]:
    """Collect source URLs/names from SEARCH_DOCUMENT, SEARCH_TAVILY, SEARCH_WIKIPEDIA (exact values from tags)."""
    sources: List[str] = []
    for key in ("SEARCH_DOCUMENT", "SEARCH_TAVILY", "SEARCH_WIKIPEDIA"):
        text = previous_results.get(key) or ""
        sources.extend(_sources_from_document_tags(text))
    return list(dict.fromkeys(sources))  # preserve order, dedupe


def _format_references(sources: List[str]) -> str:
    """Format sources as a markdown references block. URLs become clickable links."""
    items = []
    for s in sources:
        if s.startswith(("http://", "https://")):
            items.append(f"[{s}]({s})")
        else:
            items.append(s)
    return "\n\nreferences:\n" + "\n".join(f"- {item}" for item in items)

DEFAULT_MODEL = "gpt-4o-mini"
LLM = ChatOpenAI(model=DEFAULT_MODEL, temperature=0)

ACTION_LLM_OVERRIDES: dict[str, dict] = {
    "SQL_GENERATION": {"model": "gpt-4o", "temperature": 0},
}


def get_llm_for_action(action_type: str) -> ChatOpenAI:
    """Return the LLM for a given action: override if configured, else default."""
    cfg = ACTION_LLM_OVERRIDES.get(action_type)
    if cfg:
        return ChatOpenAI(**cfg)
    return LLM

# These imports come after LLM definition to avoid circular import issues
from .state import AgentState
from .planner import planning_agent
from .graph import (
    create_execution_graph,
    stream_execution,
    resume_execution,
    get_current_state,
)
from . import hitl


class ConversationAgent:
    """
    Agent that maintains conversation history and caches previous action results
    across turns so the planner can reuse them (e.g. CONTEXT_REFERENCE instead of re-running SEARCH_DOCUMENT).
    Supports optional HITL (human-in-the-loop) via resume().
    """
    
    def __init__(self, enable_hitl: bool = False, hitl_before: Optional[List[str]] = None,
                 conversation_id: Optional[str] = None, tenant_id: Optional[str] = None):
        self.tenant_id = tenant_id
        self.messages: List[BaseMessage] = []
        self.conversation_previous_results: List[Dict[str, Any]] = []
        self.enable_hitl = enable_hitl
        self.hitl_before = hitl_before or []
        self.current_graph = None
        self._pending_user_message: Optional[str] = None
        self._current_thread_id: Optional[str] = None
        self._turn_number: int = 0
        self.trace = TraceCollector(conversation_id or str(uuid.uuid4()))

    @property
    def thread_id(self) -> Optional[str]:
        """Thread ID for the current execution run (None when no run is active)."""
        return self._current_thread_id

    def get_conversation_history(self) -> List[BaseMessage]:
        """Return current conversation history."""
        return list(self.messages)

    def get_paused_payload(self) -> Optional[Dict[str, Any]]:
        """Return the HITL paused payload if the agent is currently paused, else None."""
        if not self._pending_user_message or not self.current_graph or not self._current_thread_id:
            return None
        current_state = get_current_state(self.current_graph, self._current_thread_id)
        if not current_state:
            return None
        return hitl.build_paused_payload(
            self.current_graph,
            self._current_thread_id,
            current_state,
            self._pending_user_message,
            self.hitl_before,
        )
    
    def clear_history(self) -> None:
        """Clear conversation history and cached results."""
        self.messages = []
        self.conversation_previous_results = []
        self._turn_number = 0

    def _process_execution_steps(
        self,
        step_iterator,
        current_state: Dict[str, Any],
        user_message: str,
        current_turn: int,
    ) -> Generator[Dict[str, Any], None, None]:
        """Delegate to hitl.process_execution_steps (shared by run and resume)."""
        def set_pending(msg: str) -> None:
            self._pending_user_message = msg

        yield from hitl.process_execution_steps(
            step_iterator,
            current_state,
            user_message,
            get_current_state_fn=get_current_state,
            graph=self.current_graph,
            thread_id=self._current_thread_id,
            hitl_before=self.hitl_before,
            set_pending_message=set_pending,
            collect_sources_fn=_collect_references,
            format_references_fn=_format_references,
        )

    def _append_turn_results(
        self, current_state: Dict[str, Any], user_message: str, current_turn: int
    ) -> None:
        """Append this turn's results and messages (on true completion)."""
        prev = current_state.get("previous_results", {})
        actions = current_state.get("actions", [])
        overrides = current_state.get("human_param_overrides", {})
        actions_with_results = []
        for action in actions:
            action_type = action.get("action_type", "")
            params = action.get("params", {})
            merged_params = {**params, **overrides.get(action_type, {})}
            # SQL_EXECUTION receives query from SQL_GENERATION; store it so conversation_previous_results shows actual params
            if action_type == "SQL_EXECUTION" and prev.get("SQL_GENERATION"):
                merged_params = {**merged_params, "query": (prev.get("SQL_GENERATION") or "").strip()}
            actions_with_results.append({
                **action,
                "params": merged_params,
                "result": prev.get(action_type, ""),
            })
        self.conversation_previous_results.append({
            "turn": current_turn,
            "plan": current_state.get("plan", ""),
            "need_clarification": current_state.get("need_clarification", False),
            "actions": actions_with_results,
        })
        result_text = prev.get("RESPONSE_GENERATION", "")
        sources = _collect_references(prev)
        if sources and result_text and "references:" not in result_text:
            result_text = result_text.rstrip() + _format_references(sources)
        self.messages = self.messages + [
            HumanMessage(content=user_message),
            AIMessage(content=result_text),
        ]

    def run(self, user_message: str) -> Generator[Dict[str, Any], None, None]:
        """
        Run a new turn: plan, execute, then merge this turn's results into
        conversation_previous_results and append messages.
        """
        self._turn_number += 1
        current_turn = self._turn_number
        turn_t0 = time.time()

        token = set_current_trace(self.trace)
        try:
            self.trace.record("turn_start", turn=current_turn, user_message=user_message)

            current_state: Dict[str, Any] = {
                "messages": self.messages + [HumanMessage(content=user_message)],
                "previous_results": {},
                "need_clarification": False,
                "plan": "",
                "actions": [],
            }

            yield {"phase": "planning", "status": "running"}
            
            planner_result = planning_agent(
                current_state,
                conversation_previous_results=self.conversation_previous_results,
            )
            current_state.update(planner_result)
            
            yield {
                "phase": "planning",
                "status": "complete",
                "plan": current_state.get("plan", ""),
                "actions": current_state.get("actions", []),
            }
            
            if current_state.get("need_clarification"):
                self.trace.record("turn_complete", turn=current_turn,
                                  duration_ms=round((time.time() - turn_t0) * 1000, 1),
                                  outcome="clarification")
                yield {
                    "phase": "clarification",
                    "status": "waiting",
                    "message": "Clarification needed from user",
                }
                return
            
            actions = current_state.get("actions", [])
            if not actions:
                self.trace.record("turn_complete", turn=current_turn,
                                  duration_ms=round((time.time() - turn_t0) * 1000, 1),
                                  outcome="no_actions")
                yield {
                    "phase": "execution",
                    "status": "complete",
                    "message": "No actions to execute",
                }
                return
            
            graph, _ = create_execution_graph(
                actions,
                enable_hitl=self.enable_hitl,
                hitl_before=self.hitl_before,
            )
            if not graph:
                self.trace.record("error", turn=current_turn, error="Failed to create graph")
                yield {"phase": "execution", "status": "error", "message": "Failed to create graph"}
                return
            
            self.current_graph = graph
            self._current_thread_id = str(uuid.uuid4())
            yield {"phase": "execution", "status": "running"}

            for event in self._process_execution_steps(
                stream_execution(graph, current_state, self._current_thread_id),
                current_state,
                user_message,
                current_turn,
            ):
                if event.get("status") == "complete":
                    self._append_turn_results(current_state, user_message, current_turn)
                    self.trace.record("turn_complete", turn=current_turn,
                                      duration_ms=round((time.time() - turn_t0) * 1000, 1),
                                      outcome="complete",
                                      result=event.get("result", ""))
                    yield event
                    return
                if event.get("status") == "paused":
                    self.trace.record("turn_paused", turn=current_turn,
                                      duration_ms=round((time.time() - turn_t0) * 1000, 1))
                    yield event
                    return
                yield event
        except Exception as exc:
            self.trace.record("error", turn=current_turn, error=str(exc))
            raise
        finally:
            reset_current_trace(token)

    def resume(self, human_feedback: Optional[Dict[str, Any]] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Resume execution after human review (HITL mode).
        Call after run() yielded status="paused". May yield status="paused" again if
        another interrupt_before node is hit; call resume() again to continue.
        """
        if not self.current_graph or not self.thread_id:
            yield {"status": "error", "message": "No paused execution to resume"}
            return
        if self._pending_user_message is None:
            yield {"status": "error", "message": "No pending user message"}
            return

        token = set_current_trace(self.trace)
        resume_t0 = time.time()
        try:
            self.trace.record("resume_start", turn=self._turn_number,
                              human_feedback=human_feedback)

            user_message = self._pending_user_message
            current_state = get_current_state(self.current_graph, self._current_thread_id)
            if not current_state:
                yield {"status": "error", "message": "Could not load state for resume"}
                return

            yield {"phase": "execution", "status": "resuming"}

            for event in self._process_execution_steps(
                resume_execution(self.current_graph, self._current_thread_id, human_feedback),
                current_state,
                user_message,
                self._turn_number,
            ):
                if event.get("status") == "complete":
                    self._append_turn_results(current_state, user_message, self._turn_number)
                    self._pending_user_message = None
                    self.trace.record("turn_complete", turn=self._turn_number,
                                      duration_ms=round((time.time() - resume_t0) * 1000, 1),
                                      outcome="complete",
                                      result=event.get("result", ""))
                    yield event
                    return
                if event.get("status") == "paused":
                    self.trace.record("turn_paused", turn=self._turn_number,
                                      duration_ms=round((time.time() - resume_t0) * 1000, 1))
                    yield event
                    return
                yield event
        except Exception as exc:
            self.trace.record("error", turn=self._turn_number, error=str(exc))
            raise
        finally:
            reset_current_trace(token)


def run_agent(user_message: str) -> Dict[str, Any]:
    """
    Run the agent for one turn and return final result (non-streaming).
    Uses ConversationAgent internally; each call is a fresh conversation.
    """
    agent = ConversationAgent(enable_hitl=False)
    result = None
    for step in agent.run(user_message):
        result = step
    return result


def run_agent_with_hitl(
    user_message: str, 
    hitl_before: Optional[List[str]] = None,
    human_feedback: Optional[Dict[str, Any]] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Run the agent with HITL enabled. Yields each step (planning, execution nodes, paused, complete).
    The last yielded value is the final result (phase="execution", status="complete", result="...").
    When paused, waits for input(); type 'resume' or 'r' to continue.
    """
    if hitl_before is None:
        hitl_before = ["RESPONSE_GENERATION"]
    
    agent = ConversationAgent(enable_hitl=True, hitl_before=hitl_before)
    first_run = True
    
    while True:
        inner_gen = agent.run(user_message) if first_run else agent.resume(human_feedback)
        first_run = False
        completed = False
        for step in inner_gen:
            yield step
            if step.get("phase") == "execution" and step.get("status") == "complete":
                completed = True
                return
        
        if not completed and agent.current_graph:
            try:
                graph_state = agent.current_graph.get_state(
                    {"configurable": {"thread_id": agent.thread_id}}
                )
                if graph_state and graph_state.tasks:
                    yield {
                        "phase": "execution",
                        "status": "paused",
                        "message": "Execution paused for human review. Type 'resume' or 'r' to continue.",
                    }
                    print("\n[Paused] Execution paused for human review. Type 'resume' or 'r' to continue.")
                    while True:
                        user_input = input("[Paused] Enter 'resume' or 'r' to continue: ").strip().lower()
                        if user_input in ['resume', 'r']:
                            break
                        print("Invalid input. Please type 'resume' or 'r'.")
                    continue
            except Exception:
                pass
        return
