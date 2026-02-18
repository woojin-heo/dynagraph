import re
from typing import Dict, Any, Optional, Generator, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from dotenv import load_dotenv
import uuid

load_dotenv()


def _sources_from_search_document_result(text: str) -> List[str]:
    """Extract unique document source names from SEARCH_DOCUMENT output (e.g. source=\"file.pdf\")."""
    if not text or "Error" in text or "No matching" in text:
        return []
    # Match source="..." in <Document source="..." ...>
    names = re.findall(r'source="([^"]+)"', text)
    return list(dict.fromkeys(names))  # preserve order, dedupe

LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# These imports come after LLM definition to avoid circular import issues
from .state import AgentState
from .planner import planning_agent
from .graph import (
    create_execution_graph, 
    stream_execution, 
    resume_execution,
    get_current_state,
)


class ConversationAgent:
    """
    Agent that maintains conversation history and caches previous action results
    across turns so the planner can reuse them (e.g. CONTEXT_REFERENCE instead of re-running SEARCH_DOCUMENT).
    Supports optional HITL (human-in-the-loop) via resume().
    """
    
    def __init__(self, enable_hitl: bool = False, hitl_before: Optional[List[str]] = None):
        self.messages: List[BaseMessage] = []
        self.conversation_previous_results: Dict[str, Any] = {}
        self.enable_hitl = enable_hitl
        self.hitl_before = hitl_before or []
        self.current_graph = None
        self._pending_user_message: Optional[str] = None
        # Thread ID for the current execution run only (new graph per run → new thread_id per run)
        self._current_thread_id: Optional[str] = None
        self._turn_number: int = 0  # Track turn number to preserve results across turns

    @property
    def thread_id(self) -> Optional[str]:
        """Thread ID for the current execution run (None when no run is active)."""
        return self._current_thread_id

    def get_conversation_history(self) -> List[BaseMessage]:
        """Return current conversation history."""
        return list(self.messages)
    
    def clear_history(self) -> None:
        """Clear conversation history and cached results."""
        self.messages = []
        self.conversation_previous_results = {}
        self._turn_number = 0

    def run(self, user_message: str) -> Generator[Dict[str, Any], None, None]:
        """
        Run a new turn: plan, execute, then merge this turn's results into
        conversation_previous_results and append messages.
        """
        self._turn_number += 1
        current_turn = self._turn_number
        
        # State for this turn: previous messages + new user message, empty previous_results
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
            yield {
                "phase": "clarification",
                "status": "waiting",
                "message": "Clarification needed from user",
            }
            return
        
        actions = current_state.get("actions", [])
        if not actions:
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
            yield {"phase": "execution", "status": "error", "message": "Failed to create graph"}
            return
        
        self.current_graph = graph
        self._current_thread_id = str(uuid.uuid4())
        yield {"phase": "execution", "status": "running"}
        
        completed = False
        for step in stream_execution(graph, current_state, self._current_thread_id):
            if step.get("status") == "complete":
                final_state = get_current_state(graph, self._current_thread_id)
                if final_state:
                    current_state.update(final_state)
                
                prev = current_state.get("previous_results", {})
                result_text = prev.get("RESPONSE_GENERATION", "")
                search_result = prev.get("SEARCH_DOCUMENT", "")
                sources = _sources_from_search_document_result(search_result)
                if sources and result_text and "references:" not in result_text:
                    result_text = result_text.rstrip() + "\n\nreferences: " + ", ".join(sources)
                
                # Merge results preserving all turns: store as lists per action_type
                for action_type, result in prev.items():
                    if action_type not in self.conversation_previous_results:
                        self.conversation_previous_results[action_type] = []
                    # Append new result with turn number
                    self.conversation_previous_results[action_type].append({
                        "turn": current_turn,
                        "result": result
                    })
                self.messages = self.messages + [
                    HumanMessage(content=user_message),
                    AIMessage(content=result_text),
                ]
                completed = True
                yield {
                    "phase": "execution",
                    "status": "complete",
                    "result": result_text,
                    "all_results": prev,
                }
            else:
                current = get_current_state(graph, self._current_thread_id)
                if current:
                    current_state.update(current)
                yield {
                    "phase": "execution",
                    "node": step.get("node"),
                    "output": step.get("output"),
                    "status": "running",
                }
        
        # HITL: if we exited without complete, check if paused
        if not completed and self.current_graph:
            try:
                graph_state = self.current_graph.get_state(
                    {"configurable": {"thread_id": self._current_thread_id}}
                )
                if graph_state and graph_state.tasks:
                    self._pending_user_message = user_message
                    yield {
                        "phase": "execution",
                        "status": "paused",
                        "message": "Execution paused for human review. Call agent.resume() to continue.",
                    }
            except Exception:
                pass

    def resume(self, human_feedback: Optional[Dict[str, Any]] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Resume execution after human review (HITL mode).
        Call after run() yielded status="paused".
        """
        if not self.current_graph or not self.thread_id:
            yield {"status": "error", "message": "No paused execution to resume"}
            return
        if self._pending_user_message is None:
            yield {"status": "error", "message": "No pending user message"}
            return
        
        user_message = self._pending_user_message
        yield {"phase": "execution", "status": "resuming"}
        
        current_state = None
        for step in resume_execution(self.current_graph, self._current_thread_id, human_feedback):
            if step.get("status") == "complete":
                current_state = get_current_state(self.current_graph, self._current_thread_id)
                if current_state:
                    prev = current_state.get("previous_results", {})
                    result_text = prev.get("RESPONSE_GENERATION", "")
                    search_result = prev.get("SEARCH_DOCUMENT", "")
                    sources = _sources_from_search_document_result(search_result)
                    if sources and result_text and "references:" not in result_text:
                        result_text = result_text.rstrip() + "\n\nreferences: " + ", ".join(sources)
                    
                    # Merge results preserving all turns: store as lists per action_type
                    # Note: resume() uses the same turn number as the original run()
                    current_turn = self._turn_number
                    for action_type, result in prev.items():
                        if action_type not in self.conversation_previous_results:
                            self.conversation_previous_results[action_type] = []
                        # Append new result with turn number
                        self.conversation_previous_results[action_type].append({
                            "turn": current_turn,
                            "result": result
                        })
                    self.messages = self.messages + [
                        HumanMessage(content=user_message),
                        AIMessage(content=result_text),
                    ]
                    self._pending_user_message = None
                    yield {
                        "phase": "execution",
                        "status": "complete",
                        "result": result_text,
                        "all_results": prev,
                    }
            else:
                yield {
                    "phase": "execution",
                    "node": step.get("node"),
                    "output": step.get("output"),
                    "status": "running",
                }


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
