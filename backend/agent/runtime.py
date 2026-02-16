from typing import Dict, Any, Optional, Generator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import uuid

load_dotenv()

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


class AgentRunner:
    """
    Agent runner with support for streaming and human-in-the-loop.
    """
    
    def __init__(self, enable_hitl: bool = False, hitl_before: list = None):
        """
        Args:
            enable_hitl: Enable human-in-the-loop mode.
            hitl_before: List of action types to pause before (e.g., ["RESPONSE_GENERATION"]).
        """
        self.enable_hitl = enable_hitl
        self.hitl_before = hitl_before or []
        self.current_graph = None
        self.current_thread_id = None
        self.current_state = None
    
    def run(self, user_message: str) -> Generator[Dict[str, Any], None, None]:
        """
        Run the agent with streaming output.
        
        Args:
            user_message: User's input message.
        
        Yields:
            Progress updates for each step.
        """
        self.current_thread_id = str(uuid.uuid4())
        
        # Initial state
        self.current_state = {
            "messages": [HumanMessage(content=user_message)],
            "previous_results": {},
            "need_clarification": False,
            "plan": "",
            "actions": [],
        }
        
        # Phase 1: Planning
        yield {"phase": "planning", "status": "running"}
        
        planner_result = planning_agent(self.current_state)
        self.current_state.update(planner_result)
        
        yield {
            "phase": "planning",
            "status": "complete",
            "plan": self.current_state.get("plan", ""),
            "actions": self.current_state.get("actions", []),
        }
        
        # Check if clarification is needed
        if self.current_state.get("need_clarification"):
            yield {
                "phase": "clarification",
                "status": "waiting",
                "message": "Clarification needed from user",
            }
            return
        
        # Phase 2: Execution
        actions = self.current_state.get("actions", [])
        if not actions:
            yield {
                "phase": "execution",
                "status": "complete",
                "message": "No actions to execute",
            }
            return
        
        # Create execution graph
        self.current_graph, checkpointer = create_execution_graph(
            actions,
            enable_hitl=self.enable_hitl,
            hitl_before=self.hitl_before,
        )
        
        if not self.current_graph:
            yield {"phase": "execution", "status": "error", "message": "Failed to create graph"}
            return
        
        yield {"phase": "execution", "status": "running"}
        
        # Stream execution
        for step in stream_execution(self.current_graph, self.current_state, self.current_thread_id):
            if step.get("status") == "complete":
                # Get final state from checkpointer (always available)
                final_state = get_current_state(self.current_graph, self.current_thread_id)
                if final_state:
                    self.current_state.update(final_state)
                
                yield {
                    "phase": "execution",
                    "status": "complete",
                    "result": self.current_state.get("previous_results", {}).get("RESPONSE_GENERATION", ""),
                    "all_results": self.current_state.get("previous_results", {}),
                }
            else:
                # Update current state from checkpointer (always available)
                current = get_current_state(self.current_graph, self.current_thread_id)
                if current:
                    self.current_state.update(current)
                
                yield {
                    "phase": "execution",
                    "node": step.get("node"),
                    "output": step.get("output"),
                    "status": "running",
                }
    
    def resume(self, human_feedback: Optional[Dict[str, Any]] = None) -> Generator[Dict[str, Any], None, None]:
        """
        Resume execution after human review (HITL mode).
        
        Args:
            human_feedback: Optional feedback/modifications from human.
        
        Yields:
            Progress updates for remaining steps.
        """
        if not self.current_graph or not self.current_thread_id:
            yield {"status": "error", "message": "No paused execution to resume"}
            return
        
        yield {"phase": "execution", "status": "resuming"}
        
        for step in resume_execution(self.current_graph, self.current_thread_id, human_feedback):
            if step.get("status") == "complete":
                final_state = get_current_state(self.current_graph, self.current_thread_id)
                if final_state:
                    self.current_state.update(final_state)
                
                yield {
                    "phase": "execution",
                    "status": "complete",
                    "result": self.current_state.get("previous_results", {}).get("RESPONSE_GENERATION", ""),
                    "all_results": self.current_state.get("previous_results", {}),
                }
            else:
                yield {
                    "phase": "execution",
                    "node": step.get("node"),
                    "output": step.get("output"),
                    "status": "running",
                }
    
    def get_state(self) -> Optional[Dict[str, Any]]:
        """Get current state."""
        return self.current_state


# Simple function for basic usage (no HITL)
def run_agent(user_message: str) -> Dict[str, Any]:
    """
    Run the agent and return final result (non-streaming).
    
    Args:
        user_message: User's input message.
    
    Returns:
        Final result dict.
    """
    runner = AgentRunner(enable_hitl=False)
    result = None
    
    for step in runner.run(user_message):
        result = step
    
    return result
