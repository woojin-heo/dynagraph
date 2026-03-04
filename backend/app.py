"""
Flask API for dynagraph: chat (SSE), resume (HITL), conversation, state, graph, documents.
Run from repo root: FLASK_APP=backend.app:app flask run
"""
import json
import uuid
from typing import Any, Dict, Generator, List, Tuple

from flask import Flask, Response, request, jsonify
from flask_cors import CORS

from backend.agent.runtime import ConversationAgent
from backend.agent.graph import get_current_state, get_graph_mermaid

app = Flask(__name__)
# Allow frontend origin and proxy; avoid 403 on POST (e.g. CORS preflight)
CORS(app, resources={r"/api/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type"]}})

# In-memory store: conversation_id -> ConversationAgent
_agents: Dict[str, ConversationAgent] = {}

HITL_BEFORE = ["SQL_EXECUTION"]


def _get_or_create_agent(conversation_id: str) -> ConversationAgent:
    if conversation_id not in _agents:
        _agents[conversation_id] = ConversationAgent(
            enable_hitl=True,
            hitl_before=HITL_BEFORE,
            conversation_id=conversation_id,
        )
    return _agents[conversation_id]


def _serialize_message(msg: Any) -> Dict[str, Any]:
    """Convert LangChain BaseMessage to JSON-serializable dict."""
    if hasattr(msg, "type") and hasattr(msg, "content"):
        t = getattr(msg, "type", "unknown")
        if t == "human":
            kind = "human"
        elif t == "ai":
            kind = "ai"
        else:
            kind = str(t)
        content = msg.content
        if isinstance(content, list):
            content = str(content)
        return {"role": kind, "content": content or ""}
    return {"role": "unknown", "content": str(msg)}


def _serialize_state(state: Dict[str, Any] | None) -> Dict[str, Any] | None:
    """Make agent state JSON-serializable (messages, datetime, etc.)."""
    if state is None:
        return None
    out = {}
    if "messages" in state and state["messages"]:
        out["messages"] = [_serialize_message(m) for m in state["messages"]]
    if "previous_results" in state:
        out["previous_results"] = state["previous_results"]
    if "plan" in state:
        out["plan"] = state["plan"]
    if "actions" in state:
        out["actions"] = state["actions"]
    if "need_clarification" in state:
        out["need_clarification"] = state["need_clarification"]
    if "human_param_overrides" in state:
        out["human_param_overrides"] = state["human_param_overrides"]
    return out


def _sse_stream(gen: Generator[Dict[str, Any], None, None]) -> Generator[str, None, None]:
    """Yield SSE-formatted lines from a generator of dict events."""
    for event in gen:
        payload = json.dumps(event, default=str)
        yield f"data: {payload}\n\n"


@app.route("/api/health", methods=["GET"])
def api_health():
    """Check backend is reachable (e.g. via frontend proxy)."""
    return jsonify({"status": "ok"})


@app.route("/api/chat", methods=["OPTIONS"])
def api_chat_options():
    """Explicit OPTIONS for CORS preflight (avoid 403 in strict environments)."""
    return "", 200


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Stream agent.run(message) via SSE. Body: { conversation_id?, message }."""
    try:
        body = request.get_json(force=True, silent=True) or {}
        message = (body.get("message") or "").strip()
        conversation_id = body.get("conversation_id")
        if not message:
            return jsonify({"error": "message is required"}), 400
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        agent = _get_or_create_agent(conversation_id)

        def generate():
            # Send conversation_id first so client can store it
            yield f"data: {json.dumps({'conversation_id': conversation_id})}\n\n"
            for event in agent.run(message):
                # Make event JSON-serializable (e.g. no custom objects)
                ev = {k: v for k, v in event.items() if k in (
                    "phase", "status", "message", "plan", "actions",
                    "paused_before_nodes", "pending_actions", "completed_nodes",
                    "user_message", "result", "all_results", "node", "output",
                )}
                if "pending_actions" in ev:
                    ev["pending_actions"] = [
                        {k: v for k, v in a.items() if isinstance(v, (str, int, float, bool, type(None))) or (isinstance(v, (list, dict)) and not isinstance(v, (type,)))}
                        for a in (ev.get("pending_actions") or [])
                    ]
                payload = json.dumps(ev, default=str)
                yield f"data: {payload}\n\n"

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/resume", methods=["OPTIONS"])
def api_resume_options():
    return "", 200


@app.route("/api/resume", methods=["POST"])
def api_resume():
    """Stream agent.resume(human_feedback) via SSE. Body: { conversation_id, param_overrides? }."""
    try:
        body = request.get_json(force=True, silent=True) or {}
        conversation_id = body.get("conversation_id")
        param_overrides = body.get("param_overrides") or {}
        if not conversation_id:
            return jsonify({"error": "conversation_id is required"}), 400
        if conversation_id not in _agents:
            return jsonify({"error": "No conversation found or execution not paused"}), 404
        agent = _agents[conversation_id]
        human_feedback = {"param_overrides": param_overrides}

        def generate():
            for event in agent.resume(human_feedback):
                ev = {k: v for k, v in event.items() if k in (
                    "phase", "status", "message", "plan", "actions",
                    "paused_before_nodes", "pending_actions", "completed_nodes",
                    "user_message", "result", "all_results", "node", "output",
                )}
                if "pending_actions" in ev:
                    ev["pending_actions"] = [
                        {k: v for k, v in a.items() if isinstance(v, (str, int, float, bool, type(None))) or (isinstance(v, (list, dict)) and not isinstance(v, (type,)))}
                        for a in (ev.get("pending_actions") or [])
                    ]
                payload = json.dumps(ev, default=str)
                yield f"data: {payload}\n\n"

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/conversation/<conversation_id>", methods=["GET"])
def api_conversation(conversation_id: str):
    """Return conversation history and previous turn results (plan/actions)."""
    if conversation_id not in _agents:
        return jsonify({"error": "Conversation not found"}), 404
    agent = _agents[conversation_id]
    messages = [_serialize_message(m) for m in agent.get_conversation_history()]
    return jsonify({
        "messages": messages,
        "conversation_previous_results": agent.conversation_previous_results,
    })


@app.route("/api/state", methods=["GET"])
def api_state():
    """Return current graph state for debugging. Query: conversation_id."""
    conversation_id = request.args.get("conversation_id")
    if not conversation_id or conversation_id not in _agents:
        return jsonify({"error": "conversation_id required or conversation not found"}), 400
    agent = _agents[conversation_id]
    if not agent.current_graph or not agent.thread_id:
        return jsonify({"state": None, "message": "No active run (not paused, no graph)"}), 200
    state = get_current_state(agent.current_graph, agent.thread_id)
    serialized = _serialize_state(state)
    return jsonify({"state": serialized})


def _build_graph_from_actions(actions: list) -> Tuple[List[Dict], List[Dict]]:
    """From planned actions build nodes and edges (execution_order). Returns (nodes, edges)."""
    if not actions:
        return [], []
    sorted_actions = sorted(actions, key=lambda x: x.get("execution_order", 0))
    nodes = [{"id": a.get("action_type", "unknown"), "label": a.get("description", "") or a.get("action_type", "")} for a in sorted_actions]
    execution_groups = []
    for a in sorted_actions:
        order = a.get("execution_order", 0)
        if not execution_groups or execution_groups[-1][0] != order:
            execution_groups.append((order, []))
        execution_groups[-1][1].append(a.get("action_type", "unknown"))
    edges = []
    for i, (_, group) in enumerate(execution_groups):
        if i + 1 < len(execution_groups):
            next_group = execution_groups[i + 1][1]
            for src in group:
                for tgt in next_group:
                    edges.append({"source": src, "target": tgt})
    return nodes, edges


@app.route("/api/graph", methods=["GET"])
def api_graph():
    """Return plan and graph (nodes, edges) for a turn. Query: conversation_id, optional turn (0-based)."""
    conversation_id = request.args.get("conversation_id")
    turn = request.args.get("turn", type=int)
    if not conversation_id or conversation_id not in _agents:
        return jsonify({"error": "conversation_id required or conversation not found"}), 400
    agent = _agents[conversation_id]
    if not agent.conversation_previous_results:
        return jsonify({"plan": "", "nodes": [], "edges": [], "turn_results": [], "interrupt_before": HITL_BEFORE, "graph_mermaid": None})
    results = agent.conversation_previous_results
    if turn is not None:
        if turn < 0 or turn >= len(results):
            return jsonify({"plan": "", "nodes": [], "edges": [], "turn_results": [], "interrupt_before": HITL_BEFORE, "graph_mermaid": None, "error": "turn out of range"})
        turn_data = results[turn]
    else:
        turn_data = results[-1]
    plan = turn_data.get("plan", "")
    actions = turn_data.get("actions", [])
    nodes, edges = _build_graph_from_actions(actions)
    graph_mermaid = get_graph_mermaid(actions, hitl_before=HITL_BEFORE)
    return jsonify({
        "plan": plan,
        "nodes": nodes,
        "edges": edges,
        "turn_results": turn_data,
        "interrupt_before": HITL_BEFORE,
        "graph_mermaid": graph_mermaid,
    })


@app.route("/api/trace", methods=["GET"])
def api_trace():
    """Return trace events for debugging. Query: conversation_id, optional event_type, optional summary."""
    conversation_id = request.args.get("conversation_id")
    if not conversation_id or conversation_id not in _agents:
        return jsonify({"error": "conversation_id required or conversation not found"}), 400
    agent = _agents[conversation_id]
    trace = agent.trace

    if request.args.get("summary") in ("true", "1"):
        return jsonify(trace.get_summary())

    event_type = request.args.get("event_type")
    if event_type:
        events = trace.get_trace(event_types=event_type.split(","))
    else:
        events = trace.get_trace()
    return jsonify({"conversation_id": conversation_id, "events": events})


@app.route("/api/documents", methods=["GET"])
def api_documents():
    """Return list of documents in vector DB (source, chunk_count, created_at)."""
    try:
        from backend.db.rag_indexing import list_documents
        docs = list_documents()
        out = []
        for d in docs:
            rec = {"source": d.get("source", ""), "chunk_count": d.get("chunk_count", 0)}
            created_at = d.get("created_at")
            if hasattr(created_at, "isoformat"):
                rec["created_at"] = created_at.isoformat()
            else:
                rec["created_at"] = created_at
            out.append(rec)
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/tables", methods=["GET"])
def api_tables():
    """Return list of SQL tables from backend/db/tables.yaml (name, description)."""
    try:
        from backend.db import list_tables
        tables = list_tables()
        return jsonify(tables)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
