"""
Flask API for dynagraph: chat (SSE), resume (HITL), conversation, state, graph, documents.
Multi-tenant: every request scoped by X-Tenant-ID header.
Run from repo root: FLASK_APP=backend.app:app flask run
"""
import json
import uuid
import logging
from typing import Any, Dict, Generator, List, Tuple, Optional
from functools import wraps

from flask import Flask, Response, request, jsonify, g
from flask_cors import CORS

from backend.agent.runtime import ConversationAgent
from backend.agent.graph import get_current_state, get_graph_mermaid
from backend.db.tenant import (
    ensure_tables,
    create_tenant, get_tenant, list_tenants, delete_tenant,
    create_conversation, list_conversations, get_conversation_meta,
    update_conversation_title, touch_conversation, delete_conversation,
    conversation_belongs_to_tenant,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {
    "origins": "*",
    "methods": ["GET", "POST", "DELETE", "OPTIONS"],
    "allow_headers": ["Content-Type", "X-Tenant-ID"],
}})

# In-memory store: (tenant_id, conversation_id) -> ConversationAgent
_agents: Dict[tuple, ConversationAgent] = {}

HITL_BEFORE = ["SQL_EXECUTION"]

# Bootstrap DB tables on import (idempotent)
try:
    ensure_tables()
except Exception as e:
    log.warning("Could not ensure tenant tables (DB may not be reachable yet): %s", e)


# ---------------------------------------------------------------------------
# Tenant middleware
# ---------------------------------------------------------------------------

TENANT_EXEMPT_PREFIXES = ("/api/health", "/api/tenants")


def _require_tenant():
    """Extract and validate X-Tenant-ID header. Sets g.tenant_id."""
    if request.method == "OPTIONS":
        return None
    path = request.path
    if any(path.startswith(p) for p in TENANT_EXEMPT_PREFIXES):
        return None
    tenant_id = request.headers.get("X-Tenant-ID", "").strip()
    if not tenant_id:
        return jsonify({"error": "X-Tenant-ID header is required"}), 400
    tenant = get_tenant(tenant_id)
    if not tenant:
        return jsonify({"error": f"Tenant '{tenant_id}' not found"}), 404
    g.tenant_id = tenant_id
    return None


app.before_request(_require_tenant)


def _agent_key(tenant_id: str, conversation_id: str) -> tuple:
    return (tenant_id, conversation_id)


def _make_agent(tenant_id: str, conversation_id: str) -> ConversationAgent:
    return ConversationAgent(
        enable_hitl=True,
        hitl_before=HITL_BEFORE,
        conversation_id=conversation_id,
        tenant_id=tenant_id,
    )


def _get_or_create_agent(tenant_id: str, conversation_id: str) -> ConversationAgent:
    """Return an in-memory agent, restoring from DB if necessary."""
    key = _agent_key(tenant_id, conversation_id)
    if key not in _agents:
        agent = _make_agent(tenant_id, conversation_id)
        agent.load_from_db()
        _agents[key] = agent
    return _agents[key]


def _get_agent(tenant_id: str, conversation_id: str) -> Optional[ConversationAgent]:
    return _agents.get(_agent_key(tenant_id, conversation_id))


def _get_or_restore_agent(tenant_id: str, conversation_id: str) -> Optional[ConversationAgent]:
    """Return in-memory agent, or restore from DB if the conversation exists."""
    key = _agent_key(tenant_id, conversation_id)
    if key in _agents:
        return _agents[key]
    if not conversation_belongs_to_tenant(conversation_id, tenant_id):
        return None
    agent = _make_agent(tenant_id, conversation_id)
    agent.load_from_db()
    _agents[key] = agent
    return agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_message(msg: Any) -> Dict[str, Any]:
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
    for event in gen:
        payload = json.dumps(event, default=str)
        yield f"data: {payload}\n\n"


def _build_graph_from_actions(actions: list) -> Tuple[List[Dict], List[Dict]]:
    if not actions:
        return [], []
    sorted_actions = sorted(actions, key=lambda x: x.get("execution_order", 0))
    nodes = [{"id": a.get("action_type", "unknown"), "label": a.get("description", "") or a.get("action_type", "")} for a in sorted_actions]
    execution_groups: List[Tuple[int, List[str]]] = []
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


def _first_user_message_as_title(message: str, max_len: int = 60) -> str:
    title = message.strip().split("\n")[0]
    if len(title) > max_len:
        title = title[:max_len] + "…"
    return title


# ---------------------------------------------------------------------------
# Health / Config (tenant-exempt)
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({"status": "ok"})


@app.route("/api/config", methods=["GET"])
def api_config():
    from backend.agent.actions import ACTION_REGISTRY
    from backend.agent.runtime import DEFAULT_MODEL, ACTION_LLM_OVERRIDES

    actions = []
    for key, defn in ACTION_REGISTRY.items():
        llm_override = ACTION_LLM_OVERRIDES.get(key)
        actions.append({
            "action_type": key,
            "kind": defn.kind,
            "description": defn.description,
            "llm_model": llm_override["model"] if llm_override else DEFAULT_MODEL if defn.kind == "llm" else None,
            "temperature": llm_override.get("temperature", 0) if llm_override else 0 if defn.kind == "llm" else None,
            "hitl_enabled": key in HITL_BEFORE,
        })
    return jsonify({
        "default_model": DEFAULT_MODEL,
        "default_temperature": 0,
        "hitl_before": HITL_BEFORE,
        "actions": actions,
    })


# ---------------------------------------------------------------------------
# Tenant CRUD (tenant-exempt – no X-Tenant-ID needed)
# ---------------------------------------------------------------------------

@app.route("/api/tenants", methods=["GET"])
def api_tenants_list():
    return jsonify(list_tenants())


@app.route("/api/tenants", methods=["POST"])
def api_tenants_create():
    body = request.get_json(force=True, silent=True) or {}
    name = (body.get("name") or "").strip()
    if not name:
        return jsonify({"error": "name is required"}), 400
    try:
        tenant = create_tenant(name, tenant_id=body.get("id"))
        return jsonify(tenant), 201
    except Exception as e:
        if "unique" in str(e).lower() or "duplicate" in str(e).lower():
            return jsonify({"error": f"Tenant name '{name}' already exists"}), 409
        return jsonify({"error": str(e)}), 500


@app.route("/api/tenants/<tenant_id>", methods=["GET"])
def api_tenant_get(tenant_id: str):
    tenant = get_tenant(tenant_id)
    if not tenant:
        return jsonify({"error": "Tenant not found"}), 404
    return jsonify(tenant)


@app.route("/api/tenants/<tenant_id>", methods=["DELETE"])
def api_tenant_delete(tenant_id: str):
    keys_to_remove = [k for k in _agents if k[0] == tenant_id]
    for k in keys_to_remove:
        del _agents[k]
    if delete_tenant(tenant_id):
        return jsonify({"deleted": True})
    return jsonify({"error": "Tenant not found"}), 404


@app.route("/api/tenants", methods=["OPTIONS"])
def api_tenants_options():
    return "", 200


# ---------------------------------------------------------------------------
# Conversation list (tenant-scoped)
# ---------------------------------------------------------------------------

@app.route("/api/conversations", methods=["GET"])
def api_conversations_list():
    tenant_id = g.tenant_id
    return jsonify(list_conversations(tenant_id))


@app.route("/api/conversations/<conversation_id>", methods=["DELETE"])
def api_conversation_delete(conversation_id: str):
    tenant_id = g.tenant_id
    key = _agent_key(tenant_id, conversation_id)
    if key in _agents:
        del _agents[key]
    if delete_conversation(conversation_id, tenant_id):
        return jsonify({"deleted": True})
    return jsonify({"error": "Conversation not found"}), 404


@app.route("/api/conversations", methods=["OPTIONS"])
def api_conversations_options():
    return "", 200


# ---------------------------------------------------------------------------
# Chat / Resume (tenant-scoped)
# ---------------------------------------------------------------------------

@app.route("/api/chat", methods=["OPTIONS"])
def api_chat_options():
    return "", 200


@app.route("/api/chat", methods=["POST"])
def api_chat():
    try:
        tenant_id = g.tenant_id
        body = request.get_json(force=True, silent=True) or {}
        message = (body.get("message") or "").strip()
        conversation_id = body.get("conversation_id")
        if not message:
            return jsonify({"error": "message is required"}), 400

        is_new = False
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            is_new = True
        else:
            if not conversation_belongs_to_tenant(conversation_id, tenant_id):
                return jsonify({"error": "Conversation not found for this tenant"}), 404

        if is_new:
            title = _first_user_message_as_title(message)
            create_conversation(tenant_id, conversation_id=conversation_id, title=title)
        else:
            touch_conversation(conversation_id, tenant_id)

        agent = _get_or_create_agent(tenant_id, conversation_id)

        def generate():
            yield f"data: {json.dumps({'conversation_id': conversation_id})}\n\n"
            for event in agent.run(message):
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
    try:
        tenant_id = g.tenant_id
        body = request.get_json(force=True, silent=True) or {}
        conversation_id = body.get("conversation_id")
        param_overrides = body.get("param_overrides") or {}
        if not conversation_id:
            return jsonify({"error": "conversation_id is required"}), 400
        agent = _get_agent(tenant_id, conversation_id)
        if not agent:
            return jsonify({"error": "No conversation found or execution not paused"}), 404
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


# ---------------------------------------------------------------------------
# Conversation detail, state, graph, trace (tenant-scoped)
# ---------------------------------------------------------------------------

@app.route("/api/conversation/<conversation_id>", methods=["GET"])
def api_conversation(conversation_id: str):
    tenant_id = g.tenant_id
    agent = _get_or_restore_agent(tenant_id, conversation_id)
    if not agent:
        return jsonify({"error": "Conversation not found"}), 404
    messages = [_serialize_message(m) for m in agent.get_conversation_history()]
    paused_payload = agent.get_paused_payload()
    return jsonify({
        "messages": messages,
        "conversation_previous_results": agent.conversation_previous_results,
        "paused": paused_payload,
        "hitl_before": HITL_BEFORE,
    })


@app.route("/api/state", methods=["GET"])
def api_state():
    tenant_id = g.tenant_id
    conversation_id = request.args.get("conversation_id")
    if not conversation_id:
        return jsonify({"state": None, "message": "conversation_id required"}), 400

    agent = _get_or_restore_agent(tenant_id, conversation_id)
    if not agent:
        return jsonify({"state": None, "message": "No conversation state yet. Send a message to start."}), 200

    # If there's an active LangGraph run, show its live execution state
    if agent.current_graph and agent.thread_id:
        state = get_current_state(agent.current_graph, agent.thread_id)
        serialized = _serialize_state(state)
        return jsonify({"state": serialized})

    # Otherwise, return the persisted conversation state from DB
    messages = [_serialize_message(m) for m in agent.get_conversation_history()]
    last_turn = agent.conversation_previous_results[-1] if agent.conversation_previous_results else {}
    return jsonify({"state": {
        "messages": messages,
        "previous_results": last_turn.get("actions", []),
        "plan": last_turn.get("plan", ""),
        "actions": last_turn.get("actions", []),
        "turn_number": agent._turn_number,
        "conversation_previous_results": agent.conversation_previous_results,
    }})


@app.route("/api/graph", methods=["GET"])
def api_graph():
    tenant_id = g.tenant_id
    conversation_id = request.args.get("conversation_id")
    turn = request.args.get("turn", type=int)
    if not conversation_id:
        return jsonify({"error": "conversation_id required"}), 400
    agent = _get_or_restore_agent(tenant_id, conversation_id)
    if not agent or not agent.conversation_previous_results:
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
    tenant_id = g.tenant_id
    conversation_id = request.args.get("conversation_id")
    if not conversation_id:
        return jsonify({"error": "conversation_id required"}), 400
    agent = _get_or_restore_agent(tenant_id, conversation_id)
    if not agent:
        return jsonify({"error": "Conversation not found"}), 404
    trace = agent.trace
    if request.args.get("summary") in ("true", "1"):
        return jsonify(trace.get_summary())
    event_type = request.args.get("event_type")
    if event_type:
        events = trace.get_trace(event_types=event_type.split(","))
    else:
        events = trace.get_trace()
    return jsonify({"conversation_id": conversation_id, "events": events})


# ---------------------------------------------------------------------------
# Documents / Tables (shared resources, tenant-scoped via header validation)
# ---------------------------------------------------------------------------

@app.route("/api/documents", methods=["GET"])
def api_documents():
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
    try:
        from backend.db import list_tables
        tables = list_tables()
        return jsonify(tables)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
