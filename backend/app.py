"""
FastAPI backend for dynagraph: chat (SSE), resume (HITL), conversation, state, graph, documents.
Multi-tenant: every request scoped by X-Tenant-ID header.
Run from repo root: PYTHONPATH=. uvicorn backend.app:app --host 0.0.0.0 --port 5001
"""
import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from backend.agent.graph import get_current_state, get_graph_mermaid
from backend.agent.runtime import ConversationAgent
from backend.db.tenant import (
    conversation_belongs_to_tenant,
    create_conversation,
    create_tenant,
    delete_conversation,
    delete_tenant,
    ensure_tables,
    get_tenant,
    list_conversations,
    list_tenants,
    touch_conversation,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-Tenant-ID"],
)

# In-memory store: (tenant_id, conversation_id) -> ConversationAgent
_agents: Dict[tuple, ConversationAgent] = {}

HITL_BEFORE = ["SQL_EXECUTION"]
TENANT_EXEMPT_PREFIXES = ("/api/health", "/api/tenants", "/docs", "/redoc", "/openapi.json")

# Bootstrap DB tables on import (idempotent)
try:
    ensure_tables()
except Exception as e:
    log.warning("Could not ensure tenant tables (DB may not be reachable yet): %s", e)


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


def _tenant_id(request: Request) -> str:
    return request.state.tenant_id


async def _json_body(request: Request) -> Dict[str, Any]:
    try:
        body = await request.json()
        if isinstance(body, dict):
            return body
    except Exception:
        pass
    return {}


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


def _build_graph_from_actions(actions: list) -> Tuple[List[Dict], List[Dict]]:
    if not actions:
        return [], []
    sorted_actions = sorted(actions, key=lambda x: x.get("execution_order", 0))
    nodes = [
        {
            "id": a.get("action_type", "unknown"),
            "label": a.get("description", "") or a.get("action_type", ""),
        }
        for a in sorted_actions
    ]
    execution_groups: List[Tuple[int, List[str]]] = []
    for action in sorted_actions:
        order = action.get("execution_order", 0)
        if not execution_groups or execution_groups[-1][0] != order:
            execution_groups.append((order, []))
        execution_groups[-1][1].append(action.get("action_type", "unknown"))
    edges = []
    for idx, (_, group) in enumerate(execution_groups):
        if idx + 1 < len(execution_groups):
            next_group = execution_groups[idx + 1][1]
            for src in group:
                for tgt in next_group:
                    edges.append({"source": src, "target": tgt})
    return nodes, edges


def _first_user_message_as_title(message: str, max_len: int = 60) -> str:
    title = message.strip().split("\n")[0]
    if len(title) > max_len:
        title = title[:max_len] + "…"
    return title


def _filter_event(event: Dict[str, Any]) -> Dict[str, Any]:
    visible_keys = (
        "phase",
        "status",
        "message",
        "plan",
        "actions",
        "paused_before_nodes",
        "pending_actions",
        "completed_nodes",
        "user_message",
        "result",
        "all_results",
        "node",
        "output",
    )
    filtered = {k: v for k, v in event.items() if k in visible_keys}
    if "pending_actions" in filtered:
        filtered["pending_actions"] = [
            {
                k: v
                for k, v in action.items()
                if isinstance(v, (str, int, float, bool, type(None)))
                or (isinstance(v, (list, dict)) and not isinstance(v, (type,)))
            }
            for action in (filtered.get("pending_actions") or [])
        ]
    return filtered


@app.middleware("http")
async def require_tenant(request: Request, call_next):
    """Extract and validate X-Tenant-ID header. Sets request.state.tenant_id."""
    if request.method == "OPTIONS":
        return await call_next(request)

    path = request.url.path
    if any(path.startswith(prefix) for prefix in TENANT_EXEMPT_PREFIXES):
        return await call_next(request)

    tenant_id = request.headers.get("X-Tenant-ID", "").strip()
    if not tenant_id:
        return JSONResponse({"error": "X-Tenant-ID header is required"}, status_code=400)

    tenant = get_tenant(tenant_id)
    if not tenant:
        return JSONResponse({"error": f"Tenant '{tenant_id}' not found"}, status_code=404)

    request.state.tenant_id = tenant_id
    return await call_next(request)


@app.get("/api/health")
def api_health():
    return {"status": "ok"}


@app.get("/api/config")
def api_config():
    from backend.agent.actions import ACTION_REGISTRY
    from backend.agent.llm_config import ACTION_LLM_OVERRIDES, DEFAULT_MODEL

    actions = []
    for key, defn in ACTION_REGISTRY.items():
        llm_override = ACTION_LLM_OVERRIDES.get(key)
        actions.append(
            {
                "action_type": key,
                "kind": defn.kind,
                "description": defn.description,
                "llm_model": (
                    llm_override["model"]
                    if llm_override
                    else DEFAULT_MODEL if defn.kind == "llm" else None
                ),
                "temperature": (
                    llm_override.get("temperature", 0)
                    if llm_override
                    else 0 if defn.kind == "llm" else None
                ),
                "hitl_enabled": key in HITL_BEFORE,
            }
        )

    return {
        "default_model": DEFAULT_MODEL,
        "default_temperature": 0,
        "hitl_before": HITL_BEFORE,
        "actions": actions,
    }


@app.get("/api/tenants")
def api_tenants_list():
    return list_tenants()


@app.post("/api/tenants")
async def api_tenants_create(request: Request):
    body = await _json_body(request)
    name = (body.get("name") or "").strip()
    if not name:
        return JSONResponse({"error": "name is required"}, status_code=400)
    try:
        tenant = create_tenant(name, tenant_id=body.get("id"))
        return JSONResponse(tenant, status_code=201)
    except Exception as e:
        if "unique" in str(e).lower() or "duplicate" in str(e).lower():
            return JSONResponse({"error": f"Tenant name '{name}' already exists"}, status_code=409)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/tenants/{tenant_id}")
def api_tenant_get(tenant_id: str):
    tenant = get_tenant(tenant_id)
    if not tenant:
        return JSONResponse({"error": "Tenant not found"}, status_code=404)
    return tenant


@app.delete("/api/tenants/{tenant_id}")
def api_tenant_delete(tenant_id: str):
    keys_to_remove = [k for k in _agents if k[0] == tenant_id]
    for key in keys_to_remove:
        del _agents[key]
    if delete_tenant(tenant_id):
        return {"deleted": True}
    return JSONResponse({"error": "Tenant not found"}, status_code=404)


@app.get("/api/conversations")
def api_conversations_list(request: Request):
    return list_conversations(_tenant_id(request))


@app.delete("/api/conversations/{conversation_id}")
def api_conversation_delete(conversation_id: str, request: Request):
    tenant_id = _tenant_id(request)
    key = _agent_key(tenant_id, conversation_id)
    if key in _agents:
        del _agents[key]
    if delete_conversation(conversation_id, tenant_id):
        return {"deleted": True}
    return JSONResponse({"error": "Conversation not found"}, status_code=404)


@app.post("/api/chat")
async def api_chat(request: Request):
    try:
        tenant_id = _tenant_id(request)
        body = await _json_body(request)
        message = (body.get("message") or "").strip()
        conversation_id = body.get("conversation_id")
        if not message:
            return JSONResponse({"error": "message is required"}, status_code=400)

        is_new = False
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            is_new = True
        elif not conversation_belongs_to_tenant(conversation_id, tenant_id):
            return JSONResponse({"error": "Conversation not found for this tenant"}, status_code=404)

        if is_new:
            title = _first_user_message_as_title(message)
            create_conversation(tenant_id, conversation_id=conversation_id, title=title)
        else:
            touch_conversation(conversation_id, tenant_id)

        agent = _get_or_create_agent(tenant_id, conversation_id)

        def generate():
            yield f"data: {json.dumps({'conversation_id': conversation_id})}\n\n"
            for event in agent.run(message):
                payload = json.dumps(_filter_event(event), default=str)
                yield f"data: {payload}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/resume")
async def api_resume(request: Request):
    try:
        tenant_id = _tenant_id(request)
        body = await _json_body(request)
        conversation_id = body.get("conversation_id")
        param_overrides = body.get("param_overrides") or {}
        if not conversation_id:
            return JSONResponse({"error": "conversation_id is required"}, status_code=400)
        agent = _get_agent(tenant_id, conversation_id)
        if not agent:
            return JSONResponse({"error": "No conversation found or execution not paused"}, status_code=404)

        human_feedback = {"param_overrides": param_overrides}

        def generate():
            for event in agent.resume(human_feedback):
                payload = json.dumps(_filter_event(event), default=str)
                yield f"data: {payload}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/conversation/{conversation_id}")
def api_conversation(conversation_id: str, request: Request):
    tenant_id = _tenant_id(request)
    agent = _get_or_restore_agent(tenant_id, conversation_id)
    if not agent:
        return JSONResponse({"error": "Conversation not found"}, status_code=404)
    messages = [_serialize_message(msg) for msg in agent.get_conversation_history()]
    paused_payload = agent.get_paused_payload()
    return {
        "messages": messages,
        "conversation_previous_results": agent.conversation_previous_results,
        "paused": paused_payload,
        "hitl_before": HITL_BEFORE,
    }


@app.get("/api/state")
def api_state(request: Request):
    tenant_id = _tenant_id(request)
    conversation_id = request.query_params.get("conversation_id")
    if not conversation_id:
        return JSONResponse({"state": None, "message": "conversation_id required"}, status_code=400)

    agent = _get_or_restore_agent(tenant_id, conversation_id)
    if not agent:
        return {"state": None, "message": "No conversation state yet. Send a message to start."}

    # If there's an active LangGraph run, show its live execution state
    if agent.current_graph and agent.thread_id:
        state = get_current_state(agent.current_graph, agent.thread_id)
        serialized = _serialize_state(state)
        return {"state": serialized}

    # Otherwise, return the persisted conversation state from DB
    messages = [_serialize_message(msg) for msg in agent.get_conversation_history()]
    last_turn = agent.conversation_previous_results[-1] if agent.conversation_previous_results else {}
    return {
        "state": {
            "messages": messages,
            "previous_results": last_turn.get("actions", []),
            "plan": last_turn.get("plan", ""),
            "actions": last_turn.get("actions", []),
            "turn_number": agent._turn_number,
            "conversation_previous_results": agent.conversation_previous_results,
        }
    }


@app.get("/api/graph")
def api_graph(request: Request):
    tenant_id = _tenant_id(request)
    conversation_id = request.query_params.get("conversation_id")
    turn_raw = request.query_params.get("turn")
    turn = None
    if turn_raw is not None:
        try:
            turn = int(turn_raw)
        except ValueError:
            return JSONResponse({"error": "turn must be an integer"}, status_code=400)

    if not conversation_id:
        return JSONResponse({"error": "conversation_id required"}, status_code=400)

    agent = _get_or_restore_agent(tenant_id, conversation_id)
    if not agent or not agent.conversation_previous_results:
        return {
            "plan": "",
            "nodes": [],
            "edges": [],
            "turn_results": [],
            "interrupt_before": HITL_BEFORE,
            "graph_mermaid": None,
        }

    results = agent.conversation_previous_results
    if turn is not None:
        if turn < 0 or turn >= len(results):
            return {
                "plan": "",
                "nodes": [],
                "edges": [],
                "turn_results": [],
                "interrupt_before": HITL_BEFORE,
                "graph_mermaid": None,
                "error": "turn out of range",
            }
        turn_data = results[turn]
    else:
        turn_data = results[-1]

    actions = turn_data.get("actions", [])
    nodes, edges = _build_graph_from_actions(actions)
    graph_mermaid = get_graph_mermaid(actions, hitl_before=HITL_BEFORE)
    return {
        "plan": turn_data.get("plan", ""),
        "nodes": nodes,
        "edges": edges,
        "turn_results": turn_data,
        "interrupt_before": HITL_BEFORE,
        "graph_mermaid": graph_mermaid,
    }


@app.get("/api/trace")
def api_trace(request: Request):
    tenant_id = _tenant_id(request)
    conversation_id = request.query_params.get("conversation_id")
    if not conversation_id:
        return JSONResponse({"error": "conversation_id required"}, status_code=400)

    agent = _get_or_restore_agent(tenant_id, conversation_id)
    if not agent:
        return JSONResponse({"error": "Conversation not found"}, status_code=404)

    trace = agent.trace
    if request.query_params.get("summary") in ("true", "1"):
        return trace.get_summary()

    event_type = request.query_params.get("event_type")
    if event_type:
        events = trace.get_trace(event_types=event_type.split(","))
    else:
        events = trace.get_trace()
    return {"conversation_id": conversation_id, "events": events}


@app.get("/api/documents")
def api_documents():
    try:
        from backend.db.rag_indexing import list_documents

        docs = list_documents()
        out = []
        for doc in docs:
            rec = {"source": doc.get("source", ""), "chunk_count": doc.get("chunk_count", 0)}
            created_at = doc.get("created_at")
            if hasattr(created_at, "isoformat"):
                rec["created_at"] = created_at.isoformat()
            else:
                rec["created_at"] = created_at
            out.append(rec)
        return out
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/tables")
def api_tables():
    try:
        from backend.db import list_tables

        return list_tables()
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import os

    import uvicorn

    port = int(os.environ.get("PORT", 5001))
    reload_mode = os.environ.get("FASTAPI_RELOAD", "0") == "1"
    uvicorn.run("backend.app:app", host="0.0.0.0", port=port, reload=reload_mode)
