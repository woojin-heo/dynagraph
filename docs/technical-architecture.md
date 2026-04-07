# Technical Architecture

This document describes implementation details for conversation memory, session management, and RAG in the DynaGraph agent.

---

## 1. Conversation memory

### Where it lives

- **State shape**: Conversation history is stored in `AgentState.messages` (`backend/agent/state.py`). The field is typed as `Annotated[List[BaseMessage], add_messages]`, so LangGraph’s `add_messages` reducer is used when multiple nodes update `messages` (e.g. appending assistant turns).
- **Scope**: Memory is in-memory and **scoped to the ConversationAgent instance**: `messages` and `conversation_previous_results` are accumulated across `agent.run()` calls while the same agent object is reused.
- **Turn-based cache**: `conversation_previous_results` keeps the planner’s plan shape and attaches execution results per turn: `[{"turn": 1, "plan": "...", "need_clarification": false, "actions": [{"action_type": "...", "description": "...", "dependencies": [...], "execution_order": N, "result": "..."}, ...]}, ...]`. This makes it easy to see “what was the plan for turn N” and “what was the result of each action” when planning (e.g. for CONTEXT_REFERENCE).

### How it’s used

- **Planning** (`planner.py`): The planner receives `recent_messages = messages[-10:]` (last 10 messages) and the last user message as `user_request`. So the planning agent sees a **sliding window of the last 10 messages** for context.
- **Execution** (`graph.py` → `action_executor`): Each action node receives `conversation_history = state.get('messages', [])[-10:]` and `user_request` from the last message. So both planning and execution use the **same 10-message window** for conversation context.

### Summary


| Aspect    | Behavior                                                                                                            |
| --------- | ------------------------------------------------------------------------------------------------------------------- |
| Storage   | In-memory `AgentState` during execution + DB persistence (`conversations.messages`) for tenant-backed conversations |
| Reducer   | `add_messages` (LangGraph) for `messages`                                                                           |
| Window    | Last 10 messages in planner and in action nodes                                                                     |
| Cross-run | Yes when reusing one ConversationAgent instance, and also restorable from DB via `load_from_db()`                   |


---

## 2. Session management

### What a “session” is

- A **session** in this codebase is effectively a **single execution run** of the agent. There is no long-lived “user session” that spans multiple `run()` calls (e.g. no session store or cookie-based session ID for the app).

### Thread ID and checkpointer

- **ConversationAgent** creates a **new thread ID per execution run** (`self._current_thread_id = str(uuid.uuid4())` at the start of each `run()` in `backend/agent/runtime.py`). Each turn builds a new graph, so using a new thread_id per run avoids mixing checkpointer state across different graphs. The same thread_id is used only for that run’s `stream_execution` and, if paused, for `resume_execution`.
- The **execution graph** is compiled with a **LangGraph checkpointer** selected by `get_checkpointer()` in `graph.py`:
  - Prefer `**PostgresSaver`** when `langgraph.checkpoint.postgres` + `psycopg` are available and `DATABASE_URL` is reachable.
  - Fallback to `**MemorySaver**` when Postgres checkpointer is unavailable.
  When streaming or resuming, the code passes `config = {"configurable": {"thread_id": thread_id}}` so LangGraph can load/save state for that thread.
- **Purpose**: The thread ID allows **human-in-the-loop (HITL)** to pause and later **resume** the same run via `resume_execution(graph, self.current_thread_id, human_feedback)`. The checkpointer stores the graph state (including `previous_results`, etc.) keyed by `thread_id`.
- **HITL parameter editing**: When paused, the client receives `pending_actions` (each with `action_type`, `description`, `params`). The user can edit params and call `agent.resume(human_feedback)` with `human_feedback = {"param_overrides": {"ACTION_TYPE": {"param_key": "value"}}}` (e.g. `{"param_overrides": {"SEARCH_TAVILY": {"query": "new query"}}}`). `resume_execution` writes this into state as `human_param_overrides` via `graph.update_state`; `action_executor` merges these overrides into each action’s `params` before running.

### Persistence and scope

- **Conversation state persistence** is implemented for tenant-backed conversations:
  - `ConversationAgent.save_to_db()` persists `messages`, `conversation_previous_results`, and `turn_number` into `conversations` table JSONB columns.
  - `ConversationAgent.load_from_db()` restores them when the agent is created/recovered.
- **Checkpoint persistence** depends on checkpointer backend:
  - With `**PostgresSaver`**, pause/resume checkpoints can survive process restarts (DB-backed).
  - With `**MemorySaver**`, checkpoints are process-local and cleared on restart.
- Thread IDs are generated per run and used to isolate each execution graph instance.

### Summary


| Aspect       | Behavior                                                                                          |
| ------------ | ------------------------------------------------------------------------------------------------- |
| Session      | One run of the agent; no built-in multi-run “user session”                                        |
| Thread ID    | Create per run; Only for each graph execution and HITL resume                                     |
| Checkpointer | `PostgresSaver` when available; otherwise `MemorySaver`                                           |
| Scope        | Conversation state can be restored from DB; checkpoint durability depends on checkpointer backend |


---

## 3. RAG implementation

### Overview

- RAG is used for **internal document search**. The planner can emit a `SEARCH_DOCUMENT` action; the corresponding tool runs a **vector similarity search** over a PostgreSQL table (`document_chunks`) using the **pgvector** extension. Setup and schema are documented in [Vector DB Setup](vector-db-setup.md).

### Retrieval: top‑k and similarity

- **Tool**: `search_document(query: str, top_k: int = 5)` in `backend/agent/tools/rag_tools.py`.
- **Flow**:
  1. Query is embedded with **OpenAI** `text-embedding-3-small` (dimension 1536).
  2. PostgreSQL query:
    `ORDER BY embedding <=> %s::vector LIMIT %s`  
     So retrieval is **nearest-neighbor by embedding distance** (operator `<=>`), and the number of returned chunks is **exactly `top_k`** (default 5).
  3. Returned columns: `content`, `source`, `chunk_index`, `metadata`. Results are formatted as `<Document source="..." chunk_index="..." ...>` blocks for the LLM.
- **Top‑k**: The planner can pass `top_k` in the action params (e.g. `params: {"query": "leave entitlements", "top_k": 5}`). The tool default is `top_k=5`; there is no separate reranking or second-stage retrieval in this implementation.

### What the planner sees

- Before planning, the agent gets a **list of available documents** (no vector search at planning time). `get_available_documents()` in `prompt_lib.py` runs:
  - `SELECT source, COUNT(*) AS chunk_count, MAX(created_at) AS created_at FROM document_chunks GROUP BY source ORDER BY ...`
- That list is injected into the planning prompt as `available_documents`, so the planner can decide whether to use `SEARCH_DOCUMENT` and with what query. **Retrieval (top‑k)** happens only when the executor runs the `SEARCH_DOCUMENT` tool with a given `query` and `top_k`.

### Indexing (for context)

- Chunking and indexing are in `backend/db/rag_indexing.py`: **RecursiveCharacterTextSplitter** with default `chunk_size=1000`, `chunk_overlap=200`; embeddings again with `text-embedding-3-small`; inserts into `document_chunks`. See [Vector DB Setup](vector-db-setup.md) for schema and usage.

### Summary


| Aspect          | Behavior                                                                   |
| --------------- | -------------------------------------------------------------------------- |
| Retrieval       | Vector similarity in PostgreSQL: `embedding <=> query_vector LIMIT top_k`  |
| Default top_k   | 5 (overridable per action via params)                                      |
| Embedding model | OpenAI `text-embedding-3-small` (1536 dims)                                |
| Planner         | Sees list of sources from `document_chunks`; no vector search at plan time |
| Indexing        | Chunk size 1000, overlap 200; same embedding model                         |


---

## 4. Other technical areas (placeholders)

The following topics are commonly part of agent system design but are **not yet implemented** in this codebase. Sections are reserved here for future documentation.

### Token budgeting and context window

*(Not yet implemented.)*

- Max input/output tokens per request; context window ceiling.
- Strategy when conversation + tools + RAG context exceed window (truncation, summarization, or error).
- Token counting and cost estimation.

### Observability and tracing

Implemented baseline tracing exists; full observability stack is still partial.

- **Implemented now**
  - `TraceCollector` captures structured events per conversation (e.g. planning/action start/end, retries, errors).
  - File logging to `logs/agent.jsonl` (JSON Lines with UTC timestamp and event payload).
  - API endpoint `GET /api/trace` supports raw event list and summary view (`?summary=true`).
- **Not yet implemented**
  - Aggregated metrics pipeline (latency/token/cost dashboards and alerting).
  - Request-level correlation IDs propagated across all layers.
  - Long-term trace storage/query infrastructure beyond local file + in-memory collector.

### Error handling and fallbacks

*(Not yet implemented.)*

- Behavior on LLM timeout or failure (retry, fallback model, user-facing error).
- Tool failure handling (partial results, retry, or graceful degradation).
- Circuit breaker or health checks for external services.

### Caching

*(Not yet implemented.)*

- Caching of LLM responses for identical or similar inputs (e.g. semantic cache).
- Embedding cache for repeated RAG queries.
- Cache invalidation and TTL policy.

### Guardrails and safety

*(Not yet implemented.)*

- Input/output content filtering (e.g. PII, harmful content).
- Validation or sanitization of tool parameters and LLM outputs.
- Allowed vs blocked tool use by context or policy.

### Tool execution semantics

*(Not yet implemented.)*

- Timeout per tool call; behavior on timeout.
- Maximum number of tool calls per run or per turn.
- Idempotency or deduplication for side-effecting tools.

### Model configuration and routing

Basic model configuration and action-level override are implemented; advanced routing is not.

- **Implemented now**
  - Global default model in `backend/agent/llm_config.py` (`DEFAULT_MODEL`).
  - Action-level override via `ACTION_LLM_OVERRIDES` (currently used for `SQL_GENERATION`).
  - Runtime selection via `get_llm_for_action(action_type)`.
  - Config visibility via `GET /api/config`.
- **Not yet implemented**
  - Automatic primary/fallback switching by runtime failures or cost policy.
  - Tenant-specific or dynamic routing policy by task complexity.
  - Per-phase token-budget governance and adaptive model selection.

### Conversation and checkpoint persistence

Partially implemented (conversation state is implemented; checkpoint durability is backend-dependent).

- **Conversation persistence (implemented)**
  - `conversations` table stores `messages` (JSONB), `turn_results` (JSONB), and `turn_number`.
  - Agent restores persisted state on cache miss (`_get_or_restore_agent()` + `load_from_db()`).
- **Checkpoint persistence (partially implemented)**
  - Durable when `PostgresSaver` is active.
  - Non-durable when fallback `MemorySaver` is active.

---

## 5. Where to look in the code


| Topic               | Main files                                                                                                                                                                                                                          |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Conversation memory | `backend/agent/state.py` (state + reducers), `backend/agent/planner.py` (recent_messages), `backend/agent/graph.py` (action_executor conversation_history)                                                                          |
| Session / thread    | `backend/agent/runtime.py` (thread_id, run/resume, DB save/load), `backend/agent/graph.py` (checkpointer selection, stream_execution, resume_execution, get_current_state), `backend/db/tenant.py` (conversation state persistence) |
| RAG retrieval       | `backend/agent/tools/rag_tools.py` (search_document), `backend/agent/prompt_lib.py` (get_available_documents)                                                                                                                       |
| RAG indexing        | `backend/db/rag_indexing.py`; setup and schema in `docs/vector-db-setup.md`                                                                                                                                                         |
| Tracing             | `backend/agent/trace.py` (collector + file logging), `backend/app.py` (`/api/trace`)                                                                                                                                                |


