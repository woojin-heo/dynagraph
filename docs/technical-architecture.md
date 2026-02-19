# Technical Architecture

This document describes implementation details for conversation memory, session management, and RAG in the DynaGraph agent.

---

## 1. Conversation memory

### Where it lives

- **State shape**: Conversation history is stored in `AgentState.messages` (`backend/agent/state.py`). The field is typed as `Annotated[List[BaseMessage], add_messages]`, so LangGraph’s `add_messages` reducer is used when multiple nodes update `messages` (e.g. appending assistant turns).
- **Scope**: When using **ConversationAgent**, memory is in-memory and **scoped to the agent instance**: `messages` and `conversation_previous_results` are accumulated across `agent.run()` calls. When using **run_agent()**, each call creates a new ConversationAgent, so there is no persistence across calls.
- **Turn-based cache**: `conversation_previous_results` keeps the planner’s plan shape and attaches execution results per turn: `[{"turn": 1, "plan": "...", "need_clarification": false, "actions": [{"action_type": "...", "description": "...", "dependencies": [...], "execution_order": N, "result": "..."}, ...]}, ...]`. This makes it easy to see “what was the plan for turn N” and “what was the result of each action” when planning (e.g. for CONTEXT_REFERENCE).

### How it’s used

- **Planning** (`planner.py`): The planner receives `recent_messages = messages[-10:]` (last 10 messages) and the last user message as `user_request`. So the planning agent sees a **sliding window of the last 10 messages** for context.
- **Execution** (`graph.py` → `action_executor`): Each action node receives `conversation_history = state.get('messages', [])[-10:]` and `user_request` from the last message. So both planning and execution use the **same 10-message window** for conversation context.

### Summary

| Aspect | Behavior |
|--------|----------|
| Storage | In-memory only; part of `AgentState` |
| Reducer | `add_messages` (LangGraph) for `messages` |
| Window | Last 10 messages in planner and in action nodes |
| Cross-run | Yes when reusing one ConversationAgent; no when using run_agent() (new agent per call) |

---

## 2. Session management

### What a “session” is

- A **session** in this codebase is effectively a **single execution run** of the agent. There is no long-lived “user session” that spans multiple `run()` calls (e.g. no session store or cookie-based session ID for the app).

### Thread ID and checkpointer

- **ConversationAgent** creates a **new thread ID per execution run** (`self._current_thread_id = str(uuid.uuid4())` at the start of each `run()` in `backend/agent/runtime.py`). Each turn builds a new graph, so using a new thread_id per run avoids mixing checkpointer state across different graphs. The same thread_id is used only for that run’s `stream_execution` and, if paused, for `resume_execution`.
- The **execution graph** is compiled with a **LangGraph checkpointer** (`MemorySaver()` in `graph.py`). When streaming or resuming, the code passes `config = {"configurable": {"thread_id": thread_id}}` so LangGraph can load/save state for that thread.
- **Purpose**: The thread ID allows **human-in-the-loop (HITL)** to pause and later **resume** the same run via `resume_execution(graph, self.current_thread_id, human_feedback)`. The checkpointer stores the graph state (including `previous_results`, etc.) keyed by `thread_id`.
- **HITL parameter editing**: When paused, the client receives `pending_actions` (each with `action_type`, `description`, `params`). The user can edit params and call `agent.resume(human_feedback)` with `human_feedback = {"param_overrides": {"ACTION_TYPE": {"param_key": "value"}}}` (e.g. `{"param_overrides": {"SEARCH_TAVILY": {"query": "new query"}}}`). `resume_execution` writes this into state as `human_param_overrides` via `graph.update_state`; `action_executor` merges these overrides into each action’s `params` before running.

### Persistence and scope

- **MemorySaver** is in-memory. Restarting the process clears all checkpoint state.
- Thread IDs are **not** persisted across process restarts. For multi-turn conversations, reuse the same **ConversationAgent** instance (e.g. one per user session in your API server); the agent keeps `messages` and `conversation_previous_results` across `run()` calls.

### Summary

| Aspect | Behavior |
|--------|----------|
| Session | One run of the agent; no built-in multi-run “user session” |
| Thread ID | Create per run; Only for each graph execution and HITL resume |
| Checkpointer | `MemorySaver()` (in-memory); used for HITL pause/resume |
| Scope | Single process; no cross-process or persistent session store |

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

| Aspect | Behavior |
|--------|----------|
| Retrieval | Vector similarity in PostgreSQL: `embedding <=> query_vector LIMIT top_k` |
| Default top_k | 5 (overridable per action via params) |
| Embedding model | OpenAI `text-embedding-3-small` (1536 dims) |
| Planner | Sees list of sources from `document_chunks`; no vector search at plan time |
| Indexing | Chunk size 1000, overlap 200; same embedding model |

---

## 4. Other technical areas (placeholders)

The following topics are commonly part of agent system design but are **not yet implemented** in this codebase. Sections are reserved here for future documentation.


### Token budgeting and context window

*(Not yet implemented.)*

- Max input/output tokens per request; context window ceiling.
- Strategy when conversation + tools + RAG context exceed window (truncation, summarization, or error).
- Token counting and cost estimation.

### Observability and tracing

*(Not yet implemented. “Tracing” is listed in key capabilities; implementation TBD.)*

- Structured logging (request ID, thread_id, phase, latency).
- Metrics (latency, token usage, error rate, tool call counts).
- End-to-end trace for a run (plan → actions → results) for debugging and analytics.

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

*(Not yet implemented.)*

- Primary vs fallback model; conditions for switching.
- Model selection by task type (e.g. planning vs generation) or by tenant.
- Configuration (temperature, max_tokens) per phase or per action.

### Conversation and checkpoint persistence

*(Not yet implemented.)*

- **Conversation persistence**: Storing `messages` (and optionally `previous_results`) in a DB or store so that multi-turn “sessions” survive across `run()` calls and restarts.
- **Checkpoint persistence**: Replacing in-memory `MemorySaver` with a durable checkpointer (e.g. LangGraph PostgreSQL/Redis checkpointer) so HITL pause/resume and state survive process restarts.

---

## 5. Where to look in the code

| Topic | Main files |
|--------|------------|
| Conversation memory | `backend/agent/state.py` (state + reducers), `backend/agent/planner.py` (recent_messages), `backend/agent/graph.py` (action_executor conversation_history) |
| Session / thread | `backend/agent/runtime.py` (thread_id, run/resume), `backend/agent/graph.py` (MemorySaver, stream_execution, resume_execution, get_current_state) |
| RAG retrieval | `backend/agent/tools/rag_tools.py` (search_document), `backend/agent/prompt_lib.py` (get_available_documents) |
| RAG indexing | `backend/db/rag_indexing.py`; setup and schema in `docs/vector-db-setup.md` |
