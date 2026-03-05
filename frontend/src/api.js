const API_BASE = '/api';

const TENANT_STORAGE_KEY = 'dynagraph_tenant_id'

function getTenantId() {
  return localStorage.getItem(TENANT_STORAGE_KEY) || ''
}

function tenantHeaders(extra = {}) {
  const tid = getTenantId()
  const headers = { 'Content-Type': 'application/json', ...extra }
  if (tid) headers['X-Tenant-ID'] = tid
  return headers
}

export async function* parseSSE(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const raw = line.slice(6).trim();
          if (raw === '[DONE]' || raw === '') continue;
          try {
            yield JSON.parse(raw);
          } catch (_) {}
        }
      }
    }
    if (buffer.startsWith('data: ')) {
      const raw = buffer.slice(6).trim();
      if (raw) {
        try {
          yield JSON.parse(raw);
        } catch (_) {}
      }
    }
  } finally {
    reader.releaseLock();
  }
}

export async function* chatStream(body) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: 'POST',
    headers: tenantHeaders(),
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const msg = res.status === 403
      ? (err.error || "Forbidden. Start the backend first (PYTHONPATH=. python -m backend.app), then open http://127.0.0.1:5001/api/health in a new tab to confirm it returns {\"status\":\"ok\"}. Use the app at http://localhost:5173.")
      : (err.error || res.statusText);
    throw new Error(msg);
  }
  yield* parseSSE(res);
}

export async function* resumeStream(body) {
  const res = await fetch(`${API_BASE}/resume`, {
    method: 'POST',
    headers: tenantHeaders(),
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || res.statusText);
  }
  yield* parseSSE(res);
}

export async function getConversation(id) {
  const res = await fetch(`${API_BASE}/conversation/${id}`, { headers: tenantHeaders() });
  if (!res.ok) throw new Error('Failed to load conversation');
  return res.json();
}

export async function getState(conversationId) {
  const res = await fetch(`${API_BASE}/state?conversation_id=${encodeURIComponent(conversationId)}`, { headers: tenantHeaders() });
  if (!res.ok) throw new Error('Failed to load state');
  return res.json();
}

export async function getGraph(conversationId, turn = null) {
  let url = `${API_BASE}/graph?conversation_id=${encodeURIComponent(conversationId)}`;
  if (turn != null) url += `&turn=${turn}`;
  const res = await fetch(url, { headers: tenantHeaders() });
  if (!res.ok) throw new Error('Failed to load graph');
  return res.json();
}

export async function getDocuments() {
  const res = await fetch(`${API_BASE}/documents`, { headers: tenantHeaders() });
  if (!res.ok) throw new Error('Failed to load documents');
  return res.json();
}

export async function getTables() {
  const res = await fetch(`${API_BASE}/tables`, { headers: tenantHeaders() });
  if (!res.ok) throw new Error('Failed to load tables');
  return res.json();
}

export async function getConfig() {
  const res = await fetch(`${API_BASE}/config`, { headers: tenantHeaders() });
  if (!res.ok) throw new Error('Failed to load config');
  return res.json();
}

// Tenant-scoped conversation list
export async function getConversations() {
  const res = await fetch(`${API_BASE}/conversations`, { headers: tenantHeaders() });
  if (!res.ok) throw new Error('Failed to load conversations');
  return res.json();
}

export async function deleteConversation(conversationId) {
  const res = await fetch(`${API_BASE}/conversations/${conversationId}`, {
    method: 'DELETE',
    headers: tenantHeaders(),
  });
  if (!res.ok) throw new Error('Failed to delete conversation');
  return res.json();
}
