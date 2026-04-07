const API_BASE = '/api';

const AUTH_STORAGE_KEY = 'dynagraph_auth_token'

function getToken() {
  return localStorage.getItem(AUTH_STORAGE_KEY) || ''
}

function authHeaders(extra = {}) {
  const token = getToken()
  const headers = { 'Content-Type': 'application/json', ...extra }
  if (token) headers['Authorization'] = `Bearer ${token}`
  return headers
}

export async function login(username, password) {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.error || 'Login failed')
  }
  return res.json()
}

export async function register(username, password, displayName) {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password, display_name: displayName }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.error || 'Registration failed')
  }
  return res.json()
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
    headers: authHeaders(),
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    const msg = res.status === 401
      ? (err.error || 'Unauthorized. Please log in.')
      : (err.error || res.statusText);
    throw new Error(msg);
  }
  yield* parseSSE(res);
}

export async function* resumeStream(body) {
  const res = await fetch(`${API_BASE}/resume`, {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || res.statusText);
  }
  yield* parseSSE(res);
}

export async function getConversation(id) {
  const res = await fetch(`${API_BASE}/conversation/${id}`, { headers: authHeaders() });
  if (!res.ok) throw new Error('Failed to load conversation');
  return res.json();
}

export async function getState(conversationId) {
  const res = await fetch(`${API_BASE}/state?conversation_id=${encodeURIComponent(conversationId)}`, { headers: authHeaders() });
  if (!res.ok) throw new Error('Failed to load state');
  return res.json();
}

export async function getGraph(conversationId, turn = null) {
  let url = `${API_BASE}/graph?conversation_id=${encodeURIComponent(conversationId)}`;
  if (turn != null) url += `&turn=${turn}`;
  const res = await fetch(url, { headers: authHeaders() });
  if (!res.ok) throw new Error('Failed to load graph');
  return res.json();
}

export async function getDocuments() {
  const res = await fetch(`${API_BASE}/documents`, { headers: authHeaders() });
  if (!res.ok) throw new Error('Failed to load documents');
  return res.json();
}

export async function getTables() {
  const res = await fetch(`${API_BASE}/tables`, { headers: authHeaders() });
  if (!res.ok) throw new Error('Failed to load tables');
  return res.json();
}

export async function getConfig() {
  const res = await fetch(`${API_BASE}/config`, { headers: authHeaders() });
  if (!res.ok) throw new Error('Failed to load config');
  return res.json();
}

export async function getConversations() {
  const res = await fetch(`${API_BASE}/conversations`, { headers: authHeaders() });
  if (!res.ok) throw new Error('Failed to load conversations');
  return res.json();
}

export async function deleteConversation(conversationId) {
  const res = await fetch(`${API_BASE}/conversations/${conversationId}`, {
    method: 'DELETE',
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error('Failed to delete conversation');
  return res.json();
}
