import { useState, useEffect, useCallback } from 'react'
import { Routes, Route, Link, useNavigate } from 'react-router-dom'
import { useTenant } from './TenantContext'
import { getConversations, deleteConversation } from './api'
import Chat from './pages/Chat'
import Graph from './pages/Graph'
import Documents from './pages/Documents'
import State from './pages/State'
import './App.css'

function TenantSelector() {
  const { tenantId, tenantName, tenants, setTenant, createTenant, refreshTenants } = useTenant()
  const [showCreate, setShowCreate] = useState(false)
  const [newName, setNewName] = useState('')
  const [error, setError] = useState(null)

  const handleCreate = async () => {
    const name = newName.trim()
    if (!name) return
    setError(null)
    try {
      await createTenant(name)
      setNewName('')
      setShowCreate(false)
    } catch (e) {
      setError(e.message)
    }
  }

  return (
    <div className="tenant-selector">
      <select
        value={tenantId || ''}
        onChange={(e) => {
          const id = e.target.value
          if (!id) { setTenant(null, null); return }
          const t = tenants.find((t) => t.id === id)
          setTenant(id, t?.name || '')
        }}
      >
        <option value="">Select tenant…</option>
        {tenants.map((t) => (
          <option key={t.id} value={t.id}>{t.name}</option>
        ))}
      </select>
      {!showCreate ? (
        <button type="button" className="tenant-add-btn" onClick={() => setShowCreate(true)} title="New tenant">+</button>
      ) : (
        <span className="tenant-create-inline">
          <input
            type="text"
            placeholder="Tenant name"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') handleCreate() }}
            autoFocus
          />
          <button type="button" onClick={handleCreate}>Create</button>
          <button type="button" onClick={() => { setShowCreate(false); setError(null) }}>Cancel</button>
        </span>
      )}
      {error && <span className="tenant-error">{error}</span>}
    </div>
  )
}

function ConversationList() {
  const { tenantId } = useTenant()
  const [conversations, setConversations] = useState([])
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()

  const refresh = useCallback(async () => {
    if (!tenantId) { setConversations([]); return }
    setLoading(true)
    try {
      const data = await getConversations()
      setConversations(data)
    } catch (_) {
      setConversations([])
    } finally {
      setLoading(false)
    }
  }, [tenantId])

  useEffect(() => { refresh() }, [refresh])

  const handleDelete = async (e, id) => {
    e.stopPropagation()
    if (!confirm('Delete this conversation?')) return
    try {
      await deleteConversation(id)
      refresh()
    } catch (_) {}
  }

  if (!tenantId) return null

  return (
    <div className="conversation-list">
      <div className="conversation-list-header">
        <span>Conversations</span>
        <button type="button" className="conv-refresh-btn" onClick={refresh} title="Refresh">&#8635;</button>
      </div>
      {loading && <div className="conv-loading">Loading…</div>}
      {!loading && conversations.length === 0 && <div className="conv-empty">No conversations yet</div>}
      {conversations.map((c) => (
        <div
          key={c.id}
          className="conv-item"
          onClick={() => navigate(`/conversation/${c.id}`)}
          title={c.title || c.id}
        >
          <span className="conv-title">{c.title || c.id.slice(0, 8)}</span>
          <button type="button" className="conv-delete-btn" onClick={(e) => handleDelete(e, c.id)} title="Delete">&times;</button>
        </div>
      ))}
    </div>
  )
}

function App() {
  const { tenantId } = useTenant()

  return (
    <div className="app">
      <nav className="nav">
        <div className="nav-links">
          <Link to="/">Chat</Link>
          <Link to="/documents">Documents</Link>
        </div>
        <TenantSelector />
      </nav>
      {tenantId && <ConversationList />}
      <main className="main">
        {!tenantId ? (
          <div className="tenant-gate">
            <h2>Select a tenant to start</h2>
            <p>Create a new tenant or select an existing one from the top-right dropdown.</p>
          </div>
        ) : (
          <Routes>
            <Route path="/" element={<Chat />} />
            <Route path="/conversation/:id" element={<Chat />} />
            <Route path="/conversation/:id/graph" element={<Graph />} />
            <Route path="/conversation/:id/state" element={<State />} />
            <Route path="/documents" element={<Documents />} />
          </Routes>
        )}
      </main>
    </div>
  )
}

export default App
