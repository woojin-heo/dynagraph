import { useState, useEffect, useCallback } from 'react'
import { Routes, Route, Link, Navigate, useNavigate } from 'react-router-dom'
import { useAuth } from './AuthContext'
import { getConversations, deleteConversation } from './api'
import Chat from './pages/Chat'
import Graph from './pages/Graph'
import Documents from './pages/Documents'
import State from './pages/State'
import Login from './pages/Login'
import Register from './pages/Register'
import './App.css'

function ProtectedRoute({ children }) {
  const { isAuthenticated, loading } = useAuth()
  if (loading) return <div className="auth-loading">Loading…</div>
  if (!isAuthenticated) return <Navigate to="/login" replace />
  return children
}

function UserMenu() {
  const { user, logout } = useAuth()
  return (
    <div className="user-menu">
      <span className="user-display-name">{user?.display_name}</span>
      <button type="button" className="signout-btn" onClick={logout}>Sign out</button>
    </div>
  )
}

function ConversationList() {
  const [conversations, setConversations] = useState([])
  const [loading, setLoading] = useState(false)
  const navigate = useNavigate()

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const data = await getConversations()
      setConversations(data)
    } catch (_) {
      setConversations([])
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { refresh() }, [refresh])

  const handleDelete = async (e, id) => {
    e.stopPropagation()
    if (!confirm('Delete this conversation?')) return
    try {
      await deleteConversation(id)
      refresh()
    } catch (_) {}
  }

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

function AppLayout() {
  return (
    <div className="app">
      <nav className="nav">
        <div className="nav-links">
          <Link to="/">Chat</Link>
          <Link to="/documents">Documents</Link>
        </div>
        <UserMenu />
      </nav>
      <ConversationList />
      <main className="main">
        <Routes>
          <Route path="/" element={<Chat />} />
          <Route path="/conversation/:id" element={<Chat />} />
          <Route path="/conversation/:id/graph" element={<Graph />} />
          <Route path="/conversation/:id/state" element={<State />} />
          <Route path="/documents" element={<Documents />} />
        </Routes>
      </main>
    </div>
  )
}

function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />
      <Route path="/*" element={
        <ProtectedRoute>
          <AppLayout />
        </ProtectedRoute>
      } />
    </Routes>
  )
}

export default App
