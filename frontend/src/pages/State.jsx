import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { getState } from '../api'
import './State.css'

function State({ conversationId: conversationIdProp, embedded, onClose }) {
  const { id: urlId } = useParams()
  const conversationId = conversationIdProp ?? urlId
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!conversationId) {
      setData({ state: null, message: 'No conversation yet. Send a message to start.' })
      return
    }
    getState(conversationId)
      .then(setData)
      .catch(() => setData({ state: null, message: 'Could not load state. The conversation may no longer exist on the server.' }))
  }, [conversationId])

  if (data === null) return <div className="page-loading">Loading…</div>

  const { state, message } = data

  return (
    <div className={`state-page${embedded ? ' embedded' : ''}`}>
      <div className="state-nav">
        {embedded && onClose ? (
          <button type="button" className="link-btn" onClick={onClose}>Close</button>
        ) : (
          conversationId ? (
            <Link to={`/conversation/${conversationId}`}>&larr; Back to Chat</Link>
          ) : (
            <Link to="/">&larr; Back to Chat</Link>
          )
        )}
      </div>
      <h1>State (debug)</h1>
      {message && <p className="muted">{message}</p>}
      {state == null ? (
        !message && <p className="muted">No active run (not paused, no graph).</p>
      ) : (
        <pre className="state-json">{JSON.stringify(state, null, 2)}</pre>
      )}
    </div>
  )
}

export default State
