import { useState, useCallback, useEffect, Fragment } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import ReactMarkdown from 'react-markdown'
import { chatStream, resumeStream, getConversation } from '../api'
import Graph from './Graph'
import State from './State'
import './Chat.css'

const STORAGE_KEY = 'dynagraph_last_conversation_id'

function Chat() {
  const { id: urlId } = useParams()
  const navigate = useNavigate()
  const [conversationId, setConversationId] = useState(urlId || null)
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [phase, setPhase] = useState(null)
  const [streamError, setStreamError] = useState(null)
  const [paused, setPaused] = useState(null)
  const [hitlOverrides, setHitlOverrides] = useState({})
  const [resolvedHitlBlocks, setResolvedHitlBlocks] = useState([])
  const [hitlEditExpanded, setHitlEditExpanded] = useState(false)
  const [resuming, setResuming] = useState(false)
  const [loading, setLoading] = useState(!!urlId)
  const [panelView, setPanelView] = useState(null)
  const [panelTurnIndex, setPanelTurnIndex] = useState(null)
  const [panelWidth, setPanelWidth] = useState(480)

  const handleResizeStart = useCallback((e) => {
    e.preventDefault()
    const startX = e.clientX
    const startWidth = panelWidth
    const onMove = (moveEvent) => {
      const delta = startX - moveEvent.clientX
      setPanelWidth((w) => Math.min(800, Math.max(280, w + delta)))
    }
    const onUp = () => {
      document.removeEventListener('mousemove', onMove)
      document.removeEventListener('mouseup', onUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
    document.addEventListener('mousemove', onMove)
    document.addEventListener('mouseup', onUp)
  }, [panelWidth])

  useEffect(() => {
    if (urlId) {
      setConversationId(urlId)
      setLoading(true)
      getConversation(urlId)
        .then((data) => {
          const msgs = (data.messages || []).map((m) => ({
            role: m.role === 'ai' ? 'ai' : 'human',
            content: m.content || '',
          }))
          setMessages(msgs)
          setResolvedHitlBlocks([])
        })
        .catch(() => setMessages([]))
        .finally(() => setLoading(false))
      return
    }
    const lastId = sessionStorage.getItem(STORAGE_KEY)
    if (lastId) {
      setLoading(true)
      setConversationId(lastId)
      getConversation(lastId)
        .then((data) => {
          const msgs = (data.messages || []).map((m) => ({
            role: m.role === 'ai' ? 'ai' : 'human',
            content: m.content || '',
          }))
          setMessages(msgs)
          setResolvedHitlBlocks([])
        })
        .catch(() => setMessages([]))
        .finally(() => setLoading(false))
    } else {
      setConversationId(null)
      setMessages([])
    }
  }, [urlId])

  useEffect(() => {
    if (conversationId) sessionStorage.setItem(STORAGE_KEY, conversationId)
  }, [conversationId])

  const handleStartNewChat = useCallback(() => {
    sessionStorage.removeItem(STORAGE_KEY)
    setConversationId(null)
    setMessages([])
    setPanelView(null)
    setPanelTurnIndex(null)
    setStreamError(null)
    setPaused(null)
    setHitlOverrides({})
    setResolvedHitlBlocks([])
    setHitlEditExpanded(false)
    setPhase(null)
    if (urlId) navigate('/', { replace: true })
  }, [urlId, navigate])

  const handleSend = useCallback(async () => {
    const text = input.trim()
    if (!text) return
    setInput('')
    setPhase('planning')
    setStreamError(null)
    setPaused(null)
    setHitlOverrides({})
    setHitlEditExpanded(false)

    setMessages((prev) => [...prev, { role: 'human', content: text }])

    const body = { message: text }
    if (conversationId) body.conversation_id = conversationId

    try {
      for await (const event of chatStream(body)) {
        if (event.conversation_id) setConversationId(event.conversation_id)
        if (event.phase === 'planning') {
          setPhase(event.status === 'complete' ? 'execution' : 'planning')
        }
        if (event.phase === 'execution') {
          if (event.status === 'complete') {
            setPhase(null)
            if (event.result !== undefined) {
              setMessages((prev) => [...prev, { role: 'ai', content: event.result }])
            }
            break
          }
          if (event.status === 'paused') {
            setPhase('paused')
            setPaused(event)
            const overrides = {}
            for (const a of event.pending_actions || []) {
              const type = a.action_type
              if (type && a.params) overrides[type] = { ...a.params }
            }
            setHitlOverrides(overrides)
            break
          }
          setPhase('execution')
        }
        if (event.phase === 'clarification') {
          setPhase(null)
          setMessages((prev) => [...prev, { role: 'ai', content: event.message || 'Clarification needed.' }])
          break
        }
      }
    } catch (e) {
      setStreamError(e.message)
      setPhase(null)
    }
  }, [input, conversationId])

  const handleResume = useCallback(async () => {
    if (!conversationId || !paused) return
    const afterMessageIndex = Math.max(0, messages.length - 1)
    setResuming(true)
    setStreamError(null)
    try {
      for await (const event of resumeStream({
        conversation_id: conversationId,
        param_overrides: hitlOverrides,
      })) {
        if (event.phase === 'execution') {
          if (event.status === 'complete') {
            setResolvedHitlBlocks((prev) => [
              ...prev,
              { id: Date.now(), payload: paused, overrides: { ...hitlOverrides }, afterMessageIndex },
            ])
            setPhase(null)
            setPaused(null)
            if (event.result !== undefined) {
              setMessages((prev) => [...prev, { role: 'ai', content: event.result }])
            }
            break
          }
          if (event.status === 'paused') {
            setPaused(event)
            setHitlEditExpanded(false)
            const overrides = {}
            for (const a of event.pending_actions || []) {
              const type = a.action_type
              if (type && a.params) overrides[type] = { ...a.params }
            }
            setHitlOverrides(overrides)
            break
          }
        }
      }
    } catch (e) {
      setStreamError(e.message)
    } finally {
      setResuming(false)
    }
  }, [conversationId, paused, hitlOverrides, messages.length])

  const updateHitlParam = (actionType, key, value) => {
    setHitlOverrides((prev) => ({
      ...prev,
      [actionType]: { ...(prev[actionType] || {}), [key]: value },
    }))
  }

  if (loading) return <div className="chat-page"><div className="messages"><p className="muted">Loading conversation…</p></div></div>

  const openGraphPanel = (turnIndex = null) => {
    setPanelTurnIndex(turnIndex)
    setPanelView('graph')
  }

  return (
    <div className={`chat-page${panelView ? ' has-panel' : ''}`}>
      <div className="chat-main">
        <div className="messages">
          {messages.map((m, i) => (
            <Fragment key={`msg-${i}`}>
              <div className={`message message-${m.role}`}>
                <div className="message-content">
                  {m.role === 'ai' ? <ReactMarkdown>{m.content}</ReactMarkdown> : m.content}
                </div>
                {m.role === 'ai' && conversationId && (
                  <div className="message-actions">
                    <button
                      type="button"
                      className="link-btn"
                      onClick={() => openGraphPanel(Math.floor((i - 1) / 2))}
                    >
                      View plan
                    </button>
                  </div>
                )}
              </div>
              {resolvedHitlBlocks.filter((b) => b.afterMessageIndex === i).map((block) => (
                <div key={block.id} className="message message-ai hitl-inline hitl-resolved">
                  <div className="hitl-panel">
                    <p className="hitl-title">Human review</p>
                    {(block.payload.pending_actions || []).map((a, ai) => (
                      <div key={ai} className="hitl-action">
                        <strong>{a.action_type}</strong>
                        {a.params && typeof a.params === 'object' && (
                          <div className="hitl-params">
                            {Object.entries(a.params).map(([k, v]) => {
                              const val = block.overrides[a.action_type]?.[k] ?? v ?? ''
                              return (
                                <label key={k} className={k === 'query' ? 'hitl-param-query' : ''}>
                                  {k}
                                  {k === 'query' ? (
                                    <pre className="hitl-query-readonly">{val}</pre>
                                  ) : (
                                    <span className="hitl-value-readonly">{val}</span>
                                  )}
                                </label>
                              )
                            })}
                          </div>
                        )}
                      </div>
                    ))}
                    <button type="button" className="btn btn-primary hitl-resume-btn" disabled>
                      Resumed
                    </button>
                  </div>
                </div>
              ))}
              {paused && Math.max(0, messages.length - 1) === i && (
                <div className="message message-ai hitl-inline">
                  <div className="hitl-panel">
                    <p className="hitl-title">Human review</p>
                    {(paused.pending_actions || []).map((a, ai) => (
                      <div key={ai} className="hitl-action">
                        <strong>{a.action_type}</strong>
                        {a.params && typeof a.params === 'object' && (
                          <div className="hitl-params">
                            {Object.entries(a.params).map(([k, v]) => {
                              const val = hitlOverrides[a.action_type]?.[k] ?? v ?? ''
                              return (
                                <label key={k} className={k === 'query' ? 'hitl-param-query' : ''}>
                                  {k}
                                  {hitlEditExpanded ? (
                                    k === 'query' ? (
                                      <textarea
                                        className="hitl-query-textarea"
                                        rows={10}
                                        value={val}
                                        onChange={(e) => updateHitlParam(a.action_type, k, e.target.value)}
                                        spellCheck={false}
                                      />
                                    ) : (
                                      <input
                                        type="text"
                                        value={val}
                                        onChange={(e) => updateHitlParam(a.action_type, k, e.target.value)}
                                      />
                                    )
                                  ) : (
                                    k === 'query' ? (
                                      <pre className="hitl-query-readonly">{val}</pre>
                                    ) : (
                                      <span className="hitl-value-readonly">{val}</span>
                                    )
                                  )}
                                </label>
                              )
                            })}
                          </div>
                        )}
                      </div>
                    ))}
                    <button
                      type="button"
                      className="btn btn-secondary hitl-edit-btn"
                      onClick={() => setHitlEditExpanded((e) => !e)}
                    >
                      {hitlEditExpanded ? 'Hide parameters' : 'Edit parameters'}
                    </button>
                    <button type="button" className="btn btn-primary hitl-resume-btn" onClick={handleResume} disabled={resuming}>
                      {resuming ? 'Resuming…' : 'Resume'}
                    </button>
                  </div>
                </div>
              )}
            </Fragment>
          ))}
        {paused && messages.length === 0 && (
          <div className="message message-ai hitl-inline">
            <div className="hitl-panel">
              <p className="hitl-title">Human review</p>
              {(paused.pending_actions || []).map((a, ai) => (
                <div key={ai} className="hitl-action">
                  <strong>{a.action_type}</strong>
                  {a.params && typeof a.params === 'object' && (
                    <div className="hitl-params">
                      {Object.entries(a.params).map(([k, v]) => {
                        const val = hitlOverrides[a.action_type]?.[k] ?? v ?? ''
                        return (
                          <label key={k} className={k === 'query' ? 'hitl-param-query' : ''}>
                            {k}
                            {hitlEditExpanded ? (
                              k === 'query' ? (
                                <textarea
                                  className="hitl-query-textarea"
                                  rows={10}
                                  value={val}
                                  onChange={(e) => updateHitlParam(a.action_type, k, e.target.value)}
                                  spellCheck={false}
                                />
                              ) : (
                                <input
                                  type="text"
                                  value={val}
                                  onChange={(e) => updateHitlParam(a.action_type, k, e.target.value)}
                                />
                              )
                            ) : (
                              k === 'query' ? (
                                <pre className="hitl-query-readonly">{val}</pre>
                              ) : (
                                <span className="hitl-value-readonly">{val}</span>
                              )
                            )}
                          </label>
                        )
                      })}
                    </div>
                  )}
                </div>
              ))}
              <button
                type="button"
                className="btn btn-secondary hitl-edit-btn"
                onClick={() => setHitlEditExpanded((e) => !e)}
              >
                {hitlEditExpanded ? 'Hide parameters' : 'Edit parameters'}
              </button>
              <button type="button" className="btn btn-primary hitl-resume-btn" onClick={handleResume} disabled={resuming}>
                {resuming ? 'Resuming…' : 'Resume'}
              </button>
            </div>
          </div>
        )}
        {phase && phase !== 'paused' && (
          <div className="message message-ai status">
            {phase === 'planning' && 'Planning…'}
            {phase === 'execution' && 'Executing…'}
          </div>
        )}
        </div>

        {streamError && <div className="error">{streamError}</div>}

        <div className="input-row">
        <textarea
          className="input"
          placeholder="Type a message…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              handleSend()
            }
          }}
          rows={2}
          disabled={!!phase && phase !== 'paused'}
        />
        <button
          className="btn btn-send"
          onClick={handleSend}
          disabled={!input.trim() || (!!phase && phase !== 'paused')}
        >
          Send
        </button>
        </div>

        <div className="links">
          {conversationId && (
            <button type="button" className="link-btn" onClick={() => { setPanelTurnIndex(null); setPanelView('state') }}>
              View state
            </button>
          )}
          <button type="button" className="link-btn link-btn-new" onClick={handleStartNewChat}>
            Restart
          </button>
        </div>
      </div>

      {panelView && conversationId && (
        <>
          <div
            className="chat-panel-resize-handle"
            onMouseDown={handleResizeStart}
            title="Drag to resize"
          />
          <div className="chat-panel" style={{ width: panelWidth }}>
          {panelView === 'graph' && (
            <Graph
              conversationId={conversationId}
              turnIndex={panelTurnIndex}
              embedded
              onClose={() => { setPanelView(null); setPanelTurnIndex(null) }}
            />
          )}
          {panelView === 'state' && (
            <State
              conversationId={conversationId}
              embedded
              onClose={() => setPanelView(null)}
            />
          )}
          </div>
        </>
      )}
    </div>
  )
}

export default Chat
