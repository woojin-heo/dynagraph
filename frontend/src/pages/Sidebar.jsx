import { useState, useEffect } from 'react'
import { getConfig } from '../api'
import './Sidebar.css'

function Section({ title, defaultOpen = false, children }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="sidebar-section">
      <button
        type="button"
        className="sidebar-section-header"
        onClick={() => setOpen((o) => !o)}
      >
        <span className={`sidebar-chevron${open ? ' open' : ''}`}>&#9654;</span>
        {title}
      </button>
      {open && <div className="sidebar-section-body">{children}</div>}
    </div>
  )
}

function Sidebar({ visible, onClose, width = 260 }) {
  const [config, setConfig] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!visible) return
    let cancelled = false
    setError(null)

    getConfig()
      .then((cfg) => { if (!cancelled) setConfig(cfg) })
      .catch(() => { if (!cancelled) setError('Failed to load configuration') })

    return () => { cancelled = true }
  }, [visible])

  if (!visible) return null

  const llmActions = config?.actions?.filter((a) => a.kind === 'llm') || []
  const toolActions = config?.actions?.filter((a) => a.kind === 'tool') || []
  const hitlBlocks = config?.actions?.filter((a) => a.hitl_enabled) || []

  return (
    <div className="sidebar" style={{ width }}>
      <div className="sidebar-header">
        <span className="sidebar-title">Settings</span>
        <button type="button" className="sidebar-close" onClick={onClose} title="Close sidebar">&times;</button>
      </div>

      {error && <div className="sidebar-error">{error}</div>}

      <div className="sidebar-body">
        <Section title="System" defaultOpen>
          {config ? (
            <div className="sidebar-kv-list">
              <div className="sidebar-kv">
                <span className="sidebar-kv-label">Default model</span>
                <span className="sidebar-kv-value">{config.default_model}</span>
              </div>
              <div className="sidebar-kv">
                <span className="sidebar-kv-label">Temperature</span>
                <span className="sidebar-kv-value">{config.default_temperature}</span>
              </div>
            </div>
          ) : (
            <span className="sidebar-loading">Loading...</span>
          )}
        </Section>

        <Section title="Action Blocks" defaultOpen>
          {config ? (
            <>
              <div className="sidebar-sub-label">LLM Actions</div>
              {llmActions.map((a) => (
                <div key={a.action_type} className="sidebar-action-card">
                  <div className="sidebar-action-name">{a.action_type}</div>
                  <div className="sidebar-action-desc">{a.description}</div>
                  <div className="sidebar-action-meta">
                    <span className="sidebar-badge badge-llm">LLM</span>
                    <span className="sidebar-action-model">{a.llm_model}</span>
                    {a.hitl_enabled && <span className="sidebar-badge badge-hitl">HITL</span>}
                  </div>
                </div>
              ))}
              <div className="sidebar-sub-label" style={{ marginTop: '0.75rem' }}>Tool Actions</div>
              {toolActions.map((a) => (
                <div key={a.action_type} className="sidebar-action-card">
                  <div className="sidebar-action-name">{a.action_type}</div>
                  <div className="sidebar-action-desc">{a.description}</div>
                  <div className="sidebar-action-meta">
                    <span className="sidebar-badge badge-tool">Tool</span>
                    {a.hitl_enabled && <span className="sidebar-badge badge-hitl">HITL</span>}
                  </div>
                </div>
              ))}
            </>
          ) : (
            <span className="sidebar-loading">Loading...</span>
          )}
        </Section>

        <Section title="HITL (Human-in-the-Loop)">
          {config ? (
            hitlBlocks.length > 0 ? (
              <div className="sidebar-hitl-list">
                <div className="sidebar-kv">
                  <span className="sidebar-kv-label">Status</span>
                  <span className="sidebar-badge badge-hitl">Enabled</span>
                </div>
                <div className="sidebar-sub-label" style={{ marginTop: '0.5rem' }}>Interrupt before</div>
                {hitlBlocks.map((a) => (
                  <div key={a.action_type} className="sidebar-hitl-item">
                    <span className="sidebar-badge badge-hitl">&#9632;</span>
                    <span>{a.action_type}</span>
                  </div>
                ))}
              </div>
            ) : (
              <span className="sidebar-muted">No HITL blocks configured</span>
            )
          ) : (
            <span className="sidebar-loading">Loading...</span>
          )}
        </Section>

      </div>
    </div>
  )
}

export default Sidebar
