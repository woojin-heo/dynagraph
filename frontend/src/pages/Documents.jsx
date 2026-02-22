import { useState, useEffect } from 'react'
import { getDocuments, getTables } from '../api'
import './Documents.css'

function Documents() {
  const [docs, setDocs] = useState([])
  const [tables, setTables] = useState([])
  const [error, setError] = useState(null)

  useEffect(() => {
    getDocuments()
      .then(setDocs)
      .catch((e) => setError(e.message))
  }, [])

  useEffect(() => {
    getTables()
      .then(setTables)
      .catch((e) => setError(e.message))
  }, [])

  if (error) return <div className="page-error">{error}</div>

  return (
    <div className="documents-page">
      <section className="doc-section">
        <h1>Vector DB documents</h1>
        {Array.isArray(docs) && docs.length === 0 ? (
          <p className="muted">No documents in the vector database.</p>
        ) : (
          <table className="doc-table">
            <thead>
              <tr>
                <th>Source</th>
                <th>Chunks</th>
                <th>Created</th>
              </tr>
            </thead>
            <tbody>
              {(docs || []).map((d, i) => (
                <tr key={i}>
                  <td>{d.source || '—'}</td>
                  <td>{d.chunk_count ?? '—'}</td>
                  <td>{d.created_at || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className="doc-section">
        <h1>SQL tables (tables.yaml)</h1>
        {Array.isArray(tables) && tables.length === 0 ? (
          <p className="muted">No table definitions. Add backend/db/tables.yaml to describe SQL tables.</p>
        ) : (
          <table className="doc-table">
            <thead>
              <tr>
                <th>Table</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>
              {(tables || []).map((t, i) => (
                <tr key={i}>
                  <td>{t.name || '—'}</td>
                  <td>{t.description || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  )
}

export default Documents
