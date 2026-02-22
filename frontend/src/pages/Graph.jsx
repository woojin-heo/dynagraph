import { useMemo, useState, useEffect, memo, useRef, useId } from 'react'
import { useParams, Link } from 'react-router-dom'
import { ReactFlow, Background, Controls, useNodesState, useEdgesState } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import mermaid from 'mermaid'
import { getGraph } from '../api'
import './Graph.css'

mermaid.initialize({ startOnLoad: false })

function MermaidDiagram({ code }) {
  const id = useId().replace(/:/g, '')
  const containerRef = useRef(null)

  useEffect(() => {
    if (!code?.trim() || !containerRef.current) return
    let active = true
    const uniqueId = `mermaid-${id}-${Date.now()}`
    mermaid
      .render(uniqueId, code.trim())
      .then(({ svg }) => {
        if (active && containerRef.current) containerRef.current.innerHTML = svg
      })
      .catch((err) => {
        if (active && containerRef.current) {
          containerRef.current.innerHTML = `<p class="mermaid-error">Diagram error: ${err.message || String(err)}</p>`
        }
      })
    return () => { active = false }
  }, [code, id])

  return <div ref={containerRef} className="graph-mermaid-container" />
}

const NODE_WIDTH = 160
const NODE_HEIGHT = 48
const LAYER_GAP = 100
const NODE_GAP = 24
const START_END_WIDTH = 100
const START_END_HEIGHT = 40

const StartNode = memo(function StartNode({ data }) {
  return (
    <div className="langgraph-node langgraph-node-start" title={data?.label}>
      <span>{data?.label ?? '__start__'}</span>
    </div>
  )
})

const EndNode = memo(function EndNode({ data }) {
  return (
    <div className="langgraph-node langgraph-node-end" title={data?.label}>
      <span>{data?.label ?? '__end__'}</span>
    </div>
  )
})

const ActionNode = memo(function ActionNode({ data }) {
  const label = data?.label ?? ''
  const interruptBefore = data?.interruptBefore === true
  return (
    <div className="langgraph-node langgraph-node-action" title={label}>
      <div className="langgraph-node-name">{label}</div>
      {interruptBefore && (
        <>
          <div className="langgraph-node-divider" />
          <div className="langgraph-node-interrupt">_interrupt = before</div>
        </>
      )}
    </div>
  )
})

const nodeTypes = { start: StartNode, end: EndNode, action: ActionNode }

function buildLayeredLayout(apiNodes, apiEdges, interruptBeforeList = []) {
  if (!apiNodes?.length) return { flowNodes: [], flowEdges: [] }
  const interruptSet = new Set(interruptBeforeList || [])
  const nodeIds = new Set(apiNodes.map((n) => n.id))
  const nodeMap = Object.fromEntries(apiNodes.map((n) => [n.id, n]))
  const incoming = {}
  nodeIds.forEach((id) => { incoming[id] = [] })
  apiEdges.forEach((e) => {
    if (nodeIds.has(e.target)) incoming[e.target].push(e.source)
  })
  const layers = []
  const assigned = new Set()
  let remaining = new Set(nodeIds)
  const orderIndex = {}
  apiNodes.forEach((n, i) => { orderIndex[n.id] = i })
  while (remaining.size > 0) {
    const layer = []
    for (const id of remaining) {
      const preds = incoming[id]
      if (preds.every((p) => assigned.has(p))) layer.push(id)
    }
    if (layer.length === 0) break
    layer.sort((a, b) => orderIndex[a] - orderIndex[b])
    layers.push(layer)
    layer.forEach((id) => assigned.add(id))
    layer.forEach((id) => remaining.delete(id))
  }
  if (remaining.size > 0) layers.push([...remaining].sort((a, b) => orderIndex[a] - orderIndex[b]))

  const flowNodes = []
  const flowEdges = []

  const startY = 0
  flowNodes.push({
    id: '__start__',
    type: 'start',
    position: { x: -START_END_WIDTH / 2, y: startY },
    data: { label: '__start__' },
  })

  const layerHeight = NODE_HEIGHT + LAYER_GAP
  layers.forEach((layerIds, layerIndex) => {
    const y = startY + (layerIndex + 1) * layerHeight
    const totalW = layerIds.length * NODE_WIDTH + (layerIds.length - 1) * NODE_GAP
    let x = -totalW / 2 + NODE_WIDTH / 2
    layerIds.forEach((id) => {
      const apiNode = nodeMap[id]
      flowNodes.push({
        id,
        type: 'action',
        position: { x, y },
        data: {
          label: apiNode?.label || id,
          interruptBefore: interruptSet.has(id),
        },
      })
      if (layerIndex === 0) {
        flowEdges.push({ id: `e-__start__-${id}`, source: '__start__', target: id })
      }
      x += NODE_WIDTH + NODE_GAP
    })
  })

  const endY = startY + (layers.length + 1) * layerHeight
  flowNodes.push({
    id: '__end__',
    type: 'end',
    position: { x: -START_END_WIDTH / 2, y: endY },
    data: { label: '__end__' },
  })

  apiEdges.forEach((e, i) => {
    flowEdges.push({ id: `e-${e.source}-${e.target}-${i}`, source: e.source, target: e.target })
  })

  const lastLayer = layers[layers.length - 1] || []
  lastLayer.forEach((id) => {
    flowEdges.push({ id: `e-${id}-__end__`, source: id, target: '__end__' })
  })

  return { flowNodes, flowEdges }
}

function Graph({ conversationId: conversationIdProp, turnIndex, embedded, onClose }) {
  const { id: urlId } = useParams()
  const conversationId = conversationIdProp ?? urlId
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!conversationId) return
    getGraph(conversationId, turnIndex ?? undefined)
      .then(setData)
      .catch((e) => setError(e.message))
  }, [conversationId, turnIndex])

  const { flowNodes, flowEdges } = useMemo(() => {
    if (!data?.nodes) return { flowNodes: [], flowEdges: [] }
    return buildLayeredLayout(
      data.nodes,
      data.edges || [],
      data.interrupt_before || []
    )
  }, [data])

  const [nodes, setNodes, onNodesChange] = useNodesState(flowNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(flowEdges)

  useEffect(() => {
    setNodes(flowNodes)
    setEdges(flowEdges)
  }, [flowNodes, flowEdges, setNodes, setEdges])

  if (error) return <div className="page-error">{error}</div>
  if (!data) return <div className="page-loading">Loading…</div>

  const { plan, nodes: apiNodes, error: dataError, graph_mermaid } = data
  const hasGraph = apiNodes?.length > 0
  const useLangGraphMermaid = graph_mermaid && graph_mermaid.trim().length > 0

  return (
    <div className={`graph-page${embedded ? ' embedded' : ''}`}>
      <div className="graph-nav">
        {embedded && onClose ? (
          <button type="button" className="link-btn" onClick={onClose}>Close</button>
        ) : (
          <Link to={`/conversation/${conversationId}`}>← Back to Chat</Link>
        )}
      </div>
      <section className="plan-section">
        <h2>Plan</h2>
        {dataError && <p className="muted">{dataError}</p>}
        <pre className="plan-text">{plan || '(No plan)'}</pre>
      </section>
      <section className="graph-section">
        <h2>Graph</h2>
        {!hasGraph && !useLangGraphMermaid ? (
          <p className="muted">No nodes</p>
        ) : useLangGraphMermaid ? (
          <div className="graph-flow-container">
            <MermaidDiagram code={graph_mermaid} />
          </div>
        ) : (
          <div className="graph-flow-container">
            <ReactFlow
              nodes={nodes}
              edges={edges}
              nodeTypes={nodeTypes}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              fitView
              fitViewOptions={{ padding: 0.2 }}
              nodesDraggable={false}
              nodesConnectable={false}
              elementsSelectable={false}
              proOptions={{ hideAttribution: true }}
            >
              <Background />
              <Controls showInteractive={false} />
            </ReactFlow>
          </div>
        )}
      </section>
    </div>
  )
}

export default Graph
