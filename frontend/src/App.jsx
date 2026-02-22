import { Routes, Route, Link } from 'react-router-dom'
import Chat from './pages/Chat'
import Graph from './pages/Graph'
import Documents from './pages/Documents'
import State from './pages/State'
import './App.css'

function App() {
  return (
    <div className="app">
      <nav className="nav">
        <Link to="/">Chat</Link>
        <Link to="/documents">Documents</Link>
      </nav>
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

export default App
