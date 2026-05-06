import { useState } from 'react'

export default function AdminPanel({ zones, events, api, onDone }) {
  const [editZone, setEditZone] = useState('')
  const [newScore, setNewScore] = useState('')
  const [msg, setMsg] = useState(null)
  const [resetting, setResetting] = useState(false)

  const handleReset = async () => {
    setResetting(true)
    setMsg(null)
    try {
      await fetch(`${api}/zones/reset`, { method: 'POST' })
      setMsg('✓ All zones reset to 100')
      await onDone()
    } catch (e) {
      setMsg('Error: ' + e)
    }
    setResetting(false)
  }

  const handleSetScore = async (e) => {
    e.preventDefault()
    setMsg(null)
    try {
      const res = await fetch(`${api}/zones/${editZone.trim()}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ score: parseFloat(newScore) })
      })
      const data = await res.json()
      setMsg(data.ok ? `✓ ${editZone} score set to ${newScore}` : 'Error updating')
      await onDone()
    } catch (e) {
      setMsg('Error: ' + e)
    }
  }

  return (
    <div className="panel">
      <div className="panel-title">Admin Mode — View Logs & Edit Zones</div>

      {/* Event Log History */}
      <div className="admin-section">
        <div className="admin-label">Event Log History</div>
        {events.length === 0
          ? <div className="dim small">No events recorded yet.</div>
          : events.map((ev, i) => (
            <div key={i} className="admin-log-row">
              <span className="dim small">{i + 1}.</span>
              <span className="small">{ev.event_type}</span>
              <span className="dim small">(conf: {typeof ev.confidence === 'number' ? ev.confidence.toFixed(1) : ev.confidence})</span>
              <span className="dim small">in {ev.zone_id}</span>
              <span className={`small ${ev.action?.includes('DISCARD') ? 'red' : 'green'}`}>→ {ev.final_action || ev.action}</span>
            </div>
          ))
        }
      </div>

      {/* Zone Scores */}
      <div className="admin-section">
        <div className="admin-label">Current Zones & Scores</div>
        {Object.entries(zones).length === 0
          ? <div className="dim small">No zones created yet.</div>
          : Object.entries(zones).map(([zid, info]) => (
            <div key={zid} className="admin-zone-row">
              <span className="small"><strong>{zid}</strong></span>
              <span className="dim small">score={info.score.toFixed(1)}</span>
              <span className="dim small">events={info.event_count}</span>
            </div>
          ))
        }
      </div>

      {/* Edit Zone Score */}
      <form className="admin-section" onSubmit={handleSetScore}>
        <div className="admin-label">Edit Zone Score (0–100)</div>
        <input className="input" placeholder="Zone ID (e.g. ZONE_A)" value={editZone}
          onChange={e => setEditZone(e.target.value)} required/>
        <input className="input" style={{marginTop:'0.4rem'}} type="number" placeholder="New score" min="0" max="100"
          value={newScore} onChange={e => setNewScore(e.target.value)} required/>
        <button className="btn btn-primary" style={{marginTop:'0.5rem'}} type="submit">Set Score</button>
      </form>

      {/* Reset All Zones */}
      <div className="admin-section">
        <div className="admin-label">Reset All Zones</div>
        <button className="btn btn-danger w-full" onClick={handleReset} disabled={resetting}>
          {resetting ? '⏳ Resetting...' : '↺ Reset All Zone Scores to 100'}
        </button>
      </div>

      {msg && <div className="step-note">{msg}</div>}
    </div>
  )
}
