import { useState } from 'react'
import { Loader, AlertTriangle, CheckCircle2, XCircle } from 'lucide-react'

export default function ReviewPanel({ queue, api, onDone }) {
  const [loading, setLoading] = useState(null)
  const [results, setResults] = useState({})

  const handle = async (index, approve) => {
    setLoading(index)
    try {
      const res = await fetch(`${api}/review/${approve ? 'approve' : 'reject'}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ index })
      })
      const data = await res.json()
      setResults(r => ({ ...r, [index]: data }))
      await onDone()
    } catch (err) {
      console.error(err)
    }
    setLoading(null)
  }

  return (
    <div className="panel panel-review">
      <div className="panel-title yellow">
        <AlertTriangle size={16} className="pulse"/> Human Review Required ({queue.length})
      </div>
      {queue.map((ev, idx) => (
        <div key={idx} className="review-card">
          <div className="review-header">
            <span className="feed-type yellow">{ev.event_type}</span>
            <span className="dim small">conf: {ev.confidence?.toFixed(2)}</span>
          </div>
          <div className="review-meta dim small">
            Zone: {ev.zone_id} &nbsp;|&nbsp; Source: {ev.source}
            {ev.tourist_id && <>&nbsp;|&nbsp; ID: {ev.tourist_id?.substring(0,20)}...</>}
          </div>
          <p className="step-note">An officer must review this detection before it can proceed.</p>
          <div className="review-btns">
            <button
              className="btn btn-success"
              onClick={() => handle(idx, true)}
              disabled={loading === idx}
            >
              {loading === idx ? <Loader size={14} className="spin"/> : <CheckCircle2 size={14}/>}
              Approve
            </button>
            <button
              className="btn btn-danger"
              onClick={() => handle(idx, false)}
              disabled={loading === idx}
            >
              <XCircle size={14}/> Reject
            </button>
          </div>
        </div>
      ))}
    </div>
  )
}
