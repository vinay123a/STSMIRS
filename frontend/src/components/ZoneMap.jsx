import { AlertTriangle } from 'lucide-react'

function scoreClass(score) {
  if (score >= 80) return 'safe'
  if (score >= 50) return 'warn'
  return 'danger'
}

export default function ZoneMap({ zones }) {
  const entries = Object.entries(zones)
  return (
    <div className="panel">
      <div className="panel-title">Live Zone Safety Map</div>
      {entries.length === 0 ? (
        <div className="empty-state">No zones active yet. Trigger an event to create a zone.</div>
      ) : (
        <div className="zone-grid">
          {entries.map(([zid, info]) => {
            const cls = scoreClass(info.score)
            return (
              <div key={zid} className={`zone-card zone-${cls}`}>
                <div className="zone-name">{zid}</div>
                <div className={`zone-score zone-score-${cls}`}>{info.score.toFixed(1)}</div>
                <div className="zone-meta">{info.event_count} event{info.event_count !== 1 ? 's' : ''}</div>
                {info.score < 50 && (
                  <div className="zone-emergency pulse">
                    <AlertTriangle size={13}/> EMERGENCY
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
