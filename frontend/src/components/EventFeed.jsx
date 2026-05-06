import { Lock, CheckCircle2, XCircle, AlertTriangle } from 'lucide-react'
import { useState } from 'react'

function StepCard({ step }) {
  const [open, setOpen] = useState(false)

  const icons = {
    3: '📡', 4: '🖥️', 5: '⛓️', 6: '🔐', 7: '📋'
  }

  const actionColor = (action) => {
    if (!action) return ''
    if (action.includes('DISCARD') || action.includes('REJECT')) return 'red'
    if (action.includes('REVIEW') || action.includes('APPROVED')) return 'yellow'
    if (action.includes('DENIED')) return 'red'
    return 'green'
  }

  return (
    <div className="step-card" onClick={() => setOpen(o => !o)} style={{ cursor: 'pointer' }}>
      <div className="step-header">
        <span className="step-icon">{icons[step.step] || '•'}</span>
        <span className="step-title">Step {step.step}: {step.title}</span>
        {step.action && (
          <span className={`badge badge-${actionColor(step.action)} small`}>{step.action}</span>
        )}
        <span className="dim">{open ? '▲' : '▼'}</span>
      </div>

      {open && (
        <div className="step-body">
          {/* Step 3 */}
          {step.detection && (
            <div className="detection-box">
              <div className="det-row"><span className="dim">Detection:</span><strong>{step.detection}</strong></div>
              <div className="det-row"><span className="dim">Confidence:</span><strong>{step.confidence}</strong></div>
              <div className="det-row"><span className="dim">Zone:</span><strong>{step.zone}</strong></div>
              <div className="det-row"><span className="dim">Source:</span><strong>{step.source}</strong></div>
              <div className="det-row"><span className="dim">→ Emergency Type:</span><strong>{step.emergency_type}</strong></div>
            </div>
          )}

          {/* Step 4 */}
          {step.penalty_applied !== undefined && (
            <div className="detection-box">
              <div className="det-row"><span className="dim">Penalty Applied:</span><span className="red">-{step.penalty_applied}</span></div>
              <div className="det-row"><span className="dim">Zone Score After:</span><strong>{step.zone_score_after}</strong></div>
              <div className="det-row"><span className="dim">Event Count:</span><strong>{step.event_count}</strong></div>
            </div>
          )}
          {step.reason && <div className="step-note">{step.reason}</div>}

          {/* Step 5 */}
          {step.tx_hash && step.step === 5 && (
            <div className="detection-box">
              <div className="det-row"><span className="dim">Emergency Type:</span><strong>{step.emergency_type}</strong></div>
              <div className="det-row"><span className="dim">Zone Score:</span><strong>{step.zone_score}</strong></div>
              <div className="det-row">
                <span className="dim">TX:</span>
                <a href={`https://sepolia.etherscan.io/tx/${step.tx_hash}`} target="_blank" rel="noreferrer" className="small" style={{color: '#3498db', textDecoration: 'underline'}}>
                  {step.tx_hash?.substring(0,24)}...
                </a>
              </div>
              <div className="det-row"><span className="dim">Block:</span><strong>{step.block}</strong></div>
              {step.event_emitted && (
                <div className="det-row"><span className="dim">EmergencyAccessGranted:</span>
                  <span className="green">Type: {step.event_emitted.type} | Score: {step.event_emitted.score}</span>
                </div>
              )}
            </div>
          )}

          {/* Step 6 */}
          {step.step === 6 && (
            <div className="detection-box">
              {step.authorized ? (
                <>
                  <div className="det-row"><span className="green">✓ Blockchain authorization verified!</span></div>
                  <div className="det-row"><span className="dim">Authorized Type:</span><strong>{step.authorized_type}</strong></div>
                  <div className="det-row"><span className="dim">→ RELEASED to emergency responder:</span></div>
                  {step.released_fields && Object.entries(step.released_fields).map(([k, v]) => (
                    <div key={k} className="det-row released-row">
                      <span className="green">✓</span>
                      <span className="dim">{k}:</span>
                      <span>{String(v)}</span>
                    </div>
                  ))}
                  <div className="det-row"><span className="red">✗ WITHHELD (privacy protected):</span></div>
                  <div className="det-row dim small">{step.withheld_message}</div>
                </>
              ) : (
                <>
                  <div className="det-row red">✗ AUTHORIZATION DENIED by smart contract</div>
                  <div className="det-row dim">The ADM will NOT decrypt without blockchain proof.</div>
                  <div className="det-row dim">This is the privacy gatekeeper working.</div>
                </>
              )}
            </div>
          )}

          {/* Step 7 */}
          {step.step === 7 && (
            <div className="detection-box">
              {step.tx_hash && (
                <div className="det-row">
                  <span className="dim">TX:</span>
                  <a href={`https://sepolia.etherscan.io/tx/${step.tx_hash}`} target="_blank" rel="noreferrer" className="small" style={{color: '#3498db', textDecoration: 'underline'}}>
                    {step.tx_hash?.substring(0,24)}...
                  </a>
                </div>
              )}
              {step.block && <div className="det-row"><span className="dim">Block:</span><strong>{step.block}</strong></div>}
              {step.confirmed && <div className="det-row green">✓ ReleaseConfirmed — immutable proof of data access recorded on-chain.</div>}
            </div>
          )}

          {/* IoT Step 5 */}
          {step.step === 5 && step.message && !step.tx_hash && (
            <div className="detection-box">
              <div className="det-row"><span className="dim">Zone:</span><strong>{step.zone}</strong></div>
              <div className="det-row"><span className="dim">Event:</span><strong>{step.event_type}</strong></div>
              <div className="det-row"><span className="dim">Emergency Type:</span><strong>{step.emergency_type}</strong></div>
              <div className="det-row"><span className="dim">Zone Score:</span><strong>{step.zone_score}</strong></div>
              <div className="step-note">{step.message}</div>
            </div>
          )}

          {step.error && <div className="error-box">{step.error}</div>}
        </div>
      )}
    </div>
  )
}

export default function EventFeed({ events }) {
  return (
    <div className="panel panel-feed">
      <div className="panel-title">Detection Feed</div>
      {events.length === 0 ? (
        <div className="empty-state">No events yet. Trigger a detection to see the full pipeline here.</div>
      ) : (
        <div className="feed-list">
          {events.map((ev, idx) => (
            <div key={idx} className={`feed-card ${(ev.action?.includes('DISCARD') || ev.action?.includes('REJECT')) ? 'discarded' : ev.action?.includes('EMERGENCY') || ev.action === 'COMPLETE' || ev.action === 'IOT_ZONE_ALERT' ? 'emergency' : 'review'}`}>
              <div className="feed-card-header">
                <span className="feed-type">{ev.event_type}</span>
                <span className="dim small">conf: {ev.confidence?.toFixed ? ev.confidence.toFixed(2) : ev.confidence}</span>
                <span className={`badge ${ev.action?.includes('DISCARD') ? 'badge-red' : ev.action?.includes('REVIEW') ? 'badge-yellow' : 'badge-green'} small`}>
                  {ev.final_action || ev.action}
                </span>
              </div>
              <div className="feed-card-meta">
                <span className="dim small">Zone: {ev.zone_id}</span>
                <span className="dim small">| {ev.source}</span>
              </div>
              {ev.steps && ev.steps.map((step, si) => (
                <StepCard key={si} step={step}/>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
