import { useState } from 'react'
import { Loader, Send } from 'lucide-react'

const AI_EVENTS = [
  { value: 'FALL', label: 'FALL  →  MEDICAL_PROBLEM' },
  { value: 'PANIC', label: 'PANIC  →  MEDICAL_EMERGENCY' },
  { value: 'AGGRESSION', label: 'AGGRESSION  →  VIOLENT_FIGHT' },
]

const IOT_EVENTS = [
  { value: 'SURGE', label: 'SURGE  →  MEDICAL_EMERGENCY' },
  { value: 'CROWD_DENSITY', label: 'CROWD_DENSITY  →  SMALL_FIGHT' },
]

export default function SimulatePanel({ tourists, api, onDone }) {
  const [source, setSource] = useState('AI_CAMERA')
  const [eventType, setEventType] = useState('FALL')
  const [confidence, setConfidence] = useState(0.95)
  const [zoneId, setZoneId] = useState('A')
  const [touristId, setTouristId] = useState('')
  const [loading, setLoading] = useState(false)
  const [lastResult, setLastResult] = useState(null)
  const [error, setError] = useState(null)

  // Verify UUID before full detection
  const [verifying, setVerifying] = useState(false)
  const [verifyResult, setVerifyResult] = useState(null)

  const handleSourceChange = (s) => {
    setSource(s)
    setEventType(s === 'AI_CAMERA' ? 'FALL' : 'SURGE')
    setVerifyResult(null)
  }

  const verifyUUID = async () => {
    if (!touristId.trim()) return
    setVerifying(true)
    setVerifyResult(null)
    try {
      const res = await fetch(`${api}/verify-uuid?adm_ref=${encodeURIComponent(touristId.trim())}`)
      const data = await res.json()
      setVerifyResult(data)
    } catch (e) {
      setVerifyResult({ error: String(e) })
    }
    setVerifying(false)
  }

  const handleDetect = async () => {
    setLoading(true)
    setError(null)
    setLastResult(null)
    try {
      const res = await fetch(`${api}/detect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source,
          event_type: eventType,
          confidence: parseFloat(confidence),
          zone_id: zoneId,
          tourist_id: source === 'AI_CAMERA' ? touristId : null,
        })
      })
      const data = await res.json()
      if (data.error) setError(data.error)
      else setLastResult(data)
      await onDone()
    } catch (e) {
      setError(String(e))
    }
    setLoading(false)
  }

  const events = source === 'AI_CAMERA' ? AI_EVENTS : IOT_EVENTS

  return (
    <div className="panel">
      <div className="panel-title">Simulate Detection — Step 3</div>

      {/* Source Selection */}
      <div className="field-group">
        <label>Detection Source</label>
        <div className="radio-group">
          <label className={`radio-opt ${source==='AI_CAMERA'?'active':''}`}>
            <input type="radio" value="AI_CAMERA" checked={source==='AI_CAMERA'} onChange={()=>handleSourceChange('AI_CAMERA')}/>
            AI Camera (Specific Person)
          </label>
          <label className={`radio-opt ${source==='IOT_SENSOR'?'active':''}`}>
            <input type="radio" value="IOT_SENSOR" checked={source==='IOT_SENSOR'} onChange={()=>handleSourceChange('IOT_SENSOR')}/>
            IoT Sensor (Zone-Wide)
          </label>
        </div>
      </div>

      {/* Event Type */}
      <div className="field-group">
        <label>What did the {source === 'AI_CAMERA' ? 'AI camera' : 'IoT sensor'} detect?</label>
        <select className="input" value={eventType} onChange={e => setEventType(e.target.value)}>
          {events.map(ev => (
            <option key={ev.value} value={ev.value}>{ev.label}</option>
          ))}
        </select>
      </div>

      {/* Confidence */}
      <div className="field-group">
        <label>Confidence Score: <strong>{parseFloat(confidence).toFixed(2)}</strong>
          &nbsp;<span className={`badge small ${confidence < 0.70 ? 'badge-red' : confidence < 0.90 ? 'badge-yellow' : 'badge-green'}`}>
            {confidence < 0.70 ? 'Will Discard' : confidence < 0.90 ? 'Human Review' : 'Auto-Process'}
          </span>
        </label>
        <input type="range" min="0.50" max="1.00" step="0.01"
          value={confidence} onChange={e => setConfidence(e.target.value)} />
        <div className="confidence-hints dim small">
          &lt;0.70 → Discard &nbsp;|&nbsp; 0.70–0.89 → Review &nbsp;|&nbsp; ≥0.90 → Auto
        </div>
      </div>

      {/* Zone ID */}
      <div className="field-group">
        <label>Zone ID</label>
        <input className="input" value={zoneId} onChange={e => setZoneId(e.target.value)} placeholder="ZONE_A"/>
      </div>

      {/* Tourist UUID (AI only) */}
      {source === 'AI_CAMERA' && (
        <div className="field-group">
          <label>Tourist UUID (adm_ref)</label>
          <div className="uuid-row">
            <select className="input" value={touristId} onChange={e => { setTouristId(e.target.value); setVerifyResult(null) }}>
              <option value="">— select or type below —</option>
              {tourists.map(t => <option key={t} value={t}>{t.substring(0,30)}...</option>)}
            </select>
          </div>
          <input className="input" style={{marginTop:'0.4rem'}} value={touristId}
            onChange={e => { setTouristId(e.target.value); setVerifyResult(null) }}
            placeholder="adm_ref:..." />
          <button className="btn" style={{marginTop:'0.4rem'}} onClick={verifyUUID} disabled={verifying || !touristId}>
            {verifying ? <Loader size={14} className="spin"/> : '✓'} Verify UUID On-Chain
          </button>
          {verifyResult && (
            verifyResult.error
              ? <div className="error-box">{verifyResult.error}</div>
              : <div className="success-row small">
                  ✓ Verified — on-chain admRef: {verifyResult.on_chain_adm_ref_preview}
                </div>
          )}
        </div>
      )}

      {error && <div className="error-box">{error}</div>}

      <button className="btn btn-primary w-full" onClick={handleDetect} disabled={loading}>
        {loading
          ? <><Loader size={15} className="spin"/> Running Pipeline (Blockchain may take ~30s)...</>
          : <><Send size={15}/> Trigger Detection Event</>}
      </button>

      {lastResult && (
        <div className="result-summary">
          <div className={`badge ${lastResult.final_action === 'COMPLETE' ? 'badge-green' : lastResult.final_action === 'DISCARDED' ? 'badge-red' : 'badge-yellow'}`}>
            {lastResult.final_action}
          </div>
          {lastResult.zone_score !== undefined && (
            <span className="dim small">Zone score: {lastResult.zone_score}</span>
          )}
          <span className="dim small">→ See Detection Feed for full step breakdown</span>
        </div>
      )}
    </div>
  )
}
