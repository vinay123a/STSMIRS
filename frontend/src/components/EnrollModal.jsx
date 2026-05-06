import { useState } from 'react'
import { CheckCircle2, Loader } from 'lucide-react'

export default function EnrollModal({ api, onClose }) {
  const [form, setForm] = useState({
    name: '', phone_number: '', emergency_contact: '',
    medical_history: '', past_police_records: '',
    nationality: '', passport_number: '', hotel: ''
  })
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${api}/enroll`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(form)
      })
      const data = await res.json()
      if (data.error) setError(data.error)
      else setResult(data)
    } catch (err) {
      setError(String(err))
    }
    setLoading(false)
  }

  const fields = [
    { key: 'name', label: 'Full Name', required: true },
    { key: 'phone_number', label: 'Phone Number' },
    { key: 'emergency_contact', label: 'Emergency Contact Number' },
    { key: 'medical_history', label: 'Medical History' },
    { key: 'past_police_records', label: 'Past Police Records' },
    { key: 'nationality', label: 'Nationality' },
    { key: 'passport_number', label: 'Passport Number' },
    { key: 'hotel', label: 'Hotel / Accommodation' },
  ]

  return (
    <div className="modal-overlay">
      <div className="modal">
        <div className="modal-header">
          <span>Enroll New Tourist</span>
          <button className="btn-icon" onClick={onClose}>✕</button>
        </div>

        {!result ? (
          <form onSubmit={handleSubmit} className="modal-body">
            <p className="dim small">All fields encrypted with AES-256-GCM + RSA-4096 before storage.</p>
            {fields.map(f => (
              <div className="field-group" key={f.key}>
                <label>{f.label}{f.required && <span className="required">*</span>}</label>
                <input
                  className="input"
                  required={f.required}
                  value={form[f.key]}
                  onChange={e => setForm({ ...form, [f.key]: e.target.value })}
                />
              </div>
            ))}
            {error && <div className="error-box">{error}</div>}
            <button type="submit" className="btn btn-primary w-full" disabled={loading}>
              {loading ? <><Loader size={15} className="spin"/> Encrypting & Enrolling On-Chain...</> : 'Encrypt & Enroll On-Chain'}
            </button>
          </form>
        ) : (
          <div className="modal-body">
            <div className="success-row"><CheckCircle2 size={20} color="var(--green)"/> Enrollment Successful!</div>
            <div className="result-box">
              <div className="result-row"><span className="dim">adm_ref (UUID):</span><code>{result.adm_ref}</code></div>
              <div className="result-row"><span className="dim">id_hash (on-chain):</span><code className="small">{result.id_hash?.substring(0,18)}...</code></div>
              <div className="result-row"><span className="dim">TX Hash:</span><code className="small">{result.tx_hash?.substring(0,18)}...</code></div>
              <div className="result-row"><span className="dim">Block:</span><span>{result.block}</span></div>
              <div className="result-row"><span className="dim">On-Chain Verified:</span><span>{result.on_chain_adm_ref_preview}</span></div>
            </div>
            <p className="dim small">This UUID is now available in the simulation panel for detection testing.</p>
            <button className="btn btn-primary w-full" onClick={onClose}>Done</button>
          </div>
        )}
      </div>
    </div>
  )
}
