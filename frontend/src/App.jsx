import { useState, useEffect } from 'react'
import { ShieldAlert, Wifi, WifiOff, RefreshCw, UserPlus, RotateCcw, Activity, Settings } from 'lucide-react'
import EnrollModal from './components/EnrollModal'
import ZoneMap from './components/ZoneMap'
import EventFeed from './components/EventFeed'
import SimulatePanel from './components/SimulatePanel'
import ReviewPanel from './components/ReviewPanel'
import AdminPanel from './components/AdminPanel'

const API = 'http://localhost:5000/api'

export default function App() {
  const [status, setStatus] = useState(null)
  const [zones, setZones] = useState({})
  const [events, setEvents] = useState([])
  const [queue, setQueue] = useState([])
  const [tourists, setTourists] = useState([])
  const [activeTab, setActiveTab] = useState('simulate')
  const [showEnroll, setShowEnroll] = useState(false)
  const [resetting, setResetting] = useState(false)

  const fetchAll = async () => {
    try {
      const [sRes, zRes, eRes, tRes] = await Promise.all([
        fetch(`${API}/status`),
        fetch(`${API}/zones`),
        fetch(`${API}/events`),
        fetch(`${API}/tourists`),
      ])
      setStatus(await sRes.json())
      setZones(await zRes.json())
      const eData = await eRes.json()
      setEvents([...(eData.processed || [])].reverse())
      setQueue(eData.queue || [])
      const tData = await tRes.json()
      setTourists(tData.tourists || [])
    } catch (err) {
      console.error('Fetch error:', err)
    }
  }

  const handleResetZones = async () => {
    setResetting(true)
    try {
      await fetch(`${API}/zones/reset`, { method: 'POST' })
      await fetchAll()
    } catch (err) {
      console.error('Reset error:', err)
    }
    setResetting(false)
  }

  useEffect(() => {
    fetchAll()
    const id = setInterval(fetchAll, 3000)
    return () => clearInterval(id)
  }, [])

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="logo">
          <ShieldAlert size={26} />
          <span>STSMIRS</span>
          <span className="subtitle">Smart Tourism Safety System</span>
        </div>

        <div className="status-bar">
          {status?.connected ? (
            <>
              <span className="badge badge-green"><Wifi size={13}/> Sepolia Connected</span>
              <span className="status-text">
                <span className="dim">Account:</span> {status.account?.slice(0,10)}...
              </span>
              <span className="status-text">
                <span className="dim">Balance:</span> {status.balance} ETH
              </span>
            </>
          ) : (
            <span className="badge badge-red"><WifiOff size={13}/> Disconnected</span>
          )}
          <button className="btn-icon" onClick={fetchAll} title="Refresh"><RefreshCw size={15}/></button>
          <button
            className="btn"
            onClick={handleResetZones}
            disabled={resetting}
            title="Reset all zone scores to 100"
            style={{ borderColor: 'rgba(239,68,68,0.5)', color: '#ef4444' }}
          >
            <RotateCcw size={15}/> {resetting ? 'Resetting...' : 'Reset Zones'}
          </button>
          <button className="btn btn-primary" onClick={() => setShowEnroll(true)}>
            <UserPlus size={15}/> Enroll Tourist
          </button>
        </div>
      </header>

      {/* Main grid */}
      <div className="main-grid">
        {/* LEFT: Controls */}
        <div className="left-col">
          <div className="tab-bar">
            <button className={`tab ${activeTab==='simulate'?'active':''}`} onClick={()=>setActiveTab('simulate')}>
              <Activity size={15}/> Simulate
            </button>
            <button className={`tab ${activeTab==='admin'?'active':''}`} onClick={()=>setActiveTab('admin')}>
              <Settings size={15}/> Admin
            </button>
          </div>

          {activeTab === 'simulate' && (
            <SimulatePanel tourists={tourists} onDone={fetchAll} api={API}/>
          )}
          {activeTab === 'admin' && (
            <AdminPanel zones={zones} events={events} api={API} onDone={fetchAll}/>
          )}

          {queue.length > 0 && (
            <ReviewPanel queue={queue} api={API} onDone={fetchAll}/>
          )}
        </div>

        {/* CENTER: Zone Map */}
        <div className="center-col">
          <ZoneMap zones={zones}/>
        </div>

        {/* RIGHT: Event Feed */}
        <div className="right-col">
          <EventFeed events={events}/>
        </div>
      </div>

      {showEnroll && (
        <EnrollModal api={API} onClose={() => { setShowEnroll(false); fetchAll() }}/>
      )}
    </div>
  )
}
