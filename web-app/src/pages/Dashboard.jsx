import { useState, useEffect, useRef, useCallback } from "react"

/* ─────────────────────────────────────────────
   Reusable small components
───────────────────────────────────────────── */
function StatusDot({ active, color = "#22c55e" }) {
  return (
    <span style={{ position: "relative", display: "inline-flex", alignItems: "center" }}>
      {active && (
        <span style={{
          position: "absolute",
          inset: -2,
          borderRadius: "50%",
          background: color,
          opacity: 0.35,
          animation: "pvDotRing 1.6s ease infinite",
        }} />
      )}
      <span style={{
        width: 8, height: 8,
        borderRadius: "50%",
        background: active ? color : "#374151",
        display: "inline-block",
      }} />
    </span>
  )
}

function StatCard({ icon, label, value, accent = "#3b5bdb" }) {
  return (
    <div style={{
      background: "#111827",
      border: "1px solid #1f2937",
      borderRadius: 14,
      padding: "16px 20px",
      display: "flex",
      alignItems: "center",
      gap: 14,
      flex: "1 1 140px",
      transition: "border-color 0.2s, transform 0.2s",
    }}
      onMouseEnter={e => { e.currentTarget.style.borderColor = accent; e.currentTarget.style.transform = "translateY(-2px)" }}
      onMouseLeave={e => { e.currentTarget.style.borderColor = "#1f2937"; e.currentTarget.style.transform = "translateY(0)" }}
    >
      <div style={{
        width: 40, height: 40,
        borderRadius: 10,
        background: `${accent}22`,
        display: "flex", alignItems: "center", justifyContent: "center",
        fontSize: 18, flexShrink: 0,
      }}>{icon}</div>
      <div>
        <div style={{ fontSize: 11, color: "#6b7280", letterSpacing: "0.5px", textTransform: "uppercase", fontWeight: 600 }}>{label}</div>
        <div style={{ fontSize: 18, color: "#f9fafb", fontWeight: 700, marginTop: 2 }}>{value}</div>
      </div>
    </div>
  )
}

/* ─────────────────────────────────────────────
   DASHBOARD
───────────────────────────────────────────── */
export default function Dashboard({ user, onLogout }) {
  const [cameraType, setCameraType] = useState("internal")
  const [ip, setIp]                 = useState("")
  const [stream, setStream]         = useState(null)
  const [isStarting, setIsStarting] = useState(false)
  const [feedReady, setFeedReady]   = useState(false)   // FIX 1: track when img actually loaded
  const [isMounted, setIsMounted]   = useState(false)
  const [currentTime, setCurrentTime] = useState(new Date())
  const imgRef = useRef(null)

  useEffect(() => { setTimeout(() => setIsMounted(true), 40) }, [])

  // Live clock
  useEffect(() => {
    const t = setInterval(() => setCurrentTime(new Date()), 1000)
    return () => clearInterval(t)
  }, [])

  const startCamera = async () => {
    setIsStarting(true)
    setFeedReady(false)   // hide img until it confirms first frame
    try {
      await fetch("http://localhost:5000/start_camera", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ source: cameraType === "external" ? ip : "0" }),
      })
      // FIX 1: small grace period so Flask opens the capture before the browser
      // tries to fetch the MJPEG stream — prevents the blank/stalled image on start
      await new Promise(r => setTimeout(r, 600))
      setStream("http://localhost:5000/video?" + Date.now())
    } catch {
      alert("Failed to connect to camera. Ensure the Flask server is running.")
      setIsStarting(false)
    }
    // isStarting cleared in onLoad / onError on the <img>
  }

  // FIX 3: actually tell Flask to release the camera — not just clear React state
  const stopFeed = useCallback(async () => {
    setStream(null)
    setFeedReady(false)
    try {
      await fetch("http://localhost:5000/stop_camera", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      })
    } catch {
      // server may already be down; ignore
    }
  }, [])

  // FIX 3: stop camera feed when user logs out so it doesn't stay alive in background
  const handleLogout = useCallback(async () => {
    if (stream) await stopFeed()
    onLogout()
  }, [stream, stopFeed, onLogout])

  const timeStr = currentTime.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false })
  const dateStr = currentTime.toLocaleDateString([], { weekday: "short", year: "numeric", month: "short", day: "numeric" })

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        html, body { height: 100%; overflow-y: auto; }

        @keyframes pvSlideUp {
          from { opacity: 0; transform: translateY(32px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes pvDotRing {
          0%  { transform: scale(1);   opacity: 0.35; }
          70% { transform: scale(2.2); opacity: 0; }
          100%{ transform: scale(2.2); opacity: 0; }
        }
        @keyframes pvScanFeed {
          0%   { top: 0%; opacity: 0.5; }
          100% { top: 100%; opacity: 0; }
        }
        @keyframes pvFeedPulse {
          0%,100% { box-shadow: 0 0 0 0 rgba(59,91,219,0.4); }
          50%     { box-shadow: 0 0 0 8px rgba(59,91,219,0); }
        }
        @keyframes pvRecBlink {
          0%,100% { opacity: 1; }
          50%     { opacity: 0.25; }
        }
        @keyframes pvStagger1 { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:none} }
        @keyframes pvStagger2 { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:none} }
        @keyframes pvStagger3 { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:none} }
        @keyframes spin        { to { transform: rotate(360deg); } }

        .pv-stagger-1 { animation: pvStagger1 0.55s cubic-bezier(0.4,0,0.2,1) 0.1s both; }
        .pv-stagger-2 { animation: pvStagger2 0.55s cubic-bezier(0.4,0,0.2,1) 0.22s both; }
        .pv-stagger-3 { animation: pvStagger3 0.55s cubic-bezier(0.4,0,0.2,1) 0.34s both; }

        .pv-db-input {
          width: 100%;
          padding: 11px 14px;
          background: #0f1117;
          border: 1.5px solid #1f2937;
          border-radius: 9px;
          color: #e5e7eb;
          font-family: 'DM Sans', sans-serif;
          font-size: 14px;
          outline: none;
          transition: border-color 0.2s, box-shadow 0.2s;
        }
        .pv-db-input:focus {
          border-color: #3b5bdb;
          box-shadow: 0 0 0 3px rgba(59,91,219,0.15);
        }
        .pv-db-input::placeholder { color: #4b5563; }

        .pv-db-select {
          padding: 11px 14px;
          background: #0f1117;
          border: 1.5px solid #1f2937;
          border-radius: 9px;
          color: #e5e7eb;
          font-family: 'DM Sans', sans-serif;
          font-size: 14px;
          cursor: pointer;
          outline: none;
          transition: border-color 0.2s;
          min-width: 200px;
        }
        .pv-db-select:focus { border-color: #3b5bdb; }

        .pv-primary-btn {
          padding: 12px 22px;
          background: linear-gradient(135deg, #1e2a78, #3b5bdb);
          color: #fff;
          border: none;
          border-radius: 10px;
          font-family: 'DM Sans', sans-serif;
          font-size: 14px;
          font-weight: 600;
          cursor: pointer;
          transition: transform 0.18s, box-shadow 0.18s, opacity 0.18s;
          letter-spacing: 0.4px;
          white-space: nowrap;
        }
        .pv-primary-btn:hover {
          transform: translateY(-2px);
          box-shadow: 0 8px 22px rgba(59,91,219,0.4);
        }
        .pv-primary-btn:active  { transform: translateY(0); }
        .pv-primary-btn:disabled{ opacity: 0.6; cursor: not-allowed; transform: none; }

        .pv-ghost-btn {
          padding: 11px 18px;
          background: transparent;
          color: #9ca3af;
          border: 1.5px solid #1f2937;
          border-radius: 10px;
          font-family: 'DM Sans', sans-serif;
          font-size: 14px;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.18s;
        }
        .pv-ghost-btn:hover {
          border-color: #374151;
          color: #e5e7eb;
          background: #111827;
        }

        .pv-logout-btn {
          padding: 8px 16px;
          background: transparent;
          color: #6b7280;
          border: 1.5px solid #1f2937;
          border-radius: 8px;
          font-family: 'DM Sans', sans-serif;
          font-size: 13px;
          font-weight: 500;
          cursor: pointer;
          transition: all 0.18s;
        }
        .pv-logout-btn:hover {
          border-color: #ef4444;
          color: #ef4444;
          background: rgba(239,68,68,0.07);
        }

        .pv-card {
          background: #111827;
          border: 1px solid #1f2937;
          border-radius: 18px;
          padding: 24px 26px;
        }
        .pv-section-title {
          font-size: 12px;
          font-weight: 700;
          letter-spacing: 1.2px;
          text-transform: uppercase;
          color: #6b7280;
          margin-bottom: 16px;
          display: flex;
          align-items: center;
          gap: 8px;
        }
      `}</style>

      <div style={{
        minHeight: "100vh",
        overflowY: "auto",          /* FIX 2: allow page to scroll */
        background: "#090d1f",
        backgroundImage: `
          radial-gradient(ellipse at 10% 5%,  rgba(59,91,219,0.12) 0%, transparent 45%),
          radial-gradient(ellipse at 90% 95%, rgba(10,14,39,0.8)   0%, transparent 45%)
        `,
        color: "#e5e7eb",
        fontFamily: "'DM Sans', sans-serif",
        /* FIX 2: animation removed from root — it created a transform stacking context
           that clipped overflow and prevented scrolling. Animation now on inner wrapper. */
      }}>

        {/* ════ TOPBAR ════ */}
        <header style={{
          borderBottom: "1px solid #1f2937",
          background: "rgba(9,13,31,0.9)",
          backdropFilter: "blur(12px)",
          position: "sticky", top: 0, zIndex: 100,
          padding: "0 28px",
        }}>
          <div style={{
            maxWidth: 1280,
            margin: "0 auto",
            height: 62,
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}>
            {/* Logo */}
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div style={{
                width: 34, height: 34,
                background: "linear-gradient(145deg, #0a0e27, #3b5bdb)",
                borderRadius: 9,
                display: "flex", alignItems: "center", justifyContent: "center",
                boxShadow: "0 4px 12px rgba(59,91,219,0.3)",
              }}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                  <ellipse cx="12" cy="12" rx="10" ry="7" fill="none" stroke="rgba(255,255,255,0.85)" strokeWidth="1.5"/>
                  <circle cx="12" cy="12" r="4" fill="rgba(255,255,255,0.85)"/>
                  <circle cx="12" cy="12" r="2" fill="#0a0e27"/>
                  <circle cx="14" cy="10" r="0.9" fill="rgba(255,255,255,0.6)"/>
                </svg>
              </div>
              <div>
                <div style={{
                  fontFamily: "'Orbitron', sans-serif",
                  fontWeight: 700,
                  fontSize: 17,
                  letterSpacing: "0.3px",
                  background: "linear-gradient(90deg, #fff, #93a8f4)",
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                }}>VisionSentinel</div>
                <div style={{ fontSize: 10, color: "#4b5563", letterSpacing: "0.8px", fontFamily: "monospace" }}>
                  SENTINEL AI ENGINE
                </div>
              </div>
            </div>

            {/* Centre: live clock */}
            <div style={{ textAlign: "center", display: "flex", alignItems: "center", gap: 10 }}>
              <StatusDot active color="#22c55e" />
              <span style={{
                fontFamily: "monospace",
                fontSize: 14,
                color: "#9ca3af",
                letterSpacing: "1px",
              }}>
                {timeStr}
              </span>
              <span style={{ color: "#374151", fontSize: 12 }}>|</span>
              <span style={{ fontSize: 12, color: "#4b5563" }}>{dateStr}</span>
            </div>

            {/* Right: user + logout */}
            <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
              <div style={{ textAlign: "right" }}>
                <div style={{ fontSize: 13, color: "#e5e7eb", fontWeight: 500 }}>
                  {user}
                </div>
                <div style={{ fontSize: 11, color: "#6b7280" }}>Administrator</div>
              </div>
              <div style={{
                width: 34, height: 34,
                borderRadius: "50%",
                background: "linear-gradient(135deg, #1e2a78, #3b5bdb)",
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 14, color: "#fff", fontWeight: 700,
                flexShrink: 0,
              }}>
                {user ? user[0].toUpperCase() : "?"}
              </div>
              <button className="pv-logout-btn" onClick={handleLogout}>
                Logout
              </button>
            </div>
          </div>
        </header>

        {/* ════ MAIN CONTENT ════ */}
        {/* FIX 2: animation lives here on the content div, not the scroll root */}
        <main style={{
          maxWidth: 1280, margin: "0 auto", padding: "32px 28px 60px",
          animation: isMounted ? "pvSlideUp 0.65s cubic-bezier(0.4,0,0.2,1) both" : "none",
        }}>

          {/* Page heading */}
          <div className="pv-stagger-1" style={{ marginBottom: 28 }}>
            <h1 style={{
              fontFamily: "'Orbitron', sans-serif",
              fontWeight: 700,
              fontSize: 24,
              color: "#f9fafb",
              letterSpacing: "0.3px",
              marginBottom: 6,
            }}>
              Surveillance Dashboard
            </h1>
            <p style={{ color: "#6b7280", fontSize: 14 }}>
              Real-time AI-powered exam proctoring and multi-camera monitoring
            </p>
          </div>

          {/* ── STAT CARDS ── */}
          <div className="pv-stagger-1" style={{
            display: "flex",
            gap: 14,
            flexWrap: "wrap",
            marginBottom: 28,
          }}>
            <StatCard icon="📡" label="Camera Status" value={stream ? "Live" : "Offline"} accent={stream ? "#22c55e" : "#ef4444"} />
            <StatCard icon="🎓" label="Session"       value="Active"  accent="#3b5bdb" />
            <StatCard icon="🧠" label="AI Engine"     value="Online"  accent="#8b5cf6" />
            <StatCard icon="🔐" label="Operator"      value={user || "—"} accent="#f59e0b" />
          </div>

          {/* ── CAMERA CONTROL CARD ── */}
          <div className="pv-stagger-2 pv-card" style={{ marginBottom: 22 }}>
            <div className="pv-section-title">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="#6b7280">
                <circle cx="12" cy="12" r="3"/><path d="M20.188 10.934l.33-.882a2 2 0 000-4.104l-.33-.882a2 2 0 00-1.856-1.256H5.668A2 2 0 003.812 5.07l-.33.882a2 2 0 000 4.104l.33.882A2 2 0 005.668 12h12.664a2 2 0 001.856-1.066z" fill="#6b7280"/>
              </svg>
              Camera Configuration
            </div>

            <div style={{ display: "flex", alignItems: "flex-end", gap: 14, flexWrap: "wrap" }}>
              <div>
                <label style={{ display: "block", fontSize: 11.5, fontWeight: 600, color: "#6b7280", letterSpacing: "0.5px", textTransform: "uppercase", marginBottom: 7 }}>
                  Source Type
                </label>
                <select
                  className="pv-db-select"
                  value={cameraType}
                  onChange={e => setCameraType(e.target.value)}
                >
                  <option value="internal">🎥  Internal Webcam</option>
                  <option value="external">📱  Mobile / IP Camera</option>
                </select>
              </div>

              {cameraType === "external" && (
                <div style={{ flex: 1, minWidth: 280 }}>
                  <label style={{ display: "block", fontSize: 11.5, fontWeight: 600, color: "#6b7280", letterSpacing: "0.5px", textTransform: "uppercase", marginBottom: 7 }}>
                    Stream URL
                  </label>
                  <input
                    className="pv-db-input"
                    type="text"
                    placeholder="e.g. http://192.168.1.100:8080/video"
                    value={ip}
                    onChange={e => setIp(e.target.value)}
                  />
                </div>
              )}

              <div style={{ display: "flex", gap: 10 }}>
                <button
                  className="pv-primary-btn"
                  onClick={startCamera}
                  disabled={isStarting || (cameraType === "external" && !ip.trim())}
                >
                  {isStarting ? "Connecting…" : stream ? "🔄 Restart Feed" : "▶ Start Feed"}
                </button>
                {stream && (
                  <button className="pv-ghost-btn" onClick={stopFeed}>
                    ■ Stop
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* ── LIVE FEED CARD ── */}
          <div className="pv-stagger-3 pv-card">
            <div className="pv-section-title" style={{ justifyContent: "space-between" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#6b7280" strokeWidth="2">
                  <rect x="2" y="7" width="20" height="15" rx="2"/><path d="M17 2l5 5-5 5"/>
                </svg>
                Live Surveillance Feed
              </div>
              {stream && (
                <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11 }}>
                  <span style={{
                    display: "inline-block",
                    width: 7, height: 7,
                    borderRadius: "50%",
                    background: "#ef4444",
                    animation: "pvRecBlink 1.2s ease infinite",
                  }} />
                  <span style={{ color: "#ef4444", fontFamily: "monospace", letterSpacing: "0.5px" }}>
                    REC
                  </span>
                </div>
              )}
            </div>

            {stream ? (
              <div style={{ position: "relative", display: "inline-block", width: "100%" }}>
                {/* Feed frame */}
                <div style={{
                  borderRadius: 12,
                  overflow: "hidden",
                  border: "2px solid #1f2937",
                  position: "relative",
                  animation: "pvFeedPulse 2.5s ease infinite",
                  background: "#000",
                }}>
                  {/* FIX 1: show skeleton until first frame confirmed via onLoad */}
                  {!feedReady && (
                    <div style={{
                      position: "absolute", inset: 0,
                      background: "#0a0e1a",
                      display: "flex", flexDirection: "column",
                      alignItems: "center", justifyContent: "center", gap: 12,
                      borderRadius: 12, zIndex: 2,
                    }}>
                      <div style={{
                        width: 36, height: 36,
                        border: "3px solid #1f2937",
                        borderTopColor: "#3b5bdb",
                        borderRadius: "50%",
                        animation: "spin 0.8s linear infinite",
                      }} />
                      <span style={{ fontSize: 13, color: "#6b7280" }}>Connecting to feed…</span>
                    </div>
                  )}
                  <img
                    ref={imgRef}
                    src={stream}
                    alt="Live camera feed"
                    onLoad={() => { setFeedReady(true); setIsStarting(false) }}
                    onError={() => { setFeedReady(false); setIsStarting(false) }}
                    style={{
                      width: "100%",
                      maxHeight: 520,
                      objectFit: "contain",
                      display: "block",
                      opacity: feedReady ? 1 : 0,   /* hide until loaded — no flash */
                      transition: "opacity 0.3s",
                    }}
                  />
                  {/* Scan line overlay */}
                  <div style={{
                    position: "absolute",
                    left: 0, right: 0,
                    height: 2,
                    background: "linear-gradient(90deg, transparent, rgba(59,91,219,0.5), transparent)",
                    animation: "pvScanFeed 3.5s linear infinite",
                    pointerEvents: "none",
                  }} />
                  {/* Corner overlays */}
                  {[{top:8,left:8}, {top:8,right:8}, {bottom:8,left:8}, {bottom:8,right:8}].map((pos, i) => (
                    <div key={i} style={{
                      position: "absolute", ...pos,
                      width: 16, height: 16,
                      borderColor: "#3b5bdb",
                      borderStyle: "solid",
                      borderWidth: pos.top !== undefined ? (pos.left !== undefined ? "2px 0 0 2px" : "2px 2px 0 0") : (pos.left !== undefined ? "0 0 2px 2px" : "0 2px 2px 0"),
                    }} />
                  ))}
                  {/* HUD overlay bottom */}
                  <div style={{
                    position: "absolute", bottom: 0, left: 0, right: 0,
                    padding: "10px 14px",
                    background: "linear-gradient(transparent, rgba(0,0,0,0.75))",
                    display: "flex", alignItems: "center", justifyContent: "space-between",
                  }}>
                    <span style={{ fontFamily: "monospace", fontSize: 11, color: "#9ca3af", letterSpacing: "0.5px" }}>
                      SENTINEL AI · LIVE
                    </span>
                    <span style={{ fontFamily: "monospace", fontSize: 11, color: "#6b7280" }}>
                      {timeStr}
                    </span>
                  </div>
                </div>
              </div>
            ) : (
              <div style={{
                borderRadius: 12,
                border: "2px dashed #1f2937",
                background: "#0a0e1a",
                height: 320,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                gap: 14,
                color: "#374151",
              }}>
                <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="#1f2937" strokeWidth="1.2">
                  <rect x="2" y="7" width="20" height="14" rx="2"/>
                  <path d="M16 3l4 4-4 4"/>
                  <circle cx="12" cy="14" r="3"/>
                </svg>
                <div style={{ textAlign: "center" }}>
                  <div style={{ fontSize: 15, fontWeight: 600, color: "#374151", marginBottom: 6 }}>No Active Feed</div>
                  <div style={{ fontSize: 13, color: "#1f2937" }}>
                    Select a camera source above and press <strong style={{ color: "#3b5bdb" }}>Start Feed</strong>
                  </div>
                </div>
              </div>
            )}
          </div>

        </main>
      </div>
    </>
  )
}