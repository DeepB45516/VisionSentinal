import { useState, useEffect, useRef } from "react"

/* ─────────────────────────────────────────────
   CCTV Camera SVG – tracks cursor, averts on
   password focus for privacy humour
───────────────────────────────────────────── */
function CCTVCamera({ angle, isLookingAway, svgRef }) {
  const px = 150, py = 82          // pivot (ball-joint) centre in SVG coords
  const finalAngle = isLookingAway ? 115 : angle

  const transition = isLookingAway
    ? "transform 0.55s cubic-bezier(0.68,-0.55,0.265,1.55)"
    : "transform 0.08s linear"

  // Camera body geometry (local, pointing straight down from pivot)
  const armH = 22
  const bW = 64, bH = 82
  const bX = px - bW / 2
  const bY = py + armH
  const lensX = px
  const lensY = bY + bH - 20   // lens near bottom of body

  return (
    <svg
      ref={svgRef}
      width="300"
      height="215"
      viewBox="0 0 300 215"
      style={{ overflow: "visible", filter: "drop-shadow(0 12px 28px rgba(0,0,0,0.22))" }}
    >
      <defs>
        {/* Mount / arm metallic gradient */}
        <linearGradient id="pvMountGrad" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%"   stopColor="#c8c8c8" />
          <stop offset="35%"  stopColor="#787878" />
          <stop offset="65%"  stopColor="#606060" />
          <stop offset="100%" stopColor="#b0b0b0" />
        </linearGradient>

        {/* Pivot ball radial */}
        <radialGradient id="pvPivotGrad" cx="32%" cy="30%">
          <stop offset="0%"   stopColor="#d8d8d8" />
          <stop offset="55%"  stopColor="#888" />
          <stop offset="100%" stopColor="#3a3a3a" />
        </radialGradient>

        {/* Camera body */}
        <linearGradient id="pvBodyGrad" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%"   stopColor="#525252" />
          <stop offset="18%"  stopColor="#303030" />
          <stop offset="82%"  stopColor="#1e1e1e" />
          <stop offset="100%" stopColor="#3a3a3a" />
        </linearGradient>
        <linearGradient id="pvBodyTopGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%"   stopColor="rgba(255,255,255,0.12)" />
          <stop offset="100%" stopColor="rgba(255,255,255,0)" />
        </linearGradient>

        {/* Lens glass */}
        <radialGradient id="pvLensGrad" cx="32%" cy="28%">
          <stop offset="0%"   stopColor="#6898d0" />
          <stop offset="35%"  stopColor="#1c3870" />
          <stop offset="75%"  stopColor="#0a1838" />
          <stop offset="100%" stopColor="#040412" />
        </radialGradient>

        {/* Lens ring bezel */}
        <radialGradient id="pvRingGrad" cx="40%" cy="38%">
          <stop offset="0%"   stopColor="#666" />
          <stop offset="100%" stopColor="#222" />
        </radialGradient>

        {/* Drop shadow for whole camera */}
        <filter id="pvCamShadow" x="-40%" y="-40%" width="180%" height="180%">
          <feDropShadow dx="0" dy="6" stdDeviation="8" floodColor="rgba(0,0,0,0.45)" />
        </filter>
      </defs>

      {/* ── WALL MOUNT PLATE ── */}
      <rect x={px - 46} y={0} width={92} height={18} rx={5}
        fill="url(#pvMountGrad)" stroke="#4a4a4a" strokeWidth="0.6" />
      {/* Mounting screws */}
      {[-28, 28].map(ox => (
        <g key={ox}>
          <circle cx={px + ox} cy={9} r={4} fill="#3a3a3a" stroke="#555" strokeWidth="0.5" />
          <line x1={px+ox-2.5} y1={6.5} x2={px+ox+2.5} y2={11.5} stroke="#555" strokeWidth="0.9" />
          <line x1={px+ox-2.5} y1={11.5} x2={px+ox+2.5} y2={6.5} stroke="#555" strokeWidth="0.9" />
        </g>
      ))}

      {/* ── VERTICAL ARM ── */}
      <rect x={px - 6} y={16} width={12} height={py - 16} rx={4}
        fill="url(#pvMountGrad)" stroke="#4a4a4a" strokeWidth="0.5" />
      {/* Arm highlight */}
      <rect x={px - 2} y={18} width={3} height={py - 20} rx={1.5}
        fill="rgba(255,255,255,0.15)" />

      {/* ── PIVOT BALL JOINT ── */}
      <circle cx={px} cy={py} r={18}
        fill="url(#pvPivotGrad)" stroke="#3a3a3a" strokeWidth="0.8" />
      <circle cx={px} cy={py} r={7}
        fill="#282828" stroke="#555" strokeWidth="0.5" />
      <circle cx={px} cy={py} r={3}
        fill="#1a1a1a" />
      {/* Pivot specular */}
      <ellipse cx={px - 5} cy={py - 5} rx={6} ry={4}
        fill="rgba(255,255,255,0.22)" transform={`rotate(-30,${px-5},${py-5})`} />

      {/* ══ ROTATING CAMERA GROUP ══ */}
      <g
        style={{
          transform: `rotate(${finalAngle}deg)`,
          transformOrigin: `${px}px ${py}px`,
          transition,
        }}
        filter="url(#pvCamShadow)"
      >
        {/* Short arm below pivot to body */}
        <rect x={px - 5} y={py} width={10} height={armH + 2} rx={2}
          fill="url(#pvMountGrad)" />

        {/* ── MAIN HOUSING BODY ── */}
        <rect x={bX} y={bY} width={bW} height={bH} rx={11}
          fill="url(#pvBodyGrad)" stroke="#444" strokeWidth="0.6" />
        {/* Body top sheen */}
        <rect x={bX + 3} y={bY + 3} width={bW - 6} height={12} rx={6}
          fill="url(#pvBodyTopGrad)" />

        {/* Ventilation slots (right side) */}
        {[0,1,2,3,4].map(i => (
          <rect key={i}
            x={bX + bW - 20} y={bY + 14 + i * 9}
            width={14} height={2.5} rx={1.2}
            fill="rgba(255,255,255,0.07)"
          />
        ))}

        {/* Brand label */}
        <text
          x={px} y={bY + bH / 2 - 12}
          textAnchor="middle"
          fill="rgba(255,255,255,0.18)"
          fontSize="7.5"
          fontFamily="monospace"
          letterSpacing="2"
        >
          PV-CAM
        </text>
        <text
          x={px} y={bY + bH / 2}
          textAnchor="middle"
          fill="rgba(255,255,255,0.1)"
          fontSize="5.5"
          fontFamily="monospace"
          letterSpacing="1"
        >
          AI PROCTORING
        </text>

        {/* ── STATUS LED ── */}
        <circle cx={bX + bW - 10} cy={bY + 12} r={4.5}
          fill={isLookingAway ? "#ff3b3b" : "#00e676"}
          style={{
            filter: isLookingAway
              ? "drop-shadow(0 0 5px #ff3b3b)"
              : "drop-shadow(0 0 5px #00e676)"
          }}
        />
        {/* LED blink ring */}
        <circle cx={bX + bW - 10} cy={bY + 12} r={7}
          fill="none"
          stroke={isLookingAway ? "rgba(255,59,59,0.25)" : "rgba(0,230,118,0.25)"}
          strokeWidth="2"
        />

        {/* ── LENS SECTION ── */}
        {/* Dark lens bay */}
        <rect x={bX + 6} y={lensY - 24} width={bW - 12} height={44} rx={8}
          fill="#111" stroke="#3a3a3a" strokeWidth="0.5" />

        {/* IR LED ring – 8 LEDs at r=20 from lens centre */}
        {[0, 45, 90, 135, 180, 225, 270, 315].map(deg => {
          const rad = deg * Math.PI / 180
          return (
            <circle key={deg}
              cx={lensX + 20 * Math.sin(rad)}
              cy={lensY + 20 * Math.cos(rad)}
              r={2.8}
              fill={isLookingAway ? "#550000" : "#e65100"}
              style={{ opacity: isLookingAway ? 0.4 : 0.85 }}
            />
          )
        })}

        {/* Lens outer bezel ring */}
        <circle cx={lensX} cy={lensY} r={15.5}
          fill="url(#pvRingGrad)" stroke="#555" strokeWidth="1.5" />

        {/* Lens glass */}
        <circle cx={lensX} cy={lensY} r={12}
          fill="url(#pvLensGrad)" />

        {/* Lens mid ring */}
        <circle cx={lensX} cy={lensY} r={7.5}
          fill="none"
          stroke="rgba(80,130,210,0.35)" strokeWidth="1.2" />

        {/* Lens pupil */}
        <circle cx={lensX} cy={lensY} r={4.5}
          fill="#02020e" />

        {/* Main specular reflection */}
        <ellipse
          cx={lensX - 4.5} cy={lensY - 5.5}
          rx={3.8} ry={2.5}
          fill="rgba(255,255,255,0.50)"
          transform={`rotate(-30,${lensX-4.5},${lensY-5.5})`}
        />
        {/* Small secondary reflection */}
        <ellipse
          cx={lensX + 4} cy={lensY + 4}
          rx={1.5} ry={1}
          fill="rgba(255,255,255,0.18)"
        />

        {/* ── CABLE ── */}
        <path
          d={`M ${px} ${bY} C ${px+18} ${bY-16} ${px+22} ${py+10} ${px+6} ${py+4}`}
          fill="none" stroke="#1a1a1a" strokeWidth="5" strokeLinecap="round"
        />
        <path
          d={`M ${px} ${bY} C ${px+18} ${bY-16} ${px+22} ${py+10} ${px+6} ${py+4}`}
          fill="none" stroke="#2d2d2d" strokeWidth="2.5" strokeLinecap="round"
        />
      </g>
    </svg>
  )
}

/* ─────────────────────────────────────────────
   LOGIN PAGE
───────────────────────────────────────────── */
export default function Login({ onLogin }) {
  const [username, setUsername]           = useState("")
  const [password, setPassword]           = useState("")
  const [isPasswordFocused, setPassFocus] = useState(false)
  const [camAngle, setCamAngle]           = useState(0)
  const [isLoading, setIsLoading]         = useState(false)
  const [isExiting, setIsExiting]         = useState(false)
  const [mounted, setMounted]             = useState(false)
  const svgRef    = useRef(null)
  const [domeAngle, setDomeAngle] = useState(0)
  const [lensGlow,  setLensGlow]  = useState(0)

  // Realistic patrol: sweep → pause → sweep back → pause, + lens beam pulses
  useEffect(() => {
    let t = 0
    const id = setInterval(() => {
      t += 0.008
      // Eased sweep using smoothstep on a sawtooth — pauses at extremes
      const raw  = (Math.sin(t) + Math.sin(t * 0.5)) / 1.8
      setDomeAngle(raw * 42)
      // Lens scan glow: brief bright flash at turn-around points
      setLensGlow(Math.max(0, Math.abs(Math.cos(t)) - 0.55) * 2.2)
    }, 16)
    return () => clearInterval(id)
  }, [])

  // Mount entrance
  useEffect(() => { setTimeout(() => setMounted(true), 50) }, [])

  // Cursor tracking
  useEffect(() => {
    const onMove = (e) => {
      if (isPasswordFocused || !svgRef.current) return
      const rect   = svgRef.current.getBoundingClientRect()
      const scaleX = rect.width  / 300
      const scaleY = rect.height / 215
      const pivotX = rect.left + 150 * scaleX
      const pivotY = rect.top  +  82 * scaleY
      const dx = e.clientX - pivotX
      const dy = e.clientY - pivotY
      const deg = -(Math.atan2(dx, dy) * 180 / Math.PI)
      setCamAngle(Math.max(-65, Math.min(65, deg)))
    }
    window.addEventListener("mousemove", onMove)
    return () => window.removeEventListener("mousemove", onMove)
  }, [isPasswordFocused])

  const handleLogin = async () => {
    if (!username || !password) return
    setIsLoading(true)
    try {
      const res  = await fetch("http://localhost:5000/login", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ username, password }),
      })
      const data = await res.json()
      if (data.status === "success") {
        setIsExiting(true)
        setTimeout(() => onLogin(username), 620)
      } else {
        // Shake the card
        const card = document.getElementById("pv-card")
        card && card.classList.add("pv-shake")
        setTimeout(() => card && card.classList.remove("pv-shake"), 500)
        alert("Invalid credentials — access denied.")
      }
    } catch {
      alert("Cannot reach server. Check the Flask backend.")
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === "Enter") handleLogin()
  }

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        @keyframes pvFadeUp {
          from { opacity: 0; transform: translateY(28px) scale(0.98); }
          to   { opacity: 1; transform: translateY(0)    scale(1); }
        }
        @keyframes pvFadeOut {
          from { opacity: 1; transform: translateY(0)    scale(1); }
          to   { opacity: 0; transform: translateY(-24px) scale(1.03); }
        }
        @keyframes pvPulse {
          0%,100% { opacity: 1; transform: scale(1); }
          50%      { opacity: 0.55; transform: scale(0.92); }
        }
        @keyframes pvShake {
          0%,100% { transform: translateX(0); }
          20%     { transform: translateX(-8px); }
          40%     { transform: translateX( 8px); }
          60%     { transform: translateX(-5px); }
          80%     { transform: translateX( 5px); }
        }
        @keyframes pvScanMove {
          from { top: -4px; }
          to   { top: 100%; }
        }
        @keyframes pvDotGrid {
          0%,100% { opacity: 0.5; }
          50%     { opacity: 0.9; }
        }

        .pv-shake { animation: pvShake 0.45s ease; }

        .pv-input {
          width: 100%;
          padding: 13px 16px;
          border: 1.5px solid #e2e8f0;
          border-radius: 10px;
          font-family: 'DM Sans', sans-serif;
          font-size: 14.5px;
          color: #111827;
          background: #f8fafc;
          outline: none;
          transition: border-color 0.2s, box-shadow 0.2s, background 0.2s;
        }
        .pv-input:focus {
          border-color: #3b5bdb;
          background: #ffffff;
          box-shadow: 0 0 0 4px rgba(59,91,219,0.12);
        }
        .pv-input::placeholder { color: #94a3b8; }

        .pv-btn {
          width: 100%;
          padding: 14px 20px;
          border: none;
          border-radius: 11px;
          background: linear-gradient(135deg, #0a0e27 0%, #1e2a78 50%, #3b5bdb 100%);
          color: #fff;
          font-family: 'DM Sans', sans-serif;
          font-size: 15px;
          font-weight: 600;
          letter-spacing: 0.6px;
          cursor: pointer;
          transition: transform 0.18s, box-shadow 0.18s, opacity 0.18s;
          position: relative;
          overflow: hidden;
        }
        .pv-btn::after {
          content: '';
          position: absolute;
          inset: 0;
          background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.08) 50%, transparent 100%);
          transform: translateX(-100%);
          transition: transform 0.4s ease;
        }
        .pv-btn:hover::after { transform: translateX(100%); }
        .pv-btn:hover {
          transform: translateY(-2px);
          box-shadow: 0 10px 28px rgba(59,91,219,0.45);
        }
        .pv-btn:active  { transform: translateY(0); box-shadow: none; }
        .pv-btn:disabled { opacity: 0.65; cursor: not-allowed; transform: none; box-shadow: none; }

        .pv-label {
          display: block;
          font-family: 'DM Sans', sans-serif;
          font-size: 11.5px;
          font-weight: 600;
          letter-spacing: 0.7px;
          text-transform: uppercase;
          color: #475569;
          margin-bottom: 7px;
        }
      `}</style>

      {/* ── PAGE WRAPPER ── */}
      <div style={{
        minHeight: "100vh",
        background: "#f4f6fb",
        backgroundImage: `
          radial-gradient(circle at 20% 20%, rgba(59,91,219,0.07) 0%, transparent 50%),
          radial-gradient(circle at 80% 80%, rgba(10,14,39,0.06)  0%, transparent 50%),
          radial-gradient(circle at 1px 1px, rgba(0,0,0,0.05) 1px, transparent 0)
        `,
        backgroundSize: "auto, auto, 26px 26px",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontFamily: "'DM Sans', sans-serif",
        animation: isExiting
          ? "pvFadeOut 0.62s cubic-bezier(0.4,0,0.2,1) forwards"
          : "pvFadeUp  0.7s  cubic-bezier(0.4,0,0.2,1) forwards",
        opacity: mounted ? 1 : 0,
      }}>

        <div style={{ width: "100%", maxWidth: 960, padding: "0 20px", display: "flex", alignItems: "center", gap: 32, justifyContent: "center" }}>

          {/* ── DOME CCTV CAMERA PANEL ── */}
          <div style={{
            flex: "0 0 400px",
            minWidth: 0,
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
          }}>

              {/* ── DOME CAMERA SVG ── */}
              <svg
                viewBox="0 0 340 300"
                width="100%"
                style={{ overflow: "visible", filter: "drop-shadow(0 20px 50px rgba(0,0,0,0.25))" }}
              >
                <defs>
                  {/* Dome body – white/light grey */}
                  <radialGradient id="domeGrad" cx="38%" cy="32%" r="65%">
                    <stop offset="0%"   stopColor="#f0f2f5" />
                    <stop offset="40%"  stopColor="#d8dce4" />
                    <stop offset="80%"  stopColor="#b8bcc8" />
                    <stop offset="100%" stopColor="#8a8e9a" />
                  </radialGradient>

                  {/* Dome underside rim */}
                  <linearGradient id="domeRimGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%"   stopColor="#7a7e8a" />
                    <stop offset="100%" stopColor="#5a5e6a" />
                  </linearGradient>

                  {/* Mounting base */}
                  <linearGradient id="mountGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%"   stopColor="#e8eaf0" />
                    <stop offset="50%"  stopColor="#c8cad4" />
                    <stop offset="100%" stopColor="#a8aab4" />
                  </linearGradient>

                  {/* Lens housing – dark */}
                  <radialGradient id="lensHousingGrad" cx="40%" cy="35%">
                    <stop offset="0%"   stopColor="#3a3a3a" />
                    <stop offset="60%"  stopColor="#1a1a1a" />
                    <stop offset="100%" stopColor="#0a0a0a" />
                  </radialGradient>

                  {/* Lens glass */}
                  <radialGradient id="lensGlassGrad" cx="30%" cy="28%">
                    <stop offset="0%"   stopColor="#7ab0e8" />
                    <stop offset="25%"  stopColor="#2860c0" />
                    <stop offset="55%"  stopColor="#0a2060" />
                    <stop offset="80%"  stopColor="#040e30" />
                    <stop offset="100%" stopColor="#020818" />
                  </radialGradient>

                  {/* Lens inner ring */}
                  <radialGradient id="lensInnerGrad" cx="35%" cy="30%">
                    <stop offset="0%"   stopColor="#556688" />
                    <stop offset="100%" stopColor="#1a2030" />
                  </radialGradient>

                  {/* IR LED glow */}
                  <radialGradient id="irGlowGrad" cx="50%" cy="50%">
                    <stop offset="0%"   stopColor="rgba(255,200,100,0.9)" />
                    <stop offset="60%"  stopColor="rgba(200,100,0,0.4)" />
                    <stop offset="100%" stopColor="rgba(200,100,0,0)" />
                  </radialGradient>

                  {/* Background glow behind camera */}
                  <radialGradient id="bgGlowGrad" cx="50%" cy="55%">
                    <stop offset="0%"   stopColor="rgba(59,91,219,0.15)" />
                    <stop offset="100%" stopColor="rgba(59,91,219,0)" />
                  </radialGradient>

                  {/* Shadow under dome */}
                  <radialGradient id="domeShadow" cx="50%" cy="50%">
                    <stop offset="0%"   stopColor="rgba(0,0,0,0.5)" />
                    <stop offset="100%" stopColor="rgba(0,0,0,0)" />
                  </radialGradient>

                  <filter id="domeSoftShadow">
                    <feDropShadow dx="0" dy="12" stdDeviation="14" floodColor="rgba(0,0,0,0.6)" />
                  </filter>
                  <filter id="irGlow">
                    <feGaussianBlur stdDeviation="3" result="blur"/>
                    <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                  </filter>
                </defs>

                {/* Background ambient glow */}
                <ellipse cx="170" cy="200" rx="130" ry="60" fill="url(#bgGlowGrad)" />

                {/* ── MOUNTING BRACKET (ceiling plate) ── */}
                <rect x="120" y="52" width="100" height="14" rx="5"
                  fill="url(#mountGrad)" stroke="#9a9caa" strokeWidth="0.8" />
                {/* Bracket screws */}
                {[138, 202].map(x => (
                  <g key={x}>
                    <circle cx={x} cy={59} r={4.5} fill="#b0b2bc" stroke="#888" strokeWidth="0.5"/>
                    <line x1={x-3} y1={56.5} x2={x+3} y2={61.5} stroke="#888" strokeWidth="1"/>
                    <line x1={x-3} y1={61.5} x2={x+3} y2={56.5} stroke="#888" strokeWidth="1"/>
                  </g>
                ))}
                {/* Bracket neck */}
                <rect x="152" y="64" width="36" height="18" rx="4"
                  fill="url(#mountGrad)" stroke="#9a9caa" strokeWidth="0.6"/>

                {/* ── DOME BODY – rotates with domeAngle ── */}
                <g
                  style={{
                    transform: `rotate(${domeAngle}deg)`,
                    transformOrigin: "170px 82px",
                    transition: "transform 0.05s linear",
                  }}
                  filter="url(#domeSoftShadow)"
                >
                  {/* Dome shell – half sphere */}
                  <ellipse cx="170" cy="145" rx="105" ry="30" fill="url(#domeShadow)" opacity="0.6"/>

                  {/* Main dome – white plastic half-sphere */}
                  <path
                    d="M 65 148 Q 65 60 170 60 Q 275 60 275 148 Z"
                    fill="url(#domeGrad)"
                    stroke="#c0c4d0"
                    strokeWidth="1.2"
                  />

                  {/* Dome surface detail lines (subtle ribs) */}
                  {[-60, -30, 0, 30, 60].map((angle, i) => {
                    const rad = (angle * Math.PI) / 180
                    const x2 = 170 + 105 * Math.sin(rad)
                    const y2 = 148 - 88 * (1 - Math.abs(Math.sin(rad)) * 0.3)
                    return (
                      <line key={i} x1="170" y1="60"
                        x2={x2} y2={y2}
                        stroke="rgba(255,255,255,0.15)" strokeWidth="0.8"
                      />
                    )
                  })}

                  {/* Dome rim ring */}
                  <ellipse cx="170" cy="148" rx="105" ry="14"
                    fill="url(#domeRimGrad)"
                    stroke="#6a6e7a" strokeWidth="1"
                  />
                  {/* Rim highlight */}
                  <ellipse cx="170" cy="145" rx="80" ry="5"
                    fill="rgba(255,255,255,0.12)"
                  />

                  {/* ── LENS ASSEMBLY ── */}
                  {/* Lens bay recess */}
                  <ellipse cx="170" cy="132" rx="52" ry="48"
                    fill="#0d0d0d" stroke="#2a2a2a" strokeWidth="1.5"
                  />

                  {/* IR LED ring – 12 LEDs */}
                  {Array.from({ length: 12 }, (_, i) => {
                    const ang = (i / 12) * Math.PI * 2 - Math.PI / 2
                    const lx = 170 + 40 * Math.cos(ang)
                    const ly = 132 + 37 * Math.sin(ang)
                    return (
                      <g key={i} filter="url(#irGlow)">
                        <circle cx={lx} cy={ly} r={3.8}
                          fill="#c06000"
                          stroke="#804000" strokeWidth="0.5"
                          style={{ animation: `pvPulse ${1.8 + (i % 3) * 0.3}s ease infinite` }}
                        />
                        {/* LED glow halo */}
                        <circle cx={lx} cy={ly} r={6}
                          fill="url(#irGlowGrad)"
                          opacity="0.55"
                          style={{ animation: `pvPulse ${1.8 + (i % 3) * 0.3}s ease infinite` }}
                        />
                      </g>
                    )
                  })}

                  {/* Outer lens bezel */}
                  <circle cx="170" cy="132" r="24"
                    fill="url(#lensHousingGrad)"
                    stroke="#404040" strokeWidth="2"
                  />
                  {/* Bezel knurling marks */}
                  {Array.from({ length: 20 }, (_, i) => {
                    const ang = (i / 20) * Math.PI * 2
                    const r1 = 22, r2 = 24
                    return (
                      <line key={i}
                        x1={170 + r1 * Math.cos(ang)} y1={132 + r1 * Math.sin(ang)}
                        x2={170 + r2 * Math.cos(ang)} y2={132 + r2 * Math.sin(ang)}
                        stroke="rgba(255,255,255,0.1)" strokeWidth="1"
                      />
                    )
                  })}

                  {/* Lens glass – multi-layer */}
                  <circle cx="170" cy="132" r="18" fill="url(#lensGlassGrad)" />
                  {/* Active scan glow — pulses when camera hits sweep extremes */}
                  <circle cx="170" cy="132" r="18"
                    fill={`rgba(80,160,255,${lensGlow * 0.5})`}
                  />
                  <circle cx="170" cy="132" r="28"
                    fill="none"
                    stroke={`rgba(59,130,246,${lensGlow * 0.6})`}
                    strokeWidth="6"
                    opacity={lensGlow}
                  />
                  {/* Lens inner ring 1 */}
                  <circle cx="170" cy="132" r="13"
                    fill="none" stroke="rgba(80,120,200,0.4)" strokeWidth="1.5" />
                  {/* Lens inner ring 2 */}
                  <circle cx="170" cy="132" r="8"
                    fill="url(#lensInnerGrad)" stroke="rgba(60,100,180,0.5)" strokeWidth="1" />
                  {/* Lens pupil */}
                  <circle cx="170" cy="132" r="4.5" fill="#010208" />
                  {/* Main lens specular */}
                  <ellipse cx="163" cy="124" rx="5" ry="3.5"
                    fill="rgba(255,255,255,0.55)"
                    transform="rotate(-30,163,124)"
                  />
                  {/* Small specular */}
                  <ellipse cx="176" cy="138" rx="2" ry="1.3"
                    fill="rgba(255,255,255,0.2)"
                  />

                  {/* Dome top specular highlight */}
                  <ellipse cx="148" cy="86" rx="38" ry="16"
                    fill="rgba(255,255,255,0.28)"
                    transform="rotate(-15,148,86)"
                  />
                  <ellipse cx="145" cy="82" rx="18" ry="7"
                    fill="rgba(255,255,255,0.18)"
                    transform="rotate(-20,145,82)"
                  />
                </g>
              </svg>
          </div>

          {/* ── LOGIN COLUMN ── */}
          <div style={{ flex: "0 0 420px", minWidth: 0 }}>

          {/* ── CCTV CAMERA ── */}
          <div style={{ display: "flex", justifyContent: "center", marginBottom: -14 }}>
            <CCTVCamera
              svgRef={svgRef}
              angle={camAngle}
              isLookingAway={isPasswordFocused}
            />
          </div>

          {/* ── CARD ── */}
          <div
            id="pv-card"
            style={{
              background: "#ffffff",
              borderRadius: 22,
              padding: "36px 38px 32px",
              boxShadow: "0 24px 64px rgba(0,0,0,0.11), 0 4px 16px rgba(0,0,0,0.07)",
              border: "1px solid rgba(255,255,255,0.9)",
              position: "relative",
              overflow: "hidden",
            }}
          >
            {/* Scan-line shimmer */}
            <div style={{
              position: "absolute",
              left: 0, right: 0,
              height: 3,
              background: "linear-gradient(90deg, transparent, rgba(59,91,219,0.25), transparent)",
              animation: "pvScanMove 4s linear infinite",
              pointerEvents: "none",
              zIndex: 10,
            }} />

            {/* ── LOGO ── */}
            <div style={{ textAlign: "center", marginBottom: 26 }}>
              <div style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 10,
                marginBottom: 6,
              }}>
                {/* Icon badge */}
                <div style={{
                  width: 38, height: 38,
                  background: "linear-gradient(145deg, #0a0e27, #3b5bdb)",
                  borderRadius: 10,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  boxShadow: "0 4px 12px rgba(59,91,219,0.35)",
                  flexShrink: 0,
                }}>
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
                    <ellipse cx="12" cy="12" rx="10" ry="7" fill="none" stroke="rgba(255,255,255,0.9)" strokeWidth="1.6"/>
                    <circle cx="12" cy="12" r="4" fill="rgba(255,255,255,0.9)"/>
                    <circle cx="12" cy="12" r="2" fill="#0a0e27"/>
                    <circle cx="14" cy="10" r="0.9" fill="rgba(255,255,255,0.7)"/>
                  </svg>
                </div>
                <span style={{
                  fontFamily: "'Orbitron', sans-serif",
                  fontWeight: 700,
                  fontSize: 22,
                  letterSpacing: "0.5px",
                  background: "linear-gradient(135deg, #0a0e27, #3b5bdb)",
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                }}>
                  VisionSentinel
                </span>
              </div>
              <p style={{ color: "#64748b", fontSize: 12.5, letterSpacing: "0.4px" }}>
                AI Exam Surveillance &amp; Proctoring Platform
              </p>
            </div>

            {/* ── PRIVACY TOAST ── */}
            <div style={{
              overflow: "hidden",
              maxHeight: isPasswordFocused ? 60 : 0,
              opacity:   isPasswordFocused ? 1 : 0,
              transition: "max-height 0.3s ease, opacity 0.3s ease",
              marginBottom: isPasswordFocused ? 16 : 0,
            }}>
              <div style={{
                background: "linear-gradient(135deg, #fffbeb, #fef3c7)",
                border: "1px solid #fcd34d",
                borderRadius: 10,
                padding: "10px 14px",
                display: "flex",
                alignItems: "center",
                gap: 8,
                fontSize: 12.5,
                color: "#78350f",
              }}>
                <span style={{ fontSize: 18 }}>🫣</span>
                <span><strong>Camera averted!</strong> Your privacy is safe.</span>
              </div>
            </div>

            {/* ── FORM FIELDS ── */}
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              <div>
                <label className="pv-label">Username</label>
                <input
                  className="pv-input"
                  placeholder="Enter your username"
                  value={username}
                  autoComplete="username"
                  onChange={e => setUsername(e.target.value)}
                  onKeyDown={handleKeyDown}
                />
              </div>

              <div>
                <label className="pv-label">Password</label>
                <input
                  className="pv-input"
                  type="password"
                  placeholder="Enter your password"
                  value={password}
                  autoComplete="current-password"
                  onChange={e => setPassword(e.target.value)}
                  onFocus={() => setPassFocus(true)}
                  onBlur={() => setPassFocus(false)}
                  onKeyDown={handleKeyDown}
                />
              </div>

              <button
                className="pv-btn"
                onClick={handleLogin}
                disabled={isLoading}
                style={{ marginTop: 4 }}
              >
                {isLoading ? "Authenticating…" : "Access System →"}
              </button>
            </div>

            {/* ── FOOTER ── */}
            <div style={{
              marginTop: 22,
              paddingTop: 18,
              borderTop: "1px solid #f1f5f9",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 6,
              fontSize: 11.5,
              color: "#94a3b8",
            }}>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                <rect x="3" y="11" width="18" height="11" rx="2" stroke="#94a3b8" strokeWidth="2"/>
                <path d="M7 11V7a5 5 0 0110 0v4" stroke="#94a3b8" strokeWidth="2" strokeLinecap="round"/>
              </svg>
              Secured with end-to-end encryption
            </div>
          </div>

          {/* Version badge */}
          <div style={{
            textAlign: "center",
            marginTop: 14,
            fontSize: 11,
            color: "#b0bec5",
            fontFamily: "monospace",
            letterSpacing: "0.5px",
          }}>
            VisionSentinel v2.0 · SENTINEL AI Engine
          </div>
          </div>{/* end login column */}
        </div>{/* end outer flex */}
      </div>
    </>
  )
}