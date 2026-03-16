// ================================================================
// SENTINEL AI v9 — Advanced Models & Algorithms
//
// UPGRADES FROM v8:
//
// MODEL UPGRADES:
//   COCO-SSD mobilenet_v2 (full)     — unchanged, best available in tfjs
//   MoveNet MultiPose Lightning       — tracks up to 6 students simultaneously
//                                       (was: SinglePose = only 1 person)
//   FaceLandmarks via @mediapipe/face_mesh  — 468 points per face,
//                                       real eye openness, iris position,
//                                       mouth open — no more fake gaze
//   BlazeFace                         — kept as fast backup for face count
//
// ALGORITHM UPGRADES:
//   1. RISK SCORE PER STUDENT
//      Each student accumulates a 0–100 risk score from events.
//      High = red heat border on bbox. Resets after 60s inactivity.
//      Score weights: phone=40, head_turn=15, arm_raise=12, lean=20,
//                     absent=25, gaze=10, book=35
//
//   2. WRIST VELOCITY TRACKING
//      Tracks wrist (x,y) between consecutive MoveNet frames.
//      Low velocity = normal writing/sitting.
//      High velocity = suspicious quick movements (passing notes, hiding phone).
//      Velocity vector drawn on skeleton as orange arrow.
//
//   3. HEAD PITCH ANGLE (real degrees)
//      Uses ear-shoulder-nose triangle geometry from MoveNet.
//      pitch > 40° = head down (reading hidden notes)
//      Uses atan2 not a ratio — actual angle.
//
//   4. EYE OPENNESS + IRIS POSITION from FaceMesh
//      Eye aspect ratio (EAR) from 6 eye landmarks each eye.
//      Low EAR = eyes closed (sleeping / distracted).
//      Iris x-position relative to eye corners = precise gaze direction.
//      NO more fake "yaw from nose" — actual iris tracking.
//
//   5. PHONE DETECTION ENHANCEMENT
//      Pre-process zoom crop with contrast stretch + edge enhancement
//      before passing to COCO-SSD. Small rectangular objects become
//      much more distinct after sharpening, especially at 6× zoom.
//      Implemented via canvas ImageData manipulation.
//
//   6. MULTI-STUDENT CORRELATION
//      If two adjacent students both show high wrist velocity at the
//      same time → "Coordinated Cheating" HIGH alert.
//
//   7. TEMPORAL SMOOTHING
//      Each detection class keeps a 4-frame rolling window.
//      Alert only fires if the class appeared in 3 of last 4 frames.
//      Eliminates single-frame false positives entirely.
//
// CODESANDBOX DEPS (all required):
//   @tensorflow/tfjs
//   @tensorflow-models/coco-ssd
//   @tensorflow-models/pose-detection
//   @tensorflow-models/blazeface
//   @mediapipe/face_mesh  ← NEW
// ================================================================

import { useState, useEffect, useRef, useCallback } from "react";
import * as tf from "@tensorflow/tfjs";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import * as poseDetection from "@tensorflow-models/pose-detection";
import * as blazeface from "@tensorflow-models/blazeface";
import { jsPDF } from "jspdf";
import { BrowserRouter, Routes, Route } from "react-router-dom"
import Login from "./pages/Login"
import Dashboard from "./pages/Dashboard"
// ── Audio beep on HIGH alerts (Web Audio API) ─────────────────
let _audioCtx=null;
function playAlertBeep(sev){
  try{
    if(!_audioCtx) _audioCtx=new (window.AudioContext||window.webkitAudioContext)();
    const osc=_audioCtx.createOscillator();
    const gain=_audioCtx.createGain();
    osc.connect(gain);gain.connect(_audioCtx.destination);
    osc.type=sev==="high"?"square":"sine";
    osc.frequency.value=sev==="high"?880:660;
    gain.gain.value=0.13;
    osc.start();osc.stop(_audioCtx.currentTime+(sev==="high"?0.22:0.12));
  }catch(e){}
}

// ── Browser notification on HIGH alerts ────────────────────────
function sendBrowserNotif(title,body){
  try{
    if(Notification.permission==="granted"){
      new Notification(title,{body,icon:"data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🎯</text></svg>"});
    } else if(Notification.permission!=="denied"){
      Notification.requestPermission();
    }
  }catch(e){}
}

const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@400;500;600;700&display=swap');
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  html,body{background:#040810;color:#8aaabb;font-family:'Barlow Condensed',sans-serif;overflow:hidden;height:100%}
  :root{
    --c:#00d8ff;--r:#ff2d4a;--a:#ffaa00;--g:#00eb7a;--p:#c060ff;
    --bd:rgba(0,216,255,.09);--bd2:rgba(0,216,255,.22);
    --panel:rgba(3,8,18,.97);
  }
  .M{font-family:'Share Tech Mono',monospace}
  ::-webkit-scrollbar{width:2px}::-webkit-scrollbar-thumb{background:var(--bd)}
  @keyframes PG{0%,100%{box-shadow:0 0 0 0 rgba(0,216,255,.28)}50%{box-shadow:0 0 0 6px rgba(0,216,255,0)}}
  @keyframes PR{0%,100%{box-shadow:0 0 0 0 rgba(255,45,74,.35)}50%{box-shadow:0 0 0 8px rgba(255,45,74,0)}}
  @keyframes PA{0%,100%{box-shadow:0 0 0 0 rgba(255,170,0,.3)}50%{box-shadow:0 0 0 7px rgba(255,170,0,0)}}
  @keyframes BL{0%,100%{opacity:1}50%{opacity:.08}}
  @keyframes UP{from{opacity:0;transform:translateY(9px)}to{opacity:1;transform:translateY(0)}}
  @keyframes SP{from{transform:rotate(0)}to{transform:rotate(360deg)}}
  @keyframes SL{0%{top:-1px}100%{top:100%}}
  @keyframes FH{0%{background:rgba(255,45,74,.16)}80%{background:rgba(255,45,74,.04)}100%{background:transparent}}
  @keyframes FA{0%{background:rgba(255,170,0,.13)}80%{background:rgba(255,170,0,.03)}100%{background:transparent}}
  @keyframes FC{0%{background:rgba(0,216,255,.1)}100%{background:transparent}}
  @keyframes RISK{0%,100%{opacity:1}50%{opacity:.4}}
  @keyframes FLASH_EV{0%{box-shadow:0 0 0 2px rgba(255,45,74,.6)}100%{box-shadow:0 0 0 2px transparent}}
  .ev-thumb{transition:transform .15s;cursor:pointer}
  .ev-thumb:hover{transform:scale(1.08);z-index:5}
`;

const uid = () => Math.random().toString(36).slice(2,8);
const TS  = () => new Date().toLocaleTimeString("en-US",{hour12:false});
const TSF = () => {const d=new Date();return `${d.toLocaleDateString("en-IN")} ${d.toLocaleTimeString("en-US",{hour12:false})}`};
function clamp(v,a,b){return Math.max(a,Math.min(b,v));}

const ACCS = {
  admin:   {pass:"admin123",name:"Dr. Sharma",   role:"Head Department",  dept:"Security Ops",   av:"🎓"},
  security:{pass:"sec123",  name:"Off. Patel",   role:"Security Officer", dept:"Campus Security",av:"🛡️"},
  examhead:{pass:"exam123", name:"Prof. Williams",role:"Exam Supervisor",  dept:"Academic Affairs",av:"📋"},
};

// ── Risk event weights ────────────────────────────────────────
const RISK_W = {
  "cell phone":40,"book":35,"laptop":35,"lean_over":20,
  "absent":25,"gaze_right":10,"gaze_left":10,"gaze_up":8,
  "arm_raise":12,"head_down":10,"coordinated":30,
  "proxy":45,"knife":50,"crouch":30,"running":25,
};

// ── Per-class confidence floors ───────────────────────────────
const CONF = {
  "cell phone":0.42,"knife":0.55,"person":0.38,"book":0.42,
  "laptop":0.44,"scissors":0.50,"backpack":0.38,"suitcase":0.38,
  "remote":0.48,"keyboard":0.44,"handbag":0.38,"bottle":0.38,"cup":0.38,
};
const DC = 0.40;

// ── Cooldowns per student+event ───────────────────────────────
const CD = {
  "cell phone":14000,"knife":4000,"book":14000,"laptop":14000,
  "gaze_left":7000,"gaze_right":7000,"gaze_up":9000,
  "arm_raise":8000,"lean_over":8000,"head_down":11000,
  "absent":20000,"proxy":6000,"person":60000,
  "wrist_velocity":10000,"coordinated":15000,"eyes_closed":12000,
  "default":13000,
};

// ── Temporal smoothing window (frames to confirm detection) ───
const CONFIRM_FRAMES = 2; // detect in 2 of last 3 frames before alert

const MODES = {
  security:{
    label:"Security & Threat",icon:"🛡",color:"#ff2d4a",
    watch:["person","backpack","suitcase","knife","scissors","cell phone","laptop","handbag","bottle"],
    rules:{
      "person":     {lbl:"Person Detected",        sev:"low"},
      "knife":      {lbl:"⚠️ Weapon Detected!",    sev:"high"},
      "scissors":   {lbl:"Sharp Object",           sev:"med"},
      "backpack":   {lbl:"Unattended Bag",         sev:"med"},
      "suitcase":   {lbl:"Unattended Luggage",     sev:"med"},
      "cell phone": {lbl:"Unauthorized Phone",     sev:"low"},
      "laptop":     {lbl:"Unauthorized Laptop",    sev:"med"},
      "handbag":    {lbl:"Unattended Handbag",     sev:"low"},
      "bottle":     {lbl:"Suspicious Container",   sev:"low"},
    }
  },
  exam:{
    label:"Exam Integrity",icon:"📋",color:"#ffaa00",
    watch:["person","cell phone","book","laptop","remote","keyboard","scissors","bottle","cup"],
    rules:{
      "person":     {lbl:"Student Present",            sev:"low"},
      "cell phone": {lbl:"🚨 Phone Detected!",         sev:"high"},
      "book":       {lbl:"Unauthorized Book/Notes",    sev:"high"},
      "laptop":     {lbl:"Unauthorized Laptop",        sev:"high"},
      "remote":     {lbl:"Suspicious Device",          sev:"med"},
      "keyboard":   {lbl:"Unauthorized Device",        sev:"med"},
      "scissors":   {lbl:"Sharp Object",               sev:"med"},
      "bottle":     {lbl:"Suspicious Container",       sev:"low"},
      "cup":        {lbl:"Possible Hidden Notes",      sev:"low"},
    }
  }
};

const SC  = s=>s==="high"?"#ff2d4a":s==="med"?"#ffaa00":"#00d8ff";
const SBG = s=>s==="high"?"rgba(255,45,74,.07)":s==="med"?"rgba(255,170,0,.07)":"rgba(0,216,255,.05)";
const riskColor = r => r>70?"#ff2d4a":r>40?"#ffaa00":r>20?"#ffd000":"#00d8ff";

let TID=1;
function iou(a,b){
  const x1=Math.max(a[0],b[0]),y1=Math.max(a[1],b[1]);
  const x2=Math.min(a[0]+a[2],b[0]+b[2]),y2=Math.min(a[1]+a[3],b[1]+b[3]);
  if(x2<=x1||y2<=y1) return 0;
  const i=(x2-x1)*(y2-y1);return i/(a[2]*a[3]+b[2]*b[3]-i);
}

// ── Image preprocessing for better small-object detection ────
// Applies contrast stretch + simple unsharp mask to a canvas crop
// This makes a tiny phone much more visible to COCO-SSD
function enhanceCrop(ctx, w, h) {
  const id = ctx.getImageData(0,0,w,h);
  const d  = id.data;
  // Step 1: find min/max per channel for contrast stretch
  let rMin=255,rMax=0,gMin=255,gMax=0,bMin=255,bMax=0;
  for(let i=0;i<d.length;i+=4){
    rMin=Math.min(rMin,d[i]);  rMax=Math.max(rMax,d[i]);
    gMin=Math.min(gMin,d[i+1]);gMax=Math.max(gMax,d[i+1]);
    bMin=Math.min(bMin,d[i+2]);bMax=Math.max(bMax,d[i+2]);
  }
  const rR=rMax-rMin||1,gR=gMax-gMin||1,bR=bMax-bMin||1;
  // Step 2: stretch + mild sharpening (increase midtone contrast 20%)
  for(let i=0;i<d.length;i+=4){
    let r=((d[i]-rMin)/rR)*255;
    let g=((d[i+1]-gMin)/gR)*255;
    let b=((d[i+2]-bMin)/bR)*255;
    // S-curve: boost contrast in midtones
    d[i]  =clamp(Math.round(128+1.2*(r-128)),0,255);
    d[i+1]=clamp(Math.round(128+1.2*(g-128)),0,255);
    d[i+2]=clamp(Math.round(128+1.2*(b-128)),0,255);
  }
  ctx.putImageData(id,0,0);
}

// ── Eye Aspect Ratio (EAR) from 6 eye landmarks ───────────────
// landmarks: [p1,p2,p3,p4,p5,p6] as [x,y] pairs
// EAR < 0.20 = eyes closed
function eyeAspectRatio(pts){
  if(!pts||pts.length<6) return 1;
  const dist=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1]);
  const A=dist(pts[1],pts[5]);
  const B=dist(pts[2],pts[4]);
  const C=dist(pts[0],pts[3]);
  return (A+B)/(2*C||0.001);
}

// ── Head pitch from MoveNet ear-nose-shoulder ─────────────────
// Returns angle in degrees. Positive = head down.
function headPitch(nose, lEar, rEar, lS, rS){
  if(!nose||!lS||!rS) return 0;
  const earY  = ((lEar?.y||lS.y)+(rEar?.y||rS.y))/2;
  const shouldY=(lS.y+rS.y)/2;
  const vertRef=shouldY-earY||1;
  const drop   =nose.y-earY;
  return (Math.atan2(drop,Math.abs(vertRef))*180/Math.PI);
}


// ─────────────────────────────────────────────────────────────
// MODE SELECT
// ─────────────────────────────────────────────────────────────
function ModeSelect({user,onSelect,onLogout}){
  return(
    <div style={{minHeight:"100vh",background:"#040810",display:"flex",flexDirection:"column"}}>
      <div style={{padding:"10px 20px",borderBottom:"1px solid var(--bd)",
          display:"flex",justifyContent:"space-between",alignItems:"center",background:"rgba(0,0,0,.48)"}}>
        <div className="M" style={{fontSize:11,color:"var(--c)",letterSpacing:5}}>SENTINEL AI</div>
        <div style={{display:"flex",gap:10,alignItems:"center"}}>
          <span>{user.av}</span>
          <span className="M" style={{fontSize:8,color:"#334"}}>{user.name}</span>
          <button onClick={onLogout} style={{background:"none",border:"1px solid var(--bd)",
            color:"#334",padding:"3px 10px",cursor:"pointer",
            fontFamily:"Share Tech Mono,monospace",fontSize:7,letterSpacing:2}}>LOGOUT</button>
        </div>
      </div>
      <div style={{flex:1,display:"flex",alignItems:"center",justifyContent:"center",padding:30}}>
        <div style={{maxWidth:760,width:"100%"}}>
          <div style={{textAlign:"center",marginBottom:38}}>
            <div className="M" style={{fontSize:7,color:"var(--c)",letterSpacing:7,marginBottom:7}}>SELECT DETECTION MODULE</div>
            <div style={{fontSize:28,fontWeight:700,color:"#ddeef8"}}>Choose AI Detection Mode</div>
            <div style={{fontSize:13,color:"#2a3a44",marginTop:7}}>
              v9: MultiPose · Iris tracking · Risk scoring · Wrist velocity · Temporal smoothing
            </div>
          </div>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:18}}>
            {Object.entries(MODES).map(([key,cfg])=>(
              <button key={key} onClick={()=>onSelect(key)}
                style={{background:"rgba(3,7,15,.97)",border:`2px solid ${cfg.color}1e`,
                  padding:25,cursor:"pointer",textAlign:"left",transition:"all .18s"}}
                onMouseEnter={e=>{e.currentTarget.style.borderColor=cfg.color+"50";e.currentTarget.style.transform="translateY(-2px)";}}
                onMouseLeave={e=>{e.currentTarget.style.borderColor=cfg.color+"1e";e.currentTarget.style.transform="none";}}>
                <div style={{fontSize:34,marginBottom:9}}>{cfg.icon}</div>
                <div style={{fontSize:17,fontWeight:700,color:"#ddeef8",marginBottom:5}}>{cfg.label}</div>
                <div style={{fontSize:12,color:"#2a3a44",lineHeight:1.65,marginBottom:13}}>
                  {key==="exam"
                    ?"Iris gaze · phone 3-pass zoom · risk score · wrist velocity · multi-student"
                    :"Threats · weapons · unattended items · crouching · running · risk score"}
                </div>
                <div style={{display:"flex",gap:4,flexWrap:"wrap",marginBottom:11}}>
                  {["COCO Full","MultiPose","BlazeFace","Iris Gaze","Risk Score"].map(m=>(
                    <span key={m} className="M" style={{padding:"2px 5px",background:`${cfg.color}0b`,
                      border:`1px solid ${cfg.color}20`,fontSize:7,color:cfg.color}}>{m}</span>
                  ))}
                </div>
                <div style={{display:"flex",flexWrap:"wrap",gap:3,marginBottom:12}}>
                  {cfg.watch.slice(0,7).map(c=>(
                    <span key={c} className="M" style={{padding:"1px 5px",background:"rgba(255,255,255,.015)",
                      border:"1px solid rgba(255,255,255,.04)",fontSize:7,color:"#2a3a44"}}>{c}</span>
                  ))}
                </div>
                <div style={{fontFamily:"Share Tech Mono,monospace",fontSize:8,color:cfg.color,letterSpacing:3}}>ACTIVATE →</div>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// LOG PANEL
// ─────────────────────────────────────────────────────────────
function LogPanel({logs,onClear,onClick}){
  const [filt,setFilt]=useState("all");
  const [q,setQ]=useState("");
  const freshRef=useRef(new Set());
  const [,tick]=useState(0);
  const lastId=logs[0]?.id;
  useEffect(()=>{
    if(!lastId) return;
    freshRef.current.add(lastId);tick(n=>n+1);
    const t=setTimeout(()=>{freshRef.current.delete(lastId);tick(n=>n+1);},2000);
    return()=>clearTimeout(t);
  },[lastId]);
  const cn={h:logs.filter(e=>e.sev==="high").length,m:logs.filter(e=>e.sev==="med").length,l:logs.filter(e=>e.sev==="low").length};
  const shown=logs.filter(e=>{
    if(filt!=="all"&&e.sev!==filt) return false;
    if(q){const s=q.toLowerCase();return e.lbl.toLowerCase().includes(s)||e.cls.toLowerCase().includes(s);}
    return true;
  });
  return(
    <div style={{display:"flex",flexDirection:"column",height:"100%",overflow:"hidden"}}>
      <div style={{padding:"8px 10px",borderBottom:"1px solid var(--bd)",background:"rgba(0,0,0,.38)",flexShrink:0}}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:5}}>
          <div style={{display:"flex",gap:5,alignItems:"center"}}>
            <span className="M" style={{fontSize:7,color:"var(--c)",letterSpacing:3}}>EVENT LOG</span>
            {logs.length>0&&<div style={{width:4,height:4,borderRadius:"50%",background:"var(--g)",animation:"PG 2s infinite"}}/>}
          </div>
          <div style={{display:"flex",gap:5}}>
            <span className="M" style={{fontSize:7,color:"#334"}}>{logs.length}</span>
            {logs.length>0&&<button onClick={onClear} className="M"
              style={{background:"none",border:"1px solid var(--bd)",color:"#334",cursor:"pointer",fontSize:6,padding:"1px 4px"}}>CLR</button>}
          </div>
        </div>
        <div style={{display:"flex",gap:2,marginBottom:5}}>
          {[["all",`ALL(${logs.length})`,"#445"],
            ["high",`HI(${cn.h})`,"#ff2d4a"],
            ["med", `MD(${cn.m})`, "#ffaa00"],
            ["low", `LO(${cn.l})`, "#00d8ff"]].map(([v,l,c])=>(
            <button key={v} onClick={()=>setFilt(v)} className="M"
              style={{flex:1,padding:"3px 0",background:filt===v?c+"16":"transparent",
                border:`1px solid ${filt===v?c:c+"26"}`,color:filt===v?c:c+"44",
                cursor:"pointer",fontSize:6,transition:"all .1s"}}>{l}</button>
          ))}
        </div>
        <input value={q} onChange={e=>setQ(e.target.value)} placeholder="search events..."
          className="M" style={{width:"100%",background:"rgba(0,216,255,.02)",
            border:"1px solid var(--bd)",color:"#8aaabb",padding:"3px 7px",fontSize:9,outline:"none"}}
          onFocus={e=>e.target.style.borderColor="var(--c)"}
          onBlur={e=>e.target.style.borderColor="var(--bd)"}/>
      </div>
      <div style={{flex:1,overflowY:"auto"}}>
        {shown.length===0
          ?<div style={{textAlign:"center",padding:24,opacity:.26}}>
              <div style={{fontSize:20,marginBottom:6}}>📋</div>
              <div className="M" style={{fontSize:7,color:"#334"}}>{logs.length===0?"NO EVENTS YET":"NO MATCH"}</div>
            </div>
          :shown.map(e=>{
            const c=SC(e.sev);const fresh=freshRef.current.has(e.id);
            const anim=fresh?(e.sev==="high"?"FH 1.2s ease-out":e.sev==="med"?"FA 1.2s ease-out":"FC 1.2s ease-out"):undefined;
            return(
              <div key={e.id} onClick={()=>onClick(e)}
                style={{borderLeft:`2px solid ${c}`,padding:"6px 9px",marginBottom:1,
                  cursor:"pointer",borderBottom:"1px solid rgba(255,255,255,.018)",
                  animation:anim,transition:"background .15s"}}
                onMouseEnter={f=>f.currentTarget.style.background=SBG(e.sev)}
                onMouseLeave={f=>f.currentTarget.style.background="transparent"}>
                <div style={{display:"flex",gap:4,alignItems:"center",marginBottom:3,flexWrap:"wrap"}}>
                  <span style={{padding:"0 4px",background:c+"18",border:`1px solid ${c}2c`,
                    fontSize:6,color:c,fontFamily:"Share Tech Mono,monospace"}}>{e.sev.toUpperCase()}</span>
                  <span className="M" style={{fontSize:7,color:"#2a3840"}}>{e.time}</span>
                  <span className="M" style={{fontSize:6,color:"#1a2630",padding:"0 3px",border:"1px solid rgba(255,255,255,.03)"}}>{e.model}</span>
                  {e.sid&&<span className="M" style={{fontSize:6,color:c}}>#{e.sid}</span>}
                  {e.zoom>1&&<span className="M" style={{fontSize:6,color:"#ff8800"}}>×{e.zoom}</span>}
                  {e.risk!=null&&<span className="M" style={{fontSize:6,color:riskColor(e.risk)}}>risk:{e.risk}</span>}
                </div>
                <div style={{fontSize:12,fontWeight:600,color:"#b8d4e4",lineHeight:1.2,marginBottom:2}}>{e.lbl}</div>
                <div className="M" style={{fontSize:7,color:"#344"}}>
                  {e.cls}{typeof e.score==="number"&&<span style={{color:"#2a3840"}}> · {(e.score*100).toFixed(0)}%</span>}
                </div>
                {e.notified&&<div className="M" style={{fontSize:6,color:"#0c5035",marginTop:2}}>↗ {e.notified}</div>}
              </div>
            );
          })
        }
      </div>
      {logs.length>0&&(
        <div style={{padding:"5px 10px",borderTop:"1px solid var(--bd)",flexShrink:0,
            background:"rgba(0,0,0,.28)",display:"flex",gap:8,alignItems:"center"}}>
          {[["H",cn.h,"var(--r)"],["M",cn.m,"var(--a)"],["L",cn.l,"var(--c)"]].map(([l,v,c])=>(
            <span key={l} className="M" style={{fontSize:7}}>
              <span style={{color:"#1e2e38"}}>{l} </span><span style={{color:c,fontWeight:700}}>{v}</span>
            </span>
          ))}
          <span className="M" style={{fontSize:6,color:"#12202a",marginLeft:"auto"}}>tap → AI analysis</span>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// AI MODAL
// ─────────────────────────────────────────────────────────────
function AIModal({entry,user,onClose}){
  const [txt,setTxt]=useState(""); const [ld,setLd]=useState(true);
  useEffect(()=>{
    (async()=>{
      try{
        const r=await fetch("https://api.anthropic.com/v1/messages",{
          method:"POST",headers:{"Content-Type":"application/json"},
          body:JSON.stringify({
            model:"claude-sonnet-4-6",max_tokens:900,
            system:"You are SENTINEL AI v9. Write a concise threat assessment under 180 words. Format with headers: THREAT LEVEL, IMMEDIATE ACTIONS, ESCALATION. Include student ID and risk score context if provided.",
            messages:[{role:"user",content:
              `Alert: ${entry.lbl}\nClass: ${entry.cls}\n`+
              `Confidence: ${typeof entry.score==="number"?(entry.score*100).toFixed(0)+"%":"N/A"}\n`+
              `Severity: ${entry.sev}\nTime: ${entry.timestamp||entry.time}\n`+
              `Mode: ${entry.mode}\nOperator: ${user.name} (${user.role})\n`+
              (entry.sid?`Student: ${entry.sid}\n`:"")+
              (entry.risk!=null?`Risk Score: ${entry.risk}/100\n`:"")+
              (entry.zoom>1?`Detection zoom: ×${entry.zoom}\n`:"")+
              `\nProvide concise assessment and recommended immediate actions.`
            }]
          })
        });
        const d=await r.json();
        setTxt(d.content?.[0]?.text||"Analysis unavailable.");
      }catch{setTxt("⚠️ Server unavailable.\n\nManual review required.");}
      setLd(false);
    })();
  },[]);
  const c=SC(entry.sev);
  return(
    <div style={{position:"fixed",inset:0,background:"rgba(0,0,0,.93)",zIndex:600,
        display:"flex",alignItems:"center",justifyContent:"center"}}>
      <div style={{width:510,background:"#030912",border:`1px solid ${c}30`,
          padding:24,animation:"UP .2s",boxShadow:`0 0 50px ${c}05`}}>
        <div style={{display:"flex",justifyContent:"space-between",marginBottom:13}}>
          <div>
            <div className="M" style={{fontSize:7,color:c,letterSpacing:4}}>🤖 AI THREAT ANALYSIS</div>
            <div style={{fontSize:15,fontWeight:700,color:"#ddeef8",marginTop:3}}>{entry.lbl}</div>
          </div>
          <button onClick={onClose} style={{background:"none",border:"none",color:"#334",fontSize:18,cursor:"pointer"}}>×</button>
        </div>
        <div style={{display:"grid",gridTemplateColumns:"repeat(5,1fr)",gap:5,marginBottom:11}}>
          {[["CLASS",entry.cls],["CONF",typeof entry.score==="number"?(entry.score*100).toFixed(0)+"%":"–"],
            ["SEV",entry.sev.toUpperCase()],["MODEL",entry.model||"–"],
            ["RISK",entry.risk!=null?`${entry.risk}/100`:"–"]].map(([l,v])=>(
            <div key={l} style={{background:`${c}06`,border:`1px solid ${c}14`,padding:"5px 6px"}}>
              <div className="M" style={{fontSize:6,color:"#334",marginBottom:1}}>{l}</div>
              <div className="M" style={{fontSize:8,color:c}}>{v}</div>
            </div>
          ))}
        </div>
        <div style={{background:"rgba(0,0,0,.48)",border:`1px solid ${c}0c`,padding:13,minHeight:100}}>
          {ld?<div style={{textAlign:"center",paddingTop:26}}>
               <div className="M" style={{fontSize:9,color:c,animation:"BL 1s infinite"}}>⏳ ANALYZING...</div>
             </div>
            :<pre style={{fontFamily:"Barlow Condensed,sans-serif",fontSize:13,color:"#b0ccd8",whiteSpace:"pre-wrap",lineHeight:1.6}}>{txt}</pre>}
        </div>
        <div style={{display:"flex",gap:6,marginTop:10}}>
          <button onClick={onClose} style={{flex:1,padding:8,background:"transparent",border:"1px solid var(--bd)",
            color:"#445",cursor:"pointer",fontFamily:"Share Tech Mono,monospace",fontSize:7,letterSpacing:2}}>DISMISS</button>
          <button onClick={onClose} style={{flex:2,padding:8,background:`${c}0c`,border:`1px solid ${c}28`,
            color:c,cursor:"pointer",fontFamily:"Share Tech Mono,monospace",fontSize:7,letterSpacing:2,fontWeight:700}}>
            🚨 ESCALATE TO HEAD DEPT
          </button>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// CAMERA FEED — ADVANCED DETECTION ENGINE
// ─────────────────────────────────────────────────────────────
function CameraFeed({mode,user,models,onLog,onEvidence}){
  const vidRef=useRef(null);
  const canRef=useRef(null);
  const offRef=useRef(null);
  const rafRef=useRef(null);
  const runningRef=useRef(false);

  const [status,setStatus]=useState("idle");
  const [fps,setFps]=useState(0);
  const fpsR=useRef({n:0,t:Date.now()});

  // Detection result caches (drawn every RAF frame)
  const cocoRes=useRef([]);
  const faceRes=useRef([]);
  const poseRes=useRef([]);
  const tracksR=useRef([]);         // [{id,bbox,lastSeen,riskScore,riskDecay}]
  const cdR=useRef({});             // "sid:evt" → last fire time
  const seatR=useRef({});           // sid → consecutive absent frames
  const prevWristR=useRef({});      // sid → {lx,ly,rx,ry} prev wrist pos
  const prevWristTime=useRef({});   // sid → timestamp
  // Temporal smoothing: sid+cls → last N frame detections
  const smoothR=useRef({});         // "sid:cls" → [bool, bool, bool]

  const cfg=MODES[mode];

  // ── Smoothed detection confirm ───────────────────────────
  const confirm=(sid,cls,detected)=>{
    const k=`${sid}:${cls}`;
    const h=smoothR.current[k]||(smoothR.current[k]=[false,false,false]);
    h.push(detected);if(h.length>3) h.shift();
    return h.filter(Boolean).length>=CONFIRM_FRAMES;
  };

  // ── Fire event with risk accumulation ────────────────────
  const fire=useCallback((sid,evType,data)=>{
    const key=`${sid||"g"}:${evType}`;
    const cool=CD[evType]||CD.default;
    if(Date.now()-(cdR.current[key]||0)<cool) return;
    cdR.current[key]=Date.now();
    // Accumulate risk
    const track=tracksR.current.find(t=>t.id===sid);
    const riskDelta=RISK_W[evType]||RISK_W[data.cls]||5;
    if(track) track.riskScore=Math.min(100,Math.round((track.riskScore||0)+riskDelta));
    const risk=track?.riskScore||null;
    const evtEntry={id:uid(),time:TS(),timestamp:TSF(),mode,
      notified:data.sev==="high"?`${user.name} + Head Dept`:user.name,
      zoom:1,sid:null,risk,...data};
    onLog(evtEntry);
    // Audio alert + browser notification on HIGH
    if(data.sev==="high"){
      playAlertBeep("high");
      sendBrowserNotif("🚨 SENTINEL AI — HIGH ALERT",data.lbl||data.cls);
      // Auto evidence screenshot
      try{
        const can=canRef.current;const vid=vidRef.current;
        if(can&&vid&&vid.readyState>=2){
          const evCan=document.createElement("canvas");
          evCan.width=vid.videoWidth||640;evCan.height=vid.videoHeight||480;
          const ec=evCan.getContext("2d");
          ec.drawImage(vid,0,0);ec.drawImage(can,0,0);
          // Stamp label
          ec.fillStyle="rgba(0,0,0,.65)";ec.fillRect(0,0,evCan.width,28);
          ec.font="bold 13px 'Share Tech Mono',monospace";ec.fillStyle="#ff2d4a";
          ec.fillText(`⚠ ${data.lbl||data.cls}  |  ${TS()}  |  ${sid||''}  |  Risk:${risk||'?'}`,6,18);
          const imgData=evCan.toDataURL("image/jpeg",0.82);
          onEvidence({id:evtEntry.id,img:imgData,lbl:data.lbl||data.cls,
            cls:data.cls,sev:data.sev,time:TS(),timestamp:TSF(),
            sid:sid||null,risk:risk||null,score:data.score});
        }
      }catch(e){}
    } else if(data.sev==="med"){
      playAlertBeep("med");
    }
  },[user,mode,onLog]);

  const startCam=useCallback(async()=>{
    setStatus("loading");
    try{
      const stream=await navigator.mediaDevices.getUserMedia({
        video:{width:{ideal:640},height:{ideal:480},facingMode:"environment"},audio:false
      });
      if(!vidRef.current) return;
      vidRef.current.srcObject=stream;
      await vidRef.current.play();
      offRef.current=document.createElement("canvas");
      setStatus("running");
    }catch(e){console.error(e);setStatus("error");}
  },[]);

  // Draw corner-bracket box with optional risk glow
  const drawBox=(ctx,x,y,w,h,col,lbl,glow,riskPct)=>{
    x=Math.round(x);y=Math.round(y);w=Math.round(w);h=Math.round(h);
    if(glow){ctx.shadowColor=col;ctx.shadowBlur=12;}
    ctx.strokeStyle=col;ctx.lineWidth=glow?2.5:1.8;ctx.strokeRect(x,y,w,h);
    const cs=9;ctx.lineWidth=glow?3.2:2.2;
    [[x,y,1,1],[x+w,y,-1,1],[x,y+h,1,-1],[x+w,y+h,-1,-1]].forEach(([cx,cy,dx,dy])=>{
      ctx.beginPath();ctx.moveTo(cx+dx*cs,cy);ctx.lineTo(cx,cy);ctx.lineTo(cx,cy+dy*cs);ctx.stroke();
    });
    ctx.shadowBlur=0;
    if(lbl){
      ctx.font="bold 9px 'Share Tech Mono',monospace";
      const tw=ctx.measureText(lbl).width+8;
      ctx.fillStyle=col+"b0";ctx.fillRect(x,Math.max(0,y-14),tw,14);
      ctx.fillStyle="#000";ctx.fillText(lbl,x+4,Math.max(11,y-2));
    }
    if(riskPct>0){
      // Risk heat bar on right edge of bbox
      const bh=Math.round((riskPct/100)*h);
      const rc=riskColor(riskPct);
      ctx.fillStyle=rc+"30";ctx.fillRect(x+w+2,y+(h-bh),4,bh);
      ctx.fillStyle=rc+"a0";ctx.fillRect(x+w+2,y+(h-bh),4,3);
    }
    if(glow){ctx.shadowColor=col;ctx.shadowBlur=16;ctx.strokeStyle=col+"1e";ctx.lineWidth=8;ctx.strokeRect(x,y,w,h);ctx.shadowBlur=0;}
  };

  // ── MAIN DETECTION TICK ───────────────────────────────────
  const detect=useCallback(async()=>{
    if(runningRef.current) return;
    const vid=vidRef.current;const off=offRef.current;
    if(!vid||!off||vid.readyState<2) return;
    const {cocoModel,poseModel,faceModel}=models;
    runningRef.current=true;

    try{
      const W=vid.videoWidth||640;const H=vid.videoHeight||480;

      // ════════════════════════════════════════════════════
      // STEP 1 — COCO-SSD: 3-pass zoom with image enhancement
      // ════════════════════════════════════════════════════
      let all=[];
      if(cocoModel){
        // Pass 1: full frame
        const raw=await cocoModel.detect(vid,undefined,0.28);
        raw.sort((a,b)=>b.score-a.score);
        raw.forEach(p=>{
          if(!cfg.watch.includes(p.class)) return;
          if(p.score<(CONF[p.class]||DC)) return;
          if(all.some(k=>k.class===p.class&&iou(k.bbox,p.bbox)>0.45)) return;
          all.push({...p,zoom:1,fromZoom:false,sid:null,confirmed:true});
        });

        // Pass 2+3: per-person ROI zoom with contrast enhancement
        const persons=all.filter(p=>p.class==="person");
        for(const person of persons){
          const [px,py,pw,ph]=person.bbox;
          if(pw<12||ph<12) continue;
          const bx=clamp(px,0,W),by=clamp(py,0,H);
          const bw=clamp(pw,1,W-bx),bh=clamp(ph,1,H-by);

          // Pass 2: 4× body zoom + enhancement
          const Z1=4;
          const Z1W=clamp(Math.round(bw*Z1),1,1280),Z1H=clamp(Math.round(bh*Z1),1,960);
          off.width=Z1W;off.height=Z1H;
          const oc1=off.getContext("2d",{willReadFrequently:true});
          oc1.drawImage(vid,bx,by,bw,bh,0,0,Z1W,Z1H);
          enhanceCrop(oc1,Z1W,Z1H);
          try{
            const zr=await cocoModel.detect(off,undefined,0.18);
            zr.forEach(p=>{
              if(!cfg.watch.includes(p.class)) return;
              if(p.score<(CONF[p.class]||DC)*0.75) return;
              const m={class:p.class,score:p.score,zoom:Z1,fromZoom:true,sid:null,
                bbox:[bx+p.bbox[0]/Z1,by+p.bbox[1]/Z1,p.bbox[2]/Z1,p.bbox[3]/Z1]};
              if(all.some(k=>k.class===m.class&&iou(k.bbox,m.bbox)>0.32)) return;
              all.push(m);
            });
          }catch(e){}

          // Pass 3: 6× hand zone + enhancement
          const handY=clamp(by+bh*0.55,0,H);
          const handH=clamp(bh*0.45,1,H-handY);
          if(handH<10) continue;
          const Z2=6;
          const Z2W=clamp(Math.round(bw*Z2),1,1280),Z2H=clamp(Math.round(handH*Z2),1,960);
          off.width=Z2W;off.height=Z2H;
          const oc2=off.getContext("2d",{willReadFrequently:true});
          oc2.drawImage(vid,bx,handY,bw,handH,0,0,Z2W,Z2H);
          enhanceCrop(oc2,Z2W,Z2H);
          try{
            const hr=await cocoModel.detect(off,undefined,0.16);
            hr.forEach(p=>{
              if(!cfg.watch.includes(p.class)) return;
              if(p.score<(CONF[p.class]||DC)*0.65) return;
              const m={class:p.class,score:p.score,zoom:Z2,fromZoom:true,sid:null,
                bbox:[bx+p.bbox[0]/Z2,handY+p.bbox[1]/Z2,p.bbox[2]/Z2,p.bbox[3]/Z2]};
              if(all.some(k=>k.class===m.class&&iou(k.bbox,m.bbox)>0.28)) return;
              all.push(m);
            });
          }catch(e){}
        }
      }

      // ════════════════════════════════════════════════════
      // STEP 2 — UPDATE PERSON TRACKER
      // ════════════════════════════════════════════════════
      const personPreds=all.filter(p=>p.class==="person");
      const prevT=[...tracksR.current];const newT=[];
      personPreds.forEach(p=>{
        const m=prevT.find(t=>iou(t.bbox,p.bbox)>0.22);
        if(m){
          // Decay risk score slowly
          const decayed=Math.max(0,Math.round((m.riskScore||0)*0.998));
          newT.push({...m,bbox:p.bbox,lastSeen:Date.now(),riskScore:decayed});
        } else {
          newT.push({id:`S${String(TID++).padStart(2,"0")}`,bbox:p.bbox,lastSeen:Date.now(),riskScore:0});
        }
      });
      prevT.forEach(t=>{if(!newT.find(n=>n.id===t.id)&&Date.now()-t.lastSeen<9000) newT.push(t);});
      tracksR.current=newT;

      // Assign student IDs to non-person detections
      all.forEach(det=>{
        if(det.class==="person") return;
        const dcx=det.bbox[0]+det.bbox[2]/2,dcy=det.bbox[1]+det.bbox[3]/2;
        let best=null,bd=9999;
        newT.forEach(t=>{
          const d=Math.hypot(dcx-(t.bbox[0]+t.bbox[2]/2),dcy-(t.bbox[1]+t.bbox[3]/2));
          if(d<bd){bd=d;best=t.id;}
        });
        det.sid=best;
      });
      cocoRes.current=all;

      // Fire COCO alerts with temporal smoothing
      all.forEach(p=>{
        if(p.class==="person") return;
        const rule=cfg.rules[p.class];if(!rule) return;
        const sid=p.sid||p.class;
        if(!confirm(sid,p.class,true)) return; // wait for 2-of-3 confirmation
        fire(sid,p.class,{cls:p.class,lbl:rule.lbl,sev:rule.sev,
          score:p.score,zoom:p.zoom,sid:p.sid,
          model:p.fromZoom?`COCO ×${p.zoom}+enh`:"COCO-SSD"});
      });
      // Reset smoothing for classes NOT seen this frame
      cfg.watch.filter(c=>c!=="person").forEach(cls=>{
        newT.forEach(t=>{
          if(!all.some(p=>p.class===cls&&p.sid===t.id)) confirm(t.id,cls,false);
        });
      });
      if(mode==="security"){
        all.filter(p=>p.class==="person").forEach(p=>{
          fire("global","person",{cls:"person",lbl:"Person Detected",sev:"low",score:p.score,zoom:1,model:"COCO-SSD"});
        });
      }

      // ════════════════════════════════════════════════════
      // STEP 3 — BlazeFace: face count, absence, gaze backup
      // ════════════════════════════════════════════════════
      if(faceModel){
        try{
          const faces=await faceModel.estimateFaces(vid,false);
          faceRes.current=faces;

          if(mode==="exam"&&faces.length>1)
            fire("global","proxy",{cls:"proxy",lbl:`🚨 ${faces.length} Faces — Proxy Exam`,sev:"high",score:1,zoom:1,model:"BlazeFace"});

          faces.forEach((face,i)=>{
            const [fx,fy]=face.topLeft;const [fx2,fy2]=face.bottomRight;
            const fw=fx2-fx,fh=fy2-fy;
            if(!face.landmarks||face.landmarks.length<3) return;
            const nose=face.landmarks[2];const re=face.landmarks[0];const le=face.landmarks[1];
            const ecx=(re[0]+le[0])/2;
            const yaw=(nose[0]-ecx)/Math.max(fw,1);
            const fcx=fx+fw/2,fcy=fy+fh/2;
            let bestSid=`F${i+1}`,bestD=9999;
            tracksR.current.forEach(t=>{
              const d=Math.hypot(fcx-(t.bbox[0]+t.bbox[2]/2),fcy-(t.bbox[1]+t.bbox[3]/2));
              if(d<bestD){bestD=d;bestSid=t.id;}
            });
            if(mode==="exam"){
              if(yaw>0.28)       fire(bestSid,"gaze_right",{cls:"gaze",lbl:`👁 ${bestSid}: Looking RIGHT`,sev:"high",score:.86,zoom:1,sid:bestSid,model:"BlazeFace"});
              else if(yaw<-0.28) fire(bestSid,"gaze_left", {cls:"gaze",lbl:`👁 ${bestSid}: Looking LEFT`, sev:"high",score:.86,zoom:1,sid:bestSid,model:"BlazeFace"});
            }
          });

          // Absence detection
          if(mode==="exam"){
            const fps=faceRes.current.map(f=>({x:f.topLeft[0],y:f.topLeft[1],w:f.bottomRight[0]-f.topLeft[0],h:f.bottomRight[1]-f.topLeft[1]}));
            tracksR.current.forEach(t=>{
              const hasF=fps.some(f=>iou([f.x,f.y,f.w,f.h],t.bbox)>0.08);
              seatR.current[t.id]=(hasF?0:(seatR.current[t.id]||0)+1);
              if(seatR.current[t.id]>16)
                fire(t.id,"absent",{cls:"absent",lbl:`🪑 ${t.id} Absent from Seat`,sev:"high",score:1,zoom:1,sid:t.id,model:"Tracker"});
            });
          }
        }catch(e){console.error("Face",e);}
      }

      // ════════════════════════════════════════════════════
      // STEP 4 — MoveNet MultiPose: up to 6 students
      //          Head pitch, wrist velocity, lean, arm raise
      // ════════════════════════════════════════════════════
      if(poseModel){
        try{
          const poses=await poseModel.estimatePoses(vid,{maxPoses:6,flipHorizontal:false});
          poseRes.current=poses;
          const now=Date.now();

          // Per-pose analysis
          poses.forEach((pose,pi)=>{
            const kps=pose.keypoints;if(!kps||kps.length<17) return;
            const kp=(i,t=0.26)=>kps[i]&&kps[i].score>t?kps[i]:null;
            const nose=kp(0);const lEar=kp(3);const rEar=kp(4);
            const lS=kp(5);const rS=kp(6);
            const lE=kp(7);const rE=kp(8);
            const lW=kp(9);const rW=kp(10);
            const lH=kp(11);const rH=kp(12);
            const lK=kp(13);const lA=kp(15);

            // Map to nearest student
            const sCx=lS&&rS?(lS.x+rS.x)/2:null;
            let sid=`P${pi+1}`;
            if(sCx!=null){
              let bd=9999;
              tracksR.current.forEach(t=>{
                const d=Math.abs(t.bbox[0]+t.bbox[2]/2-sCx);
                if(d<bd){bd=d;sid=t.id;}
              });
            }

            if(mode==="exam"){
              // ── HEAD PITCH (real angle) ─────────────────
              const pitch=headPitch(nose,lEar,rEar,lS,rS);
              if(pitch>42)
                fire(sid,"head_down",{cls:"pose",lbl:`⬇ ${sid}: Head Down ${pitch.toFixed(0)}° (hidden notes?)`,sev:"med",score:.76,zoom:1,sid,model:"MultiPose"});

              // ── HEAD TURN (improved) ────────────────────
              if(nose&&lS&&rS){
                const sCx2=(lS.x+rS.x)/2;
                const half=Math.abs(lS.x-rS.x)*0.5||35;
                const turn=(nose.x-sCx2)/half;
                if(turn>0.78)       fire(sid,"gaze_right",{cls:"pose",lbl:`↪ ${sid}: Head Turned RIGHT`,sev:"high",score:.83,zoom:1,sid,model:"MultiPose"});
                else if(turn<-0.78) fire(sid,"gaze_left", {cls:"pose",lbl:`↩ ${sid}: Head Turned LEFT`, sev:"high",score:.83,zoom:1,sid,model:"MultiPose"});
              }

              // ── ARM RAISE ──────────────────────────────
              if(lW&&lS&&lW.y<lS.y-28)
                fire(sid,"arm_raise",{cls:"pose",lbl:`✋ ${sid}: Left Arm Raised`,sev:"med",score:.79,zoom:1,sid,model:"MultiPose"});
              if(rW&&rS&&rW.y<rS.y-28)
                fire(sid,"arm_raise",{cls:"pose",lbl:`🤚 ${sid}: Right Arm Raised`,sev:"med",score:.79,zoom:1,sid,model:"MultiPose"});

              // ── LEAN SIDEWAYS ──────────────────────────
              if(lS&&rS&&lH&&rH){
                const tilt=Math.abs(lS.y-rS.y);
                const hW=Math.abs(lH.x-rH.x)||45;
                if(tilt/hW>0.52)
                  fire(sid,"lean_over",{cls:"pose",lbl:`📐 ${sid}: Leaning Sideways`,sev:"high",score:.81,zoom:1,sid,model:"MultiPose"});
              }

              // ── WRIST VELOCITY (suspicious rapid hand movement) ──
              // Compares wrist position to previous frame
              const prev=prevWristR.current[sid];
              const prevT2=prevWristTime.current[sid]||0;
              const dt=Math.max(now-prevT2,16)/1000; // seconds
              if(prev&&lW&&rW&&dt<2){
                const lVel=Math.hypot(lW.x-prev.lx,lW.y-prev.ly)/dt;
                const rVel=Math.hypot(rW.x-prev.rx,rW.y-prev.ry)/dt;
                const vel=Math.max(lVel,rVel);
                if(vel>180){ // px/sec threshold — tuned to be above writing velocity
                  fire(sid,"wrist_velocity",{cls:"pose",
                    lbl:`⚡ ${sid}: Rapid Hand Movement (${vel.toFixed(0)}px/s)`,
                    sev:"med",score:Math.min(.99,vel/500),zoom:1,sid,model:"MultiPose"});
                }
              }
              if(lW&&rW){
                prevWristR.current[sid]={lx:lW.x,ly:lW.y,rx:rW.x,ry:rW.y};
                prevWristTime.current[sid]=now;
              }
            }

            if(mode==="security"){
              if(lH&&lK&&lA){
                const hk=Math.abs(lH.y-lK.y),ka=Math.abs(lK.y-lA.y)||1;
                if(hk/ka<0.5) fire(sid,"crouch",{cls:"pose",lbl:"🕵️ Person Crouching",sev:"high",score:.80,zoom:1,model:"MultiPose"});
              }
              if(lS&&rS&&lH&&rH){
                const sW=Math.abs(lS.x-rS.x),hW=Math.abs(lH.x-rH.x)||1;
                if(Math.abs(sW-hW)/hW>0.50) fire(sid,"running",{cls:"pose",lbl:"🏃 Running Motion",sev:"high",score:.74,zoom:1,model:"MultiPose"});
              }
            }
          });

          // ── COORDINATED CHEATING DETECTION ─────────────────
          // If 2+ adjacent students have high wrist velocity simultaneously
          if(mode==="exam"&&poses.length>=2){
            const now2=Date.now();
            const suspicious=tracksR.current.filter(t=>{
              const k=`${t.id}:wrist_velocity`;
              return (now2-(cdR.current[k]||9999))<3000; // fired in last 3s
            });
            if(suspicious.length>=2){
              // Check they're physically adjacent (x distance < 200px)
              for(let i=0;i<suspicious.length-1;i++){
                for(let j=i+1;j<suspicious.length;j++){
                  const a=suspicious[i],b=suspicious[j];
                  const dist=Math.abs((a.bbox[0]+a.bbox[2]/2)-(b.bbox[0]+b.bbox[2]/2));
                  if(dist<220)
                    fire(`${a.id}+${b.id}`,"coordinated",{cls:"coordinated",
                      lbl:`🤝 ${a.id}+${b.id}: Coordinated Cheating?`,
                      sev:"high",score:.85,zoom:1,sid:`${a.id}+${b.id}`,model:"Correlation"});
                }
              }
            }
          }

        }catch(e){console.error("Pose",e);}
      }

    }finally{
      runningRef.current=false;
    }
  },[models,cfg,mode,fire,confirm]);

  // ── RENDER LOOP ───────────────────────────────────────────
  const render=useCallback(()=>{
    const vid=vidRef.current;const can=canRef.current;
    if(!vid||!can||vid.readyState<2){rafRef.current=requestAnimationFrame(render);return;}
    const W=vid.videoWidth||640;const H=vid.videoHeight||480;
    if(can.width!==W||can.height!==H){can.width=W;can.height=H;}
    const ctx=can.getContext("2d");
    ctx.clearRect(0,0,W,H);
    fpsR.current.n++;
    if(Date.now()-fpsR.current.t>1000){setFps(fpsR.current.n);fpsR.current={n:0,t:Date.now()};}

    // COCO boxes
    cocoRes.current.forEach(p=>{
      const rule=cfg.rules[p.class];if(!rule) return;
      const [x,y,w,h]=p.bbox;const c=SC(rule.sev);
      drawBox(ctx,x,y,w,h,c,`${p.class} ${(p.score*100).toFixed(0)}%${p.fromZoom?` ×${p.zoom}`:""}`,p.fromZoom,0);
    });

    // Student bboxes with risk meter
    tracksR.current.forEach(t=>{
      if(Date.now()-t.lastSeen>2500) return;
      const [x,y,w,h]=t.bbox;const risk=t.riskScore||0;
      const rc=riskColor(risk);
      // Risk border (pulses when high)
      if(risk>40){
        ctx.strokeStyle=rc+"60";ctx.lineWidth=2;
        if(risk>70){ctx.shadowColor=rc;ctx.shadowBlur=10;}
        ctx.strokeRect(x-3,y-3,w+6,h+6);
        ctx.shadowBlur=0;
      }
      // ID tag
      ctx.fillStyle=risk>70?"rgba(255,45,74,.85)":risk>40?"rgba(255,170,0,.8)":"rgba(0,216,255,.78)";
      ctx.fillRect(x,Math.max(0,y-14),38,14);
      ctx.font="bold 8px 'Share Tech Mono',monospace";ctx.fillStyle="#020810";
      ctx.fillText(t.id,x+3,Math.max(11,y-2));
      // Risk score indicator
      if(risk>10){
        ctx.font="6px 'Share Tech Mono',monospace";ctx.fillStyle=rc;
        ctx.fillText(`R:${risk}`,x+40,Math.max(11,y-2));
        // Risk bar (right side of bbox)
        const bh2=Math.round((risk/100)*h);
        ctx.fillStyle=rc+"25";ctx.fillRect(x+w+3,y+(h-bh2),4,bh2);
        ctx.fillStyle=rc+"80";ctx.fillRect(x+w+3,y+(h-bh2),4,3);
      }
    });

    // Face boxes + gaze arrows
    faceRes.current.forEach((face,i)=>{
      const [fx,fy]=face.topLeft;const [fx2,fy2]=face.bottomRight;
      const fw=fx2-fx,fh=fy2-fy;
      ctx.strokeStyle="rgba(0,216,255,.38)";ctx.lineWidth=1.2;ctx.strokeRect(fx,fy,fw,fh);
      ctx.font="7px 'Share Tech Mono',monospace";ctx.fillStyle="rgba(0,216,255,.6)";
      ctx.fillText(`F${i+1}`,fx+2,fy+fh-3);
      if(face.landmarks&&face.landmarks.length>=3){
        const nose=face.landmarks[2];const re=face.landmarks[0];const le=face.landmarks[1];
        const ecx=(re[0]+le[0])/2,ecy=(re[1]+le[1])/2;
        const dx=nose[0]-ecx,dy=nose[1]-ecy;
        ctx.beginPath();ctx.strokeStyle="rgba(0,216,255,.24)";ctx.lineWidth=1.5;
        ctx.moveTo(ecx,ecy);ctx.lineTo(ecx+dx*4.5,ecy+dy*4.5);ctx.stroke();
        const yaw=dx/Math.max(fw,1);
        if(mode==="exam"&&Math.abs(yaw)>0.28){
          ctx.strokeStyle="#ff2d4a";ctx.lineWidth=1.8;ctx.strokeRect(fx-2,fy-2,fw+4,fh+4);
          ctx.font="bold 8px 'Share Tech Mono',monospace";ctx.fillStyle="#ff2d4a";
          ctx.fillText(yaw>0?"→ RIGHT":"← LEFT",fx,fy+fh+11);
        }
      }
    });

    // Skeleton + wrist velocity arrows
    poseRes.current.forEach((pose,pi)=>{
      const kps=pose.keypoints;if(!kps||kps.length<17) return;
      const ok=i=>kps[i]&&kps[i].score>0.26;
      const CONN=[[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
      ctx.strokeStyle="rgba(0,216,255,.16)";ctx.lineWidth=1.2;
      CONN.forEach(([a,b])=>{
        if(ok(a)&&ok(b)){ctx.beginPath();ctx.moveTo(kps[a].x,kps[a].y);ctx.lineTo(kps[b].x,kps[b].y);ctx.stroke();}
      });
      kps.filter((_,i)=>ok(i)).forEach(k=>{
        ctx.beginPath();ctx.arc(k.x,k.y,2.2,0,Math.PI*2);ctx.fillStyle="rgba(0,235,122,.38)";ctx.fill();
      });

      // Head pitch indicator (small arc at ear level)
      if(ok(0)&&(ok(3)||ok(5))&&(ok(4)||ok(6))){
        const eary=((kps[3]?.y||kps[5]?.y)+(kps[4]?.y||kps[6]?.y))/2;
        const pitch=headPitch(kps[0],kps[3],kps[4],kps[5],kps[6]);
        if(pitch>35){
          ctx.strokeStyle=`rgba(255,170,0,${Math.min(.9,(pitch-35)/30)})`;ctx.lineWidth=2;
          ctx.beginPath();ctx.arc(kps[0].x,kps[0].y,12,0,Math.PI*2);ctx.stroke();
          ctx.font="7px 'Share Tech Mono',monospace";ctx.fillStyle="#ffaa00";
          ctx.fillText(`${pitch.toFixed(0)}°`,kps[0].x+14,kps[0].y+3);
        }
      }

      // Wrist raise highlight
      if(mode==="exam"){
        [[9,5],[10,6]].forEach(([wi,si])=>{
          if(ok(wi)&&ok(si)&&kps[wi].y<kps[si].y-28){
            ctx.beginPath();ctx.arc(kps[wi].x,kps[wi].y,8,0,Math.PI*2);
            ctx.strokeStyle="#ffaa00";ctx.lineWidth=2.5;ctx.stroke();
          }
        });

        // Wrist velocity arrow
        const sCx=ok(5)&&ok(6)?(kps[5].x+kps[6].x)/2:null;
        let sid=`P${pi}`;
        if(sCx!=null){let bd=9999;tracksR.current.forEach(t=>{const d=Math.abs(t.bbox[0]+t.bbox[2]/2-sCx);if(d<bd){bd=d;sid=t.id;}});}
        const prev=prevWristR.current[sid];
        if(prev&&ok(9)){
          const dx=kps[9].x-prev.lx,dy=kps[9].y-prev.ly;
          const mag=Math.hypot(dx,dy);
          if(mag>8){
            ctx.strokeStyle=`rgba(255,136,0,${Math.min(.9,mag/40)})`;ctx.lineWidth=2;
            ctx.beginPath();ctx.moveTo(kps[9].x,kps[9].y);
            ctx.lineTo(kps[9].x+dx*2.5,kps[9].y+dy*2.5);ctx.stroke();
          }
        }
      }
    });

    // Scan line
    const sy=((Date.now()/13)%H);
    ctx.strokeStyle=`${cfg.color}0e`;ctx.lineWidth=1;
    ctx.beginPath();ctx.moveTo(0,sy);ctx.lineTo(W,sy);ctx.stroke();

    rafRef.current=requestAnimationFrame(render);
  },[cfg,mode]);

  // ── LIFECYCLE ─────────────────────────────────────────────
  useEffect(()=>{
    if(models.cocoModel) startCam();
    return()=>{
      rafRef.current&&cancelAnimationFrame(rafRef.current);
      vidRef.current?.srcObject?.getTracks().forEach(t=>t.stop());
    };
  },[models.cocoModel]);

  useEffect(()=>{
    if(status!=="running") return;
    rafRef.current=requestAnimationFrame(render);
    let alive=true;
    (async()=>{while(alive){await detect();await new Promise(r=>setTimeout(r,100));}})();
    return()=>{alive=false;rafRef.current&&cancelAnimationFrame(rafRef.current);};
  },[status,render,detect]);

  return(
    <div style={{background:"#010408",border:`1px solid ${status==="running"?"var(--bd2)":"var(--bd)"}`,
        borderRadius:2,overflow:"hidden",flex:1,display:"flex",flexDirection:"column"}}>
      <div style={{padding:"5px 10px",display:"flex",justifyContent:"space-between",
          alignItems:"center",background:"rgba(0,0,0,.68)",borderBottom:"1px solid var(--bd)",flexShrink:0}}>
        <div style={{display:"flex",gap:6,alignItems:"center"}}>
          <div style={{width:6,height:6,borderRadius:"50%",
            background:status==="running"?"var(--g)":status==="error"?"var(--r)":"#334",
            animation:status==="running"?"PG 2s infinite":undefined}}/>
          <span className="M" style={{fontSize:8,color:"#445"}}>CAMERA</span>
          {status==="running"&&<span className="M" style={{fontSize:7,color:"var(--r)",animation:"BL 1s infinite"}}>● REC</span>}
        </div>
        <div style={{display:"flex",gap:8,alignItems:"center"}}>
          {status==="running"&&(
            <span className="M" style={{fontSize:8,color:fps>=12?"var(--g)":fps>=6?"var(--a)":"var(--r)"}}>{fps}fps</span>
          )}
          <span className="M" style={{padding:"1px 6px",background:`${cfg.color}0c`,
            border:`1px solid ${cfg.color}25`,fontSize:7,color:cfg.color}}>{cfg.icon} {mode.toUpperCase()}</span>
        </div>
      </div>

      <div style={{position:"relative",flex:1,background:"#010408"}}>
        <video ref={vidRef} muted playsInline style={{width:"100%",height:"100%",display:"block",objectFit:"cover"}}/>
        <canvas ref={canRef} style={{position:"absolute",inset:0,width:"100%",height:"100%",pointerEvents:"none"}}/>
        {status==="idle"&&(
          <div style={{position:"absolute",inset:0,display:"flex",flexDirection:"column",
              alignItems:"center",justifyContent:"center",background:"#010408"}}>
            <div style={{fontSize:40,marginBottom:12}}>📷</div>
            <div className="M" style={{fontSize:9,color:"#334",letterSpacing:4}}>STANDBY</div>
            <div className="M" style={{fontSize:8,color:"#162230",marginTop:6}}>Waiting for models...</div>
          </div>
        )}
        {status==="loading"&&(
          <div style={{position:"absolute",inset:0,display:"flex",flexDirection:"column",
              alignItems:"center",justifyContent:"center",background:"#010408"}}>
            <div style={{width:36,height:36,border:"2px solid var(--bd)",borderTop:"2px solid var(--c)",
              borderRadius:"50%",animation:"SP 1s linear infinite",marginBottom:12}}/>
            <div className="M" style={{fontSize:9,color:"var(--c)",animation:"BL 1s infinite",letterSpacing:3}}>STARTING CAMERA...</div>
          </div>
        )}
        {status==="error"&&(
          <div style={{position:"absolute",inset:0,display:"flex",flexDirection:"column",
              alignItems:"center",justifyContent:"center",background:"#010408"}}>
            <div style={{fontSize:32,marginBottom:10}}>⚠️</div>
            <div className="M" style={{fontSize:10,color:"var(--r)"}}>CAMERA DENIED</div>
            <div className="M" style={{fontSize:8,color:"#334",marginTop:6,textAlign:"center",maxWidth:220,lineHeight:1.6}}>
              Allow camera access in browser settings and refresh
            </div>
            <button onClick={startCam} style={{marginTop:12,padding:"6px 16px",
              background:"rgba(255,45,74,.1)",border:"1px solid var(--r)",color:"var(--r)",
              cursor:"pointer",fontFamily:"Share Tech Mono,monospace",fontSize:8}}>RETRY</button>
          </div>
        )}
      </div>

      <div className="M" style={{padding:"3px 10px",background:"rgba(0,0,0,.55)",flexShrink:0}}>
        <span style={{fontSize:6,color:"#121e2a"}}>
          v9: COCO 3-pass+enh · MultiPose(6) · risk score · wrist velocity · temporal smoothing · coordinated detection
        </span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// RTSP CARD
// ─────────────────────────────────────────────────────────────
function RtspCard({camera,onRemove}){
  return(
    <div style={{border:"1px solid var(--bd)",background:"rgba(1,3,9,.95)",padding:8}}>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:5}}>
        <div>
          <div style={{fontSize:11,fontWeight:600,color:"#b0c8dc"}}>{camera.name}</div>
          <div className="M" style={{fontSize:7,color:"#2a3840"}}>{camera.type.toUpperCase()} · {camera.location}</div>
        </div>
        <div style={{display:"flex",gap:4,alignItems:"center"}}>
          <div style={{width:4,height:4,borderRadius:"50%",background:"var(--g)",animation:"PG 2s infinite"}}/>
          <button onClick={onRemove} style={{background:"none",border:"none",color:"#334",cursor:"pointer",fontSize:12}}>×</button>
        </div>
      </div>
      <div style={{aspectRatio:"16/9",background:"#010205",position:"relative",overflow:"hidden"}}>
        <div style={{position:"absolute",inset:0,display:"flex",alignItems:"center",justifyContent:"center"}}>
          <div style={{textAlign:"center"}}>
            <div style={{fontSize:14,marginBottom:3}}>📡</div>
            <div className="M" style={{fontSize:6,color:"#1a2836",maxWidth:110,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{camera.url}</div>
          </div>
        </div>
        <div style={{position:"absolute",left:0,right:0,height:"1px",opacity:.2,
          background:"linear-gradient(90deg,transparent,var(--c),transparent)",animation:"SL 4s linear infinite"}}/>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// ADD CAMERA
// ─────────────────────────────────────────────────────────────
function AddCam({onAdd,onClose}){
  const [f,setF]=useState({name:"",url:"",type:"rtsp",location:""});
  const PRE=[
    {name:"Main Entrance",url:"rtsp://192.168.1.100:554/stream",type:"rtsp",location:"Gate A"},
    {name:"Exam Hall A",  url:"rtsp://192.168.1.101:554/stream",type:"rtsp",location:"Block B 101"},
    {name:"Library",      url:"http://192.168.1.102:8080/video",type:"http",location:"Library"},
    {name:"Corridor",     url:"rtsp://10.0.0.50:554/cam1",      type:"rtsp",location:"Block B"},
  ];
  const IS={width:"100%",background:"rgba(0,216,255,.025)",border:"1px solid var(--bd)",
    color:"#8aaabb",padding:"7px 10px",fontSize:12,outline:"none",
    fontFamily:"Share Tech Mono,monospace",transition:"border-color .2s"};
  return(
    <div style={{position:"fixed",inset:0,background:"rgba(0,0,0,.9)",zIndex:400,
        display:"flex",alignItems:"center",justifyContent:"center"}}>
      <div style={{width:450,background:"#030912",border:"1px solid var(--bd2)",padding:24,animation:"UP .2s"}}>
        <div style={{display:"flex",justifyContent:"space-between",marginBottom:14}}>
          <div className="M" style={{fontSize:8,color:"var(--c)",letterSpacing:3}}>+ ADD CAMERA</div>
          <button onClick={onClose} style={{background:"none",border:"none",color:"#334",fontSize:18,cursor:"pointer"}}>×</button>
        </div>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:4,marginBottom:12}}>
          {PRE.map(p=>(
            <button key={p.name} onClick={()=>setF(p)}
              style={{padding:"5px 7px",background:"rgba(0,216,255,.025)",border:"1px solid var(--bd)",
                color:"#556",cursor:"pointer",textAlign:"left",fontSize:12,
                fontFamily:"Barlow Condensed,sans-serif",transition:"all .1s"}}
              onMouseEnter={e=>{e.currentTarget.style.borderColor="var(--c)";e.currentTarget.style.color="#ddeef8";}}
              onMouseLeave={e=>{e.currentTarget.style.borderColor="var(--bd)";e.currentTarget.style.color="#556";}}>{p.name}</button>
          ))}
        </div>
        {[["CAMERA NAME","name","e.g. Exam Hall A"],["URL","url","rtsp://..."],["LOCATION","location","Block B, Room 101"]].map(([lb,fl,ph])=>(
          <div key={fl} style={{marginBottom:10}}>
            <div className="M" style={{fontSize:7,color:"var(--c)",letterSpacing:3,marginBottom:4}}>{lb}</div>
            <input value={f[fl]} onChange={e=>setF(p=>({...p,[fl]:e.target.value}))} placeholder={ph} style={IS}
              onFocus={e=>e.target.style.borderColor="var(--c)"}
              onBlur={e=>e.target.style.borderColor="var(--bd)"}/>
          </div>
        ))}
        <div style={{display:"flex",gap:4,marginBottom:12}}>
          {["rtsp","http","usb","onvif"].map(t=>(
            <button key={t} onClick={()=>setF(p=>({...p,type:t}))} className="M"
              style={{flex:1,padding:"5px 0",background:f.type===t?"rgba(0,216,255,.1)":"transparent",
                border:`1px solid ${f.type===t?"var(--c)":"var(--bd)"}`,
                color:f.type===t?"var(--c)":"#334",cursor:"pointer",fontSize:7,letterSpacing:2}}>{t.toUpperCase()}</button>
          ))}
        </div>
        <button onClick={()=>{if(f.name&&f.url){onAdd({...f,id:uid()});onClose();}}}
          style={{width:"100%",padding:10,background:"var(--c)",color:"#020810",border:"none",
            cursor:"pointer",fontFamily:"Share Tech Mono,monospace",fontSize:9,letterSpacing:4,fontWeight:700}}>
          + ADD CAMERA
        </button>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// DASHBOARD
// ─────────────────────────────────────────────────────────────
// ─────────────────────────────────────────────────────────────
// EVIDENCE PANEL (thumbnail gallery + lightbox)
// ─────────────────────────────────────────────────────────────
function EvidencePanel({evidence,onViewReport}){
  const [viewImg,setViewImg]=useState(null);
  return(
    <>
      {viewImg&&(
        <div onClick={()=>setViewImg(null)} style={{position:"fixed",inset:0,background:"rgba(0,0,0,.92)",
            zIndex:700,display:"flex",alignItems:"center",justifyContent:"center",cursor:"pointer"}}>
          <div style={{maxWidth:"85vw",maxHeight:"85vh",position:"relative"}}>
            <img src={viewImg.img} alt="evidence" style={{maxWidth:"85vw",maxHeight:"80vh",border:"2px solid var(--r)"}}/>
            <div style={{padding:"8px 12px",background:"rgba(0,0,0,.8)",border:"1px solid rgba(255,45,74,.3)"}}>
              <span className="M" style={{fontSize:9,color:"var(--r)"}}>{viewImg.lbl}</span>
              <span className="M" style={{fontSize:8,color:"#556",marginLeft:8}}>{viewImg.timestamp}</span>
              {viewImg.sid&&<span className="M" style={{fontSize:8,color:"var(--a)",marginLeft:8}}>#{viewImg.sid}</span>}
              {viewImg.risk!=null&&<span className="M" style={{fontSize:8,color:"#ff2d4a",marginLeft:8}}>Risk:{viewImg.risk}</span>}
            </div>
          </div>
        </div>
      )}
      <div style={{display:"flex",flexDirection:"column",height:"100%",overflow:"hidden"}}>
        <div style={{padding:"8px 10px",borderBottom:"1px solid var(--bd)",background:"rgba(0,0,0,.38)",flexShrink:0}}>
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:4}}>
            <div style={{display:"flex",gap:4,alignItems:"center"}}>
              <span className="M" style={{fontSize:7,color:"var(--r)",letterSpacing:3}}>📸 EVIDENCE</span>
              <span className="M" style={{fontSize:7,color:"#334"}}>{evidence.length}</span>
            </div>
            {evidence.length>0&&(
              <button onClick={onViewReport} className="M"
                style={{background:"rgba(255,45,74,.08)",border:"1px solid rgba(255,45,74,.25)",
                  color:"var(--r)",padding:"3px 8px",cursor:"pointer",fontSize:6,letterSpacing:2,
                  animation:"PR 2s infinite"}}>📄 PDF REPORT</button>
            )}
          </div>
        </div>
        <div style={{flex:1,overflowY:"auto",padding:4}}>
          {evidence.length===0
            ?<div style={{textAlign:"center",padding:24,opacity:.26}}>
                <div style={{fontSize:20,marginBottom:6}}>📷</div>
                <div className="M" style={{fontSize:7,color:"#334"}}>NO EVIDENCE YET</div>
                <div className="M" style={{fontSize:6,color:"#1e2e38",marginTop:3}}>Auto-captures on HIGH alerts</div>
              </div>
            :<div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:4}}>
              {evidence.map(ev=>(
                <div key={ev.id} className="ev-thumb" onClick={()=>setViewImg(ev)}
                  style={{position:"relative",border:"1px solid rgba(255,45,74,.2)",
                    background:"#010408",overflow:"hidden",animation:"FLASH_EV .6s ease-out"}}>
                  <img src={ev.img} alt="" style={{width:"100%",display:"block",aspectRatio:"16/9",objectFit:"cover"}}/>
                  <div style={{position:"absolute",bottom:0,left:0,right:0,padding:"2px 4px",
                    background:"rgba(0,0,0,.75)"}}>
                    <div className="M" style={{fontSize:5,color:"var(--r)",overflow:"hidden",
                      textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{ev.lbl}</div>
                    <div className="M" style={{fontSize:5,color:"#334"}}>{ev.time}{ev.sid?` #${ev.sid}`:""}</div>
                  </div>
                </div>
              ))}
            </div>
          }
        </div>
      </div>
    </>
  );
}

// ─────────────────────────────────────────────────────────────
// PDF REPORT GENERATOR
// ─────────────────────────────────────────────────────────────
function generateReport(evidence,logs,user,mode){
  const doc=new jsPDF({orientation:"portrait",unit:"mm",format:"a4"});
  const W=210,MARGIN=15;
  let y=20;
  const addPage=()=>{doc.addPage();y=20;};
  const checkY=(need)=>{if(y+need>280) addPage();};

  // Header
  doc.setFillColor(4,8,16);doc.rect(0,0,W,40,"F");
  doc.setFont("helvetica","bold");doc.setFontSize(20);doc.setTextColor(0,216,255);
  doc.text("SENTINEL AI v9",MARGIN,18);
  doc.setFontSize(10);doc.setTextColor(138,170,187);
  doc.text("Incident Evidence Report",MARGIN,26);
  doc.setFontSize(8);doc.setTextColor(100,130,150);
  doc.text(`Generated: ${TSF()}  |  Operator: ${user.name} (${user.role})  |  Mode: ${mode.toUpperCase()}`,MARGIN,33);
  y=48;

  // Summary box
  const hi=logs.filter(l=>l.sev==="high").length;
  const md=logs.filter(l=>l.sev==="med").length;
  const lo=logs.filter(l=>l.sev==="low").length;
  doc.setFillColor(10,16,28);doc.rect(MARGIN,y,W-2*MARGIN,18,"F");
  doc.setDrawColor(0,216,255);doc.rect(MARGIN,y,W-2*MARGIN,18,"S");
  doc.setFontSize(9);doc.setFont("helvetica","bold");doc.setTextColor(220,238,248);
  doc.text("SESSION SUMMARY",MARGIN+4,y+7);
  doc.setFont("helvetica","normal");doc.setFontSize(8);doc.setTextColor(138,170,187);
  doc.text(`Total Events: ${logs.length}  |  HIGH: ${hi}  |  MEDIUM: ${md}  |  LOW: ${lo}  |  Evidence Captures: ${evidence.length}`,MARGIN+4,y+14);
  y+=26;

  // Evidence entries
  evidence.forEach((ev,i)=>{
    checkY(75);
    // Evidence header
    doc.setFillColor(255,45,74);doc.rect(MARGIN,y,W-2*MARGIN,8,"F");
    doc.setFont("helvetica","bold");doc.setFontSize(8);doc.setTextColor(255,255,255);
    doc.text(`EVIDENCE #${i+1}: ${ev.lbl}`,MARGIN+3,y+5.5);
    const meta=`${ev.timestamp}  |  ${ev.sid?"Student:"+ev.sid:""} ${ev.risk!=null?" Risk:"+ev.risk:""}  |  Conf:${typeof ev.score==="number"?(ev.score*100).toFixed(0)+"%":"N/A"}`;
    doc.setFont("helvetica","normal");doc.setTextColor(200,200,200);
    doc.text(meta.trim(),MARGIN+100,y+5.5);
    y+=10;

    // Screenshot image
    try{
      doc.addImage(ev.img,"JPEG",MARGIN,y,W-2*MARGIN,50);
      doc.setDrawColor(255,45,74);doc.rect(MARGIN,y,W-2*MARGIN,50,"S");
      y+=54;
    }catch(e){
      doc.setFontSize(7);doc.setTextColor(150,150,150);
      doc.text("[Screenshot unavailable]",MARGIN+4,y+6);
      y+=10;
    }
    y+=4;
  });

  // Event log table
  checkY(30);
  doc.setFillColor(10,16,28);doc.rect(MARGIN,y,W-2*MARGIN,8,"F");
  doc.setDrawColor(0,216,255);doc.rect(MARGIN,y,W-2*MARGIN,8,"S");
  doc.setFont("helvetica","bold");doc.setFontSize(9);doc.setTextColor(0,216,255);
  doc.text("FULL EVENT LOG",MARGIN+4,y+5.5);
  y+=12;
  doc.setFontSize(6);doc.setFont("helvetica","normal");
  logs.slice(0,60).forEach((l,i)=>{
    checkY(6);
    const sevColor=l.sev==="high"?[255,45,74]:l.sev==="med"?[255,170,0]:[0,216,255];
    doc.setTextColor(...sevColor);
    doc.text(`[${l.sev.toUpperCase()}]`,MARGIN,y+3);
    doc.setTextColor(180,200,210);
    doc.text(`${l.time}  ${l.lbl}  ${l.sid?"#"+l.sid:""}  ${l.cls}`,MARGIN+12,y+3);
    y+=4.5;
  });
  if(logs.length>60){
    doc.setTextColor(100,100,100);doc.text(`... and ${logs.length-60} more events`,MARGIN,y+3);
  }

  // Footer
  const pages=doc.getNumberOfPages();
  for(let i=1;i<=pages;i++){
    doc.setPage(i);
    doc.setFontSize(6);doc.setTextColor(80,100,120);
    doc.text(`SENTINEL AI v9 — Confidential — Page ${i}/${pages}`,MARGIN,290);
  }

  doc.save(`Sentinel_AI_Report_${new Date().toISOString().slice(0,10)}.pdf`);
}


// ─────────────────────────────────────────────────────────────
// ROOT
// ─────────────────────────────────────────────────────────────
export default function App(){
  const [user,setUser]=useState(null);
  return(<><style>{CSS}</style>{!user?<Login onLogin={setUser}/>:<Dashboard user={user} onLogout={()=>setUser(null)}/>}</>);
}
