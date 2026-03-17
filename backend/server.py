import sys
import os
import math

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

import cv2
from modules.face_intelligence      import FaceIntelligence
from modules.emotion_detector       import EmotionDetector       # v3 new
from modules.object_detection       import ObjectDetector
from modules.gemini_inspector       import GeminiInspector       # v4 new
from modules.pose_tracking          import PoseTracker
from modules.wrist_velocity         import WristVelocityTracker
from modules.coordination_detector  import CoordinationDetector
from modules.risk_scoring           import RiskScorer
from modules.temporal_filter        import TemporalFilter
from flask import Flask, Response, request, jsonify
from flask_cors import CORS

from auth import create_user, verify_user
from auth import create_default_user

app = Flask(__name__)
CORS(app)

# ── Camera state (unchanged) ──────────────────────────────────────────────
camera = None
MAX_STUDENTS = 6

# ── Module instances ──────────────────────────────────────────────────────
face_module   = FaceIntelligence(max_students=MAX_STUDENTS)
emotion_module = EmotionDetector()                              # v3 new
obj_module    = ObjectDetector()
gemini_module = GeminiInspector(api_key="AIzaSyCCLRpSZACuGQubLZ4_RCy-GIX7yFnDgy0")                             # v4 new (needs GEMINI_API_KEY)
pose_module   = PoseTracker(max_students=MAX_STUDENTS)
wrist_module  = WristVelocityTracker(max_students=MAX_STUDENTS)
coord_module  = CoordinationDetector()
risk_module   = RiskScorer()
temp_filter   = TemporalFilter(window_size=4, min_confirmations=2)

# ── Shared state — written by generate_frames(), read by API routes ───────
_latest_state: dict = {
    "risk_scores": {},
    "alerts":      [],
    "proxy_alert": False,
    "emotions":    {},
    "frame_count": 0,
}

# ── Drawing helpers ───────────────────────────────────────────────────────
_SEVERITY_COLORS = {
    "CRITICAL": (0,   0,   255),
    "HIGH":     (0,   100, 255),
    "MEDIUM":   (0,   180, 255),
    "LOW":      (0,   220, 180),
    "critical": (0,   0,   255),
    "high":     (0,   100, 255),
    "med":      (0,   180, 255),
    "low":      (0,   220, 180),
    "safe":     (0,   200, 0  ),
}

_RISK_SCORE_COLOR = {
    "critical":   (0, 0, 255),
    "high":       (0, 80, 255),
    "suspicious": (0, 165, 255),
    "watch":      (0, 220, 200),
    "safe":       (0, 200, 0),
}


def _draw_risk_badge(frame, x: int, y: int, sid: str,
                     score: int, severity: str) -> None:
    """Draw a compact risk score badge above a face box."""
    label = f"{sid}  Risk:{score}"
    color = _RISK_SCORE_COLOR.get(severity, (200, 200, 200))
    # Background pill
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y - 2), color, -1)
    cv2.putText(frame, label, (x + 3, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)


def _draw_alert_ticker(frame, alerts: list, h: int) -> None:
    """Draw latest global alerts at the bottom of the frame."""
    y = h - 10
    for alert in reversed(alerts[-4:]):    # show last 4
        label = alert.get("label", alert.get("type", ""))
        color = _SEVERITY_COLORS.get(alert.get("severity", "low"), (200, 200, 200))
        cv2.putText(frame, label, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1)
        y -= 20


# ── Main video pipeline ───────────────────────────────────────────────────
def generate_frames():

    global camera, _latest_state

    frame_count = 0

    # Initialise cached module results so later steps never hit NameError
    # on frames where the if-gate skips inference.
    obj_result  = {"detections": [], "persons": []}
    pose_result = {"poses": [],      "behaviors": []}

    while True:

        success, frame = camera.read()
        if not success:
            break

        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── STEP 1 — Face Intelligence ────────────────────────────────
        face_result   = face_module.process(frame_rgb, h, w)
        faces         = face_result["faces"]
        gazes         = face_result["gaze_directions"]
        absences      = face_result["absences"]
        raw_emotions  = face_result.get("emotions", [])
        proxy_alert   = face_result.get("proxy_alert", False)
        shifty_sids   = set(face_result.get("shifty_students", []))

        gaze_map = {g["id"]: g["direction"] for g in gazes}

        # Draw face boxes + gaze label
        for face in faces:
            x, y, bw, bh = face["bbox"]
            sid = face.get("student_id", face["id"])
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            gaze_dir = gaze_map.get(sid, "")
            if gaze_dir and gaze_dir != "CENTER":
                cv2.putText(frame, f"Gaze:{gaze_dir}", (x, y + bh + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 180), 1)

        # Proxy alert banner
        if proxy_alert:
            cv2.putText(frame, "!! PROXY ALERT !!",
                        (w // 2 - 100, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # ── STEP 2 — Emotion Analysis ─────────────────────────────────
        emotion_events = emotion_module.process(raw_emotions)
        emo_label_map  = {e["id"]: e["label"] for e in raw_emotions}

        # Draw emotion label next to face
        for face in faces:
            x, y, bw, bh = face["bbox"]
            sid = face.get("student_id", face["id"])
            emo = emo_label_map.get(sid, "")
            if emo and emo != "neutral":
                cv2.putText(frame, emo, (x + bw + 4, y + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 180, 0), 1)

        # ── STEP 3 — Object Detection (every 3rd frame) ───────────────
        student_positions = [
            {"id": f.get("student_id", f["id"]), "center": f["center"]}
            for f in faces
        ]

        if frame_count % 3 == 0:
            obj_result = obj_module.process(frame_rgb, student_positions)

            # Fire async Gemini scan if enabled (non-blocking)
            if gemini_module.enabled:
                face_bbox_map = {
                    f.get("student_id", f["id"]): f["bbox"] for f in faces
                }
                for sid in (f.get("student_id", f["id"]) for f in faces):
                    gemini_module.scan_student_async(
                        sid, frame_rgb, face_bbox_map.get(sid)
                    )
                # Merge cached Gemini results from previous scan cycle
                existing_keys = {
                    (d["class"], d.get("student_id"))
                    for d in obj_result["detections"]
                }
                for sid, gemini_dets in gemini_module.get_all_cached().items():
                    for gdet in gemini_dets:
                        if (gdet["class"], sid) not in existing_keys:
                            obj_result["detections"].append(gdet)

        # Draw object detections (guard for Gemini bbox=None)
        for det in obj_result["detections"]:
            if det.get("bbox") is not None:
                ox, oy, ow, oh = det["bbox"]
                sev_color = _SEVERITY_COLORS.get(det.get("severity", "LOW"),
                                                  (0, 0, 255))
                cv2.rectangle(frame, (ox, oy), (ox + ow, oy + oh), sev_color, 2)
                cv2.putText(frame, det["class"], (ox, oy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, sev_color, 1)
            else:
                # Gemini detection: show text near assigned student face
                sid = det.get("student_id")
                face_match = next(
                    (f for f in faces if f.get("student_id") == sid), None
                )
                if face_match:
                    fx, fy, fw, fh = face_match["bbox"]
                    label = f"[G] {det['class']}"
                    cv2.putText(frame, label, (fx, fy - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        # ── STEP 4 — Pose Tracking (every 2nd frame) ──────────────────
        person_bboxes = [f["bbox"] for f in faces]

        if frame_count % 2 == 0:
            pose_result = pose_module.process(frame_rgb, h, w, person_bboxes)

            for pose in pose_result["poses"]:
                for kp in pose["keypoints"].values():
                    if kp["confidence"] > 0.3:
                        cv2.circle(frame, (kp["x"], kp["y"]),
                                   3, (255, 255, 0), -1)

            # Draw pose behavior labels above face
            for beh in pose_result["behaviors"]:
                sid = beh["student_id"]
                face_match = next(
                    (f for f in faces
                     if f.get("student_id") == sid), None
                )
                if face_match:
                    bx, by, _, _ = face_match["bbox"]
                    color = _SEVERITY_COLORS.get(beh["severity"], (200, 200, 0))
                    cv2.putText(frame, beh["label"],
                                (bx, by - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        # ── STEP 5 — Wrist Velocity ────────────────────────────────────
        wrist_positions = pose_module.get_wrist_positions(pose_result["poses"])
        wrist_result    = wrist_module.process(wrist_positions)

        for alert in wrist_result["alerts"]:
            cv2.putText(frame, alert["label"],
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ── STEP 6 — Coordination Detection ───────────────────────────
        coord_module.set_frame_diagonal(math.hypot(w, h))
        coord_result = coord_module.process(
            student_positions=student_positions,
            gaze_data=gazes,
            velocity_data=wrist_result,
            pose_behaviors=pose_result["behaviors"],
            emotions=raw_emotions,
        )

        # ── STEP 7 — Risk Scoring ──────────────────────────────────────
        # Gaze events
        for gaze in gazes:
            sid = gaze["id"]
            direction = gaze["direction"]
            if direction in ("LEFT", "RIGHT", "DOWN", "UP"):
                event_key = f"gaze_{direction.lower()}"
                if temp_filter.update(sid, event_key, True, gaze["confidence"]):
                    risk_module.add_event(sid, event_key)

        # Object detection events
        for det in obj_result["detections"]:
            sid = det.get("student_id")
            if sid:
                if temp_filter.update(sid, det["class"],
                                      det.get("confirmed", True),
                                      det["confidence"]):
                    risk_module.add_event(sid, det["class"])

        # Pose behavior events
        for beh in pose_result["behaviors"]:
            sid = beh["student_id"]
            if temp_filter.update(sid, beh["type"], True,
                                  beh.get("confidence", 0.7)):
                risk_module.add_event(sid, beh["type"])

        # Wrist velocity events
        for alert in wrist_result["alerts"]:
            sid = alert["student_id"]
            if temp_filter.update(sid, alert["type"], True,
                                  alert.get("confidence", 0.7)):
                risk_module.add_event(sid, alert["type"])

        # Emotion events
        for evt in emotion_events:
            risk_module.add_event(evt["student_id"], evt["event_type"])

        # Coordination events
        for alert in coord_result["alerts"]:
            for sid in alert.get("student_ids", []):
                risk_module.add_event(sid, alert["type"])

        # Absence / proxy events
        for ab in absences:
            risk_module.add_event(ab["id"], "face_absent")
        if proxy_alert:
            for sid in (f.get("student_id", f["id"]) for f in faces):
                risk_module.add_event(sid, "proxy")

        # Shifty eyes
        for sid in shifty_sids:
            risk_module.add_event(sid, "repeated_head_turn")

        # ── STEP 8 — Draw risk badges + alert ticker ───────────────────
        all_scores = risk_module.get_all_scores()

        for face in faces:
            x, y, bw, bh = face["bbox"]
            sid      = face.get("student_id", face["id"])
            score    = all_scores.get(sid, 0)
            severity = risk_module.get_severity(sid)
            _draw_risk_badge(frame, x, y, sid, score, severity)

        # Collect all alerts for the ticker and API state
        all_alerts = (
            coord_result["alerts"]
            + [{"label": a["label"], "severity": a.get("severity", "med"),
                "type": a["type"]}
               for a in wrist_result["alerts"]]
            + [{"label": b["label"], "severity": b.get("severity", "med"),
                "type": b["type"]}
               for b in pose_result["behaviors"]]
            + [{"label": f"{ab['id']} absent {ab['absent_seconds']:.0f}s",
                "severity": "high", "type": "face_absent"}
               for ab in absences]
        )
        if proxy_alert:
            all_alerts.append(
                {"label": "PROXY ALERT", "severity": "critical",
                 "type": "proxy"}
            )

        _draw_alert_ticker(frame, all_alerts, h)

        # ── Update shared state for API routes ─────────────────────────
        _latest_state["risk_scores"] = all_scores
        _latest_state["alerts"]      = all_alerts[-20:]
        _latest_state["proxy_alert"] = proxy_alert
        _latest_state["emotions"]    = emo_label_map
        _latest_state["frame_count"] = frame_count

        # ── Encode and yield ───────────────────────────────────────────
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        frame_count += 1


# ── Routes (all original routes preserved exactly) ────────────────────────

@app.route("/video")
def video():

    if camera is None:
        return "Camera not started"

    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/start_camera", methods=["POST"])
def start_camera():

    global camera

    data   = request.json
    source = data["source"]

    if camera:
        camera.release()

    if source.isdigit():
        camera = cv2.VideoCapture(int(source))
    else:
        camera = cv2.VideoCapture(source, cv2.CAP_FFMPEG)

    return {"status": "camera_started"}


@app.route("/register", methods=["POST"])
def register():

    data = request.json

    if create_user(data["username"], data["password"]):
        return {"status": "success"}
    else:
        return {"status": "user_exists"}


@app.route("/login", methods=["POST"])
def login():

    data = request.json

    if verify_user(data["username"], data["password"]):
        return {"status": "success"}
    else:
        return {"status": "invalid"}


# ── New routes ─────────────────────────────────────────────────────────────

@app.route("/risk_report", methods=["GET"])
def risk_report():
    """Return current per-student risk scores and full session report."""
    return jsonify({
        "scores":      _latest_state["risk_scores"],
        "proxy_alert": _latest_state["proxy_alert"],
        "emotions":    _latest_state["emotions"],
        "alerts":      _latest_state["alerts"],
        "report":      risk_module.get_report(),
        "frame":       _latest_state["frame_count"],
    })


@app.route("/stop_camera", methods=["POST"])
def stop_camera():
    """Release the camera cleanly."""
    global camera
    if camera:
        camera.release()
        camera = None
    return {"status": "camera_stopped"}


@app.route("/reset_scores", methods=["POST"])
def reset_scores():
    """Reset all risk scores (e.g. start of a new exam session)."""
    data = request.json or {}
    sid  = data.get("student_id")
    if sid:
        risk_module.reset_student(sid)
        return {"status": "reset", "student_id": sid}
    # Reset all students
    for s in list(risk_module.scores.keys()):
        risk_module.reset_student(s)
    temp_filter.reset_all()
    return {"status": "all_reset"}

@app.route("/")
def home():
    return "VisionSeninal Backend Running 🚀"
# ── Entry point (unchanged) ────────────────────────────────────────────────

if __name__ == "__main__":
    create_default_user()
    app.run(host="0.0.0.0", port=5000)
