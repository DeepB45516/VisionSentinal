"""
Sentinel AI — Main Detection Pipeline
========================================
Real-time AI cheating detection system for exam surveillance.

Integrates all modules:
  - FaceIntelligence (BlazeFace + FaceMesh)
  - ObjectDetector (COCO-SSD)
  - PoseTracker (MoveNet/MediaPipe Pose)
  - WristVelocityTracker
  - CoordinationDetector
  - RiskScorer
  - TemporalFilter

Runs on webcam or CCTV feed with live visual dashboard overlay.

Usage:
    python main.py                  # webcam
    python main.py --source 0       # webcam index 0
    python main.py --source rtsp://192.168.1.100:554/stream
"""

import sys
import time
import argparse
import cv2
import numpy as np

from modules.face_intelligence import FaceIntelligence
from modules.object_detection import ObjectDetector
from modules.pose_tracking import PoseTracker
from modules.wrist_velocity import WristVelocityTracker
from modules.coordination_detector import CoordinationDetector
from modules.risk_scoring import RiskScorer
from modules.temporal_filter import TemporalFilter


# ─── Constants ────────────────────────────────────────────────
MAX_STUDENTS = 6
TARGET_FPS = 25
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Colors (BGR for OpenCV)
COL_CYAN = (255, 216, 0)
COL_RED = (74, 45, 255)
COL_AMBER = (0, 170, 255)
COL_GREEN = (122, 235, 0)
COL_WHITE = (255, 255, 255)
COL_DARK = (16, 8, 4)
COL_PANEL = (18, 12, 3)


def risk_color(score):
    """Return BGR color for a risk score."""
    if score >= 70:
        return COL_RED
    elif score >= 40:
        return COL_AMBER
    elif score >= 20:
        return (0, 208, 255)
    return COL_CYAN


def draw_corner_box(frame, x, y, w, h, color, thickness=2, corner=10):
    """Draw a corner-bracket bounding box."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
    for cx, cy, dx, dy in [(x, y, 1, 1), (x + w, y, -1, 1),
                            (x, y + h, 1, -1), (x + w, y + h, -1, -1)]:
        cv2.line(frame, (cx, cy), (cx + dx * corner, cy), color, thickness)
        cv2.line(frame, (cx, cy), (cx, cy + dy * corner), color, thickness)


def draw_label(frame, text, x, y, color, bg_color=None):
    """Draw a text label with optional background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    thick = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    if bg_color:
        cv2.rectangle(frame, (x, y - th - 4), (x + tw + 6, y + 2), bg_color, -1)
    cv2.putText(frame, text, (x + 3, y - 2), font, scale, color, thick, cv2.LINE_AA)


def draw_skeleton(frame, keypoints, color=(255, 216, 0)):
    """Draw pose skeleton from keypoint dict."""
    connections = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ]
    for a, b in connections:
        if a in keypoints and b in keypoints:
            pa, pb = keypoints[a], keypoints[b]
            if pa["confidence"] > 0.3 and pb["confidence"] > 0.3:
                cv2.line(frame, (pa["x"], pa["y"]), (pb["x"], pb["y"]),
                         (*color[:3],), 1, cv2.LINE_AA)

    # Draw keypoints as dots
    for name, kp in keypoints.items():
        if kp["confidence"] > 0.3:
            cv2.circle(frame, (kp["x"], kp["y"]), 3, COL_GREEN, -1)


def draw_risk_bar(frame, x, y, h, score):
    """Draw vertical risk bar next to a student bbox."""
    bar_h = int((score / 100) * h)
    color = risk_color(score)
    cv2.rectangle(frame, (x, y + h - bar_h), (x + 5, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + 5, y + h), (40, 40, 40), 1)


def draw_hud(frame, stats, fps, all_scores, alerts):
    """Draw the heads-up-display overlay panel."""
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 30), COL_DARK, -1)
    cv2.putText(frame, "SENTINEL AI v9", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_CYAN, 1, cv2.LINE_AA)

    # FPS
    fps_color = COL_GREEN if fps >= 20 else COL_AMBER if fps >= 10 else COL_RED
    cv2.putText(frame, f"{fps:.0f} FPS", (w - 80, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, fps_color, 1, cv2.LINE_AA)

    # Stats bar
    y_off = 20
    stats_text = f"Faces:{stats.get('faces', 0)}  Objects:{stats.get('objects', 0)}  Students:{stats.get('students', 0)}"
    cv2.putText(frame, stats_text, (180, y_off),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (187, 170, 138), 1, cv2.LINE_AA)

    # REC indicator
    if int(time.time() * 2) % 2:
        cv2.circle(frame, (w - 100, 15), 4, COL_RED, -1)
        cv2.putText(frame, "REC", (w - 93, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COL_RED, 1, cv2.LINE_AA)

    # Bottom panel — student risk scores
    panel_h = 40
    cv2.rectangle(frame, (0, h - panel_h), (w, h), COL_DARK, -1)
    x_pos = 10
    for sid, score in sorted(all_scores.items()):
        color = risk_color(score)
        cv2.putText(frame, f"{sid}:", (x_pos, h - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{score}", (x_pos + 25, h - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        # Mini bar
        bar_w = int((score / 100) * 40)
        cv2.rectangle(frame, (x_pos, h - 16), (x_pos + 40, h - 10), (40, 40, 40), -1)
        cv2.rectangle(frame, (x_pos, h - 16), (x_pos + bar_w, h - 10), color, -1)
        x_pos += 70

    # Right side — recent alerts
    if alerts:
        y_alert = h - panel_h - 10
        for alert in alerts[-4:]:
            sev = alert.get("severity", "low")
            color = COL_RED if sev == "high" else COL_AMBER if sev == "med" else COL_CYAN
            label = alert.get("label", "")[:50]
            cv2.putText(frame, f"! {label}", (10, y_alert),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, color, 1, cv2.LINE_AA)
            y_alert -= 16

    # Scan line effect
    scan_y = int((time.time() * 80) % h)
    cv2.line(frame, (0, scan_y), (w, scan_y), (255, 216, 0, 20), 1)


def main():
    parser = argparse.ArgumentParser(description="Sentinel AI — Real-time Exam Surveillance")
    parser.add_argument("--source", default="0",
                        help="Video source: 0 for webcam, or RTSP/file path")
    parser.add_argument("--width", type=int, default=FRAME_WIDTH)
    parser.add_argument("--height", type=int, default=FRAME_HEIGHT)
    parser.add_argument("--no-display", action="store_true",
                        help="Run headless (no window)")
    args = parser.parse_args()

    # Parse source
    source = int(args.source) if args.source.isdigit() else args.source

    print("=" * 60)
    print("  SENTINEL AI v9 — Real-time Exam Surveillance")
    print("=" * 60)
    print(f"  Source: {source}")
    print(f"  Max students: {MAX_STUDENTS}")
    print()

    # Initialize modules
    print("[1/7] Loading Face Intelligence (BlazeFace + FaceLandmarker)...")
    face_module = FaceIntelligence(max_students=MAX_STUDENTS)
    print("  ✓ Face module ready")

    print("[2/7] Loading Object Detector (EfficientDet-Lite0)...")
    obj_module = ObjectDetector()
    print("  ✓ Object detector ready")

    print("[3/7] Loading Pose Tracker (PoseLandmarker)...")
    pose_module = PoseTracker(max_students=MAX_STUDENTS)
    print("  ✓ Pose module ready")

    print("[4/7] Initializing Wrist Velocity Tracker...")
    wrist_module = WristVelocityTracker(max_students=MAX_STUDENTS)
    print("  ✓ Wrist velocity ready")

    print("[5/7] Initializing Coordination Detector...")
    coord_module = CoordinationDetector()
    print("  ✓ Coordination detector ready")

    print("[6/7] Initializing Risk Scorer...")
    risk_module = RiskScorer()
    print("  ✓ Risk scoring ready")

    print("[7/7] Initializing Temporal Filter...")
    temp_filter = TemporalFilter(window_size=4, min_confirmations=2)
    print("  ✓ Temporal filter ready")

    print()
    print("All modules loaded. Starting video capture...")
    print("Press 'q' to quit, 'r' for risk report, 's' for screenshot")
    print()

    # Open video (use DirectShow on Windows for better compatibility)
    if isinstance(source, int) and sys.platform == "win32":
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video source: {source}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # FPS tracking
    fps = 0.0
    frame_count = 0
    fps_timer = time.time()
    recent_alerts = []
    obj_result = {"detections": [], "persons": []}
    pose_result = {"poses": [], "behaviors": []}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream.")
                break

            h, w = frame.shape[:2]
            frame_rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # ═══════════════════════════════════════════════
            # STEP 1 — Face Intelligence
            # ═══════════════════════════════════════════════
            face_result = face_module.process(frame_rgb, h, w)

            # Feed face events to risk scorer (with temporal filtering)
            for gaze in face_result["gaze_directions"]:
                sid = gaze["id"]
                direction = gaze["direction"]
                if direction in ("LEFT", "RIGHT"):
                    event_key = f"gaze_{direction.lower()}"
                    detected = temp_filter.update(sid, event_key, True)
                    if detected:
                        risk_module.add_event(sid, event_key)
                elif direction == "DOWN":
                    if temp_filter.update(sid, "gaze_down", True):
                        risk_module.add_event(sid, "gaze_down")
                else:
                    temp_filter.update(sid, "gaze_left", False)
                    temp_filter.update(sid, "gaze_right", False)
                    temp_filter.update(sid, "gaze_down", False)

            # Proxy detection (multiple faces)
            if face_result["face_count"] > 1:
                if temp_filter.update("global", "proxy", True):
                    risk_module.add_event("global", "proxy")
                    recent_alerts.append({
                        "label": f"PROXY: {face_result['face_count']} faces detected!",
                        "severity": "high"
                    })
            else:
                temp_filter.update("global", "proxy", False)

            # Absence alerts
            for absence in face_result["absences"]:
                sid = absence["id"]
                if temp_filter.update(sid, "face_absent", True):
                    risk_module.add_event(sid, "face_absent")
                    recent_alerts.append({
                        "label": f"{sid}: Absent {absence['absent_seconds']}s",
                        "severity": "high"
                    })

            # ═══════════════════════════════════════════════
            # STEP 2 — Object Detection (every 3rd frame for speed)
            # ═══════════════════════════════════════════════
            student_positions = [
                {"id": f["student_id"], "center": f["center"]}
                for f in face_result["faces"] if "student_id" in f
            ]
            if frame_count % 3 == 0:
                obj_result = obj_module.process(frame_rgb, student_positions)

            for det in obj_result["detections"]:
                sid = det["student_id"]
                label = det["class"]
                if sid and temp_filter.update(sid, label, True):
                    risk_module.add_event(sid, label)
                    recent_alerts.append({
                        "label": f"{sid}: {label.upper()} detected ({det['confidence']:.0%})",
                        "severity": "high",
                    })

            # ═══════════════════════════════════════════════
            # STEP 3 — Pose Tracking (every 2nd frame for speed)
            # ═══════════════════════════════════════════════
            # Collect person bboxes from face detections for multi-person crop
            person_bboxes = [f["bbox"] for f in face_result["faces"]]
            if frame_count % 2 == 0:
                pose_result = pose_module.process(frame_rgb, h, w, person_bboxes or None)

            # Feed pose behavior alerts
            for behavior in pose_result["behaviors"]:
                sid = behavior["student_id"]
                btype = behavior["type"]
                if temp_filter.update(sid, btype, True):
                    risk_module.add_event(sid, btype)
                    recent_alerts.append({
                        "label": f"{sid}: {behavior['label']}",
                        "severity": behavior["severity"]
                    })

            # ═══════════════════════════════════════════════
            # STEP 4 — Wrist Velocity
            # ═══════════════════════════════════════════════
            wrist_positions = pose_module.get_wrist_positions(pose_result["poses"])
            wrist_result = wrist_module.process(wrist_positions)

            for alert in wrist_result["alerts"]:
                sid = alert["student_id"]
                if temp_filter.update(sid, "wrist_velocity", True):
                    risk_module.add_event(sid, "wrist_velocity")
                    recent_alerts.append({
                        "label": alert["label"],
                        "severity": alert["severity"]
                    })

            # ═══════════════════════════════════════════════
            # STEP 5 — Coordination Detection
            # ═══════════════════════════════════════════════
            coord_result = coord_module.process(
                student_positions,
                face_result["gaze_directions"],
                wrist_result
            )
            for alert in coord_result["alerts"]:
                for sid in alert.get("student_ids", []):
                    risk_module.add_event(sid, alert["type"])
                recent_alerts.append({
                    "label": alert["label"],
                    "severity": alert["severity"]
                })

            # ═══════════════════════════════════════════════
            # STEP 6 — Get Risk Scores
            # ═══════════════════════════════════════════════
            all_scores = risk_module.get_all_scores()

            # Trim recent alerts to last 20
            if len(recent_alerts) > 20:
                recent_alerts = recent_alerts[-20:]

            # ═══════════════════════════════════════════════
            # STEP 7 — Draw Visual Dashboard
            # ═══════════════════════════════════════════════
            if not args.no_display:
                # Draw face boxes
                for face in face_result["faces"]:
                    x, y2, bw, bh = face["bbox"]
                    sid = face.get("student_id", "?")
                    score = all_scores.get(sid, 0)
                    color = risk_color(score)
                    draw_corner_box(frame, x, y2, bw, bh, color)
                    draw_label(frame, f"{sid} R:{score}", x, y2, color, COL_DARK)
                    draw_risk_bar(frame, x + bw + 3, y2, bh, score)

                # Draw gaze arrows
                for gaze in face_result["gaze_directions"]:
                    sid = gaze["id"]
                    direction = gaze["direction"]
                    face = next((f for f in face_result["faces"]
                                 if f.get("student_id") == sid), None)
                    if face and direction != "CENTER":
                        cx, cy = face["center"]
                        dx = 40 if direction == "RIGHT" else -40 if direction == "LEFT" else 0
                        dy = 30 if direction == "DOWN" else 0
                        arrow_color = COL_RED if direction in ("LEFT", "RIGHT") else COL_AMBER
                        cv2.arrowedLine(frame, (cx, cy), (cx + dx, cy + dy),
                                        arrow_color, 2, tipLength=0.3)

                # Draw skeletons
                for pose in pose_result["poses"]:
                    draw_skeleton(frame, pose["keypoints"], COL_CYAN)

                # Draw wrist velocity arrows
                for sid, vel_info in wrist_result.get("velocities", {}).items():
                    if vel_info.get("suspicious"):
                        # Find wrist position
                        wrist_pos = wrist_positions.get(sid)
                        if wrist_pos:
                            lx, ly = wrist_pos["left"]
                            cv2.circle(frame, (lx, ly), 10, COL_AMBER, 2)
                            draw_label(frame, f"{vel_info['smoothed_vel']:.0f}px/s",
                                       lx + 12, ly, COL_AMBER)

                # Draw object detections
                for det in obj_result["detections"]:
                    ox, oy, ow, oh = det["bbox"]
                    draw_corner_box(frame, ox, oy, ow, oh, COL_RED, thickness=2, corner=8)
                    draw_label(frame, f'{det["class"]} {det["confidence"]:.0%}',
                               ox, oy, COL_RED, COL_DARK)

                # Draw HUD
                stats = {
                    "faces": face_result["face_count"],
                    "objects": len(obj_result["detections"]),
                    "students": len(pose_result["poses"])
                }
                draw_hud(frame, stats, fps, all_scores, recent_alerts)

                cv2.imshow("Sentinel AI v9", frame)

            # FPS calculation
            frame_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_timer = time.time()

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                # Print risk report
                print("\n" + "=" * 50)
                print("  RISK REPORT")
                print("=" * 50)
                for entry in risk_module.get_report():
                    sev = entry["severity"].upper()
                    print(f"  {entry['student_id']}: Score={entry['risk_score']} "
                          f"[{sev}] Events={entry['event_count']}")
                    for ev in entry["recent_events"]:
                        print(f"    - {ev['type']} (+{ev['delta']})")
                print("=" * 50 + "\n")
            elif key == ord("s"):
                # Save screenshot
                fname = f"sentinel_capture_{int(time.time())}.jpg"
                cv2.imwrite(fname, frame)
                print(f"Screenshot saved: {fname}")

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        # Final report
        print("\n" + "=" * 60)
        print("  SESSION SUMMARY")
        print("=" * 60)
        for entry in risk_module.get_report():
            sev = entry["severity"].upper()
            print(f"  {entry['student_id']}: Score={entry['risk_score']} [{sev}]")
        print(f"  Total alerts: {len(recent_alerts)}")
        print("=" * 60)

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        face_module.cleanup()
        obj_module.cleanup()
        pose_module.cleanup()
        print("Sentinel AI stopped.")


if __name__ == "__main__":
    main()
