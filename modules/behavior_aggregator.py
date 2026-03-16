"""
Sentinel AI — Behavior Aggregator  (NEW in v3)
===============================================
Central coordinator that drives all detection modules in a single
.process(frame) call, manages adaptive frame-budget, deduplicates
alerts, and returns a clean per-frame result to the UI/CCTV layer.

Architecture
------------
                ┌─────────────────────────────────────┐
    frame_rgb → │       BehaviorAggregator             │
                │                                     │
                │  FaceIntelligence  (gaze, emotion,   │
                │                    absence, proxy)   │
                │  ObjectDetector    (phone, book,…)   │
                │  PoseTracker       (pose behaviors)  │
                │  WristVelocity     (hand speed)      │
                │  CoordDetector     (multi-student)   │
                │  EmotionDetector   (blendshapes)     │
                │  TemporalFilter    (smoothing)       │
                │  RiskScorer        (0–100 per student│
                └──────────────┬──────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  AggregatorResult   │
                    │  - students[]       │
                    │  - global_alerts[]  │
                    │  - proxy_alert bool │
                    │  - frame_ms float   │
                    └─────────────────────┘

Adaptive frame budget
---------------------
If the total processing time for a frame exceeds _BUDGET_HIGH_MS, the
aggregator increases the frame_skip on slow modules (object/pose) by 1.
If processing drops below _BUDGET_LOW_MS, it decreases frame_skip back
toward 0. Face module skip is never raised above MAX_FACE_SKIP.

Usage
-----
    from modules.behavior_aggregator import BehaviorAggregator
    agg = BehaviorAggregator()

    # Each video frame:
    result = agg.process(frame_rgb)

    for s in result["students"]:
        print(s["student_id"], s["risk_score"], s["severity"])
        for alert in s["alerts"]:
            print("  →", alert["label"])
"""

import math
import time
import numpy as np

from .face_intelligence     import FaceIntelligence
from .emotion_detector      import EmotionDetector
from .object_detection      import ObjectDetector
from .gemini_inspector      import GeminiInspector
from .pose_tracking         import PoseTracker
from .wrist_velocity        import WristVelocityTracker
from .coordination_detector import CoordinationDetector
from .risk_scoring          import RiskScorer
from .temporal_filter       import TemporalFilter


# ---------------------------------------------------------------------------
# Adaptive frame-budget constants
# ---------------------------------------------------------------------------
_BUDGET_HIGH_MS  = 55.0   # if frame takes longer → increase slow-module skip
_BUDGET_LOW_MS   = 25.0   # if frame is fast → decrease slow-module skip
_MAX_FACE_SKIP   = 2      # face module never skips more than this
_MAX_SLOW_SKIP   = 4      # object/pose skip ceiling

# Event type → risk event key mapping (for events coming from module output)
_POSE_TYPE_TO_RISK = {
    "head_down":           "head_down",
    "arm_raise":           "arm_raise",
    "lean_over":           "lean_over",
    "hiding_object":       "hiding_object",
    "look_behind":         "look_behind",
    "repeated_head_turn":  "repeated_head_turn",
    "shielding_paper":     "shielding_paper",
}

_COORD_TYPE_TO_RISK = {
    "coordinated_movement": "coordinated_movement",
    "mutual_gaze":          "mutual_gaze",
    "gaze_movement_combo":  "gaze_movement_combo",
    "sync_head_down":       "sync_head_down",
    "pass_back":            "pass_back",
    "emotion_correlation":  "emotion_correlation",
}


class BehaviorAggregator:
    """
    Single-call driver for all Sentinel AI detection modules.
    """

    def __init__(
        self,
        max_students:      int   = 6,
        absence_threshold: float = 5.0,
        face_skip:         int   = 1,
        obj_skip:          int   = 2,
        pose_skip:         int   = 2,
    ):
        # Instantiate all modules
        self.face  = FaceIntelligence(
            max_students=max_students,
            absence_threshold=absence_threshold,
            frame_skip=face_skip,
            detect_scale=0.5,
        )
        self.emo    = EmotionDetector()
        self.obj    = ObjectDetector(frame_skip=obj_skip, detect_scale=0.70)
        self.gemini = GeminiInspector()  # uses GEMINI_API_KEY env var
        self.pose  = PoseTracker(max_students=max_students,
                                 frame_skip=pose_skip, detect_scale=0.50)
        self.wrist = WristVelocityTracker()
        self.coord = CoordinationDetector()
        self.risk  = RiskScorer()
        self.tf    = TemporalFilter(window_size=4, min_confirmations=2)

        # Adaptive skip state
        self._obj_skip  = obj_skip
        self._pose_skip = pose_skip
        self._frame_ms_history = []   # last 10 frame times

        self._frame_count  = 0
        self._frame_h: int = 0
        self._frame_w: int = 0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(self, frame_rgb: np.ndarray) -> dict:
        """
        Run all detection modules on one RGB frame.

        Returns
        -------
        dict:
            students      list[dict]  — per-student summary (see below)
            global_alerts list[dict]  — coordination / proxy alerts
            proxy_alert   bool
            frame_ms      float       — total processing time for this frame
            frame_number  int

        Per-student dict keys:
            student_id, risk_score, severity,
            gaze, emotion, alerts (list of individual alert dicts),
            absent, absent_seconds
        """
        t0 = time.perf_counter()

        h, w       = frame_rgb.shape[:2]
        self._frame_h = h
        self._frame_w = w
        self._frame_count += 1

        # ── Face ─────────────────────────────────────────────────────
        face_result = self.face.process(frame_rgb, h, w)
        faces       = face_result["faces"]
        gazes       = face_result["gaze_directions"]
        absences    = face_result["absences"]
        raw_emotions = face_result.get("emotions", [])
        proxy_alert  = face_result.get("proxy_alert", False)
        shifty       = set(face_result.get("shifty_students", []))

        # Build student_positions for downstream modules
        student_positions = [
            {"id": f.get("student_id", f["id"]), "center": f["center"]}
            for f in faces
        ]

        # Gaze map for risk events
        gaze_map = {g["id"]: g["direction"] for g in gazes}

        # ── Emotion ──────────────────────────────────────────────────
        emotion_events = self.emo.process(raw_emotions)
        emo_map = {e["id"]: e for e in raw_emotions}

        # ── Object detection (YOLOv8) ────────────────────────────────
        obj_result  = self.obj.process(frame_rgb, student_positions)
        detections  = obj_result["detections"]

        # ── Gemini deep scan (async, non-blocking) ────────────────────
        if self.gemini.enabled:
            face_bbox_map = {f.get("student_id", f["id"]): f["bbox"] for f in faces}
            for sid in (f.get("student_id", f["id"]) for f in faces):
                self.gemini.scan_student_async(sid, frame_rgb, face_bbox_map.get(sid))
            for sid, gemini_dets in self.gemini.get_all_cached().items():
                for gdet in gemini_dets:
                    already = any(d["class"] == gdet["class"] and d.get("student_id") == sid for d in detections)
                    if not already:
                        detections.append(gdet)

        # ── Pose ─────────────────────────────────────────────────────
        pose_result    = self.pose.process(frame_rgb, h, w)
        pose_behaviors = pose_result["behaviors"]
        wrist_positions = self.pose.get_wrist_positions(pose_result["poses"])

        # ── Wrist velocity ───────────────────────────────────────────
        vel_result = self.wrist.process(wrist_positions)

        # ── Coordination ─────────────────────────────────────────────
        self.coord.set_frame_diagonal(math.hypot(w, h))
        coord_result = self.coord.process(
            student_positions=student_positions,
            gaze_data=gazes,
            velocity_data=vel_result,
            pose_behaviors=pose_behaviors,
            emotions=raw_emotions,
        )

        # ── Assemble per-student data ─────────────────────────────────
        all_sids = set(f.get("student_id", f["id"]) for f in faces)
        # Include absent students
        for ab in absences:
            all_sids.add(ab["id"])

        student_summaries: dict = {}

        for sid in all_sids:
            student_summaries[sid] = {
                "student_id":     sid,
                "risk_score":     0,
                "severity":       "safe",
                "gaze":           gaze_map.get(sid, "CENTER"),
                "emotion":        emo_map.get(sid, {}).get("label", "neutral"),
                "alerts":         [],
                "absent":         False,
                "absent_seconds": 0.0,
            }

        # ── Risk events: Gaze ─────────────────────────────────────────
        for gaze in gazes:
            sid = gaze["id"]
            if gaze["direction"] in ("LEFT", "RIGHT", "DOWN", "UP"):
                event_key = f"gaze_{gaze['direction'].lower()}"
                confirmed = self.tf.update(sid, event_key, True, gaze["confidence"])
                if confirmed:
                    self.risk.add_event(sid, event_key)
                    if sid in student_summaries:
                        student_summaries[sid]["alerts"].append({
                            "type": event_key,
                            "label": f"Gaze {gaze['direction']}",
                            "severity": "med",
                            "confidence": gaze["confidence"],
                        })

        # Shifty eyes
        for sid in shifty:
            self.risk.add_event(sid, "repeated_head_turn")  # reuse weight
            if sid in student_summaries:
                student_summaries[sid]["alerts"].append({
                    "type": "shifty_eyes",
                    "label": "Shifty Eyes (rapid gaze switches)",
                    "severity": "med", "confidence": 0.75,
                })

        # ── Risk events: Objects ──────────────────────────────────────
        for det in detections:
            sid   = det.get("student_id")
            label = det["class"]
            confirmed = self.tf.update(sid or "unknown", label,
                                       det.get("confirmed", False),
                                       det["confidence"])
            if confirmed or det.get("confirmed", False):
                self.risk.add_event(sid or "unknown", label)
                target_sid = sid or (all_sids and next(iter(all_sids)))
                if target_sid and target_sid in student_summaries:
                    student_summaries[target_sid]["alerts"].append({
                        "type":       label,
                        "label":      f"{det['severity']}: {label.title()} Detected",
                        "severity":   det["severity"].lower(),
                        "confidence": det["confidence"],
                    })

        # ── Risk events: Pose behaviors ──────────────────────────────
        for beh in pose_behaviors:
            sid   = beh["student_id"]
            btype = beh["type"]
            risk_key = _POSE_TYPE_TO_RISK.get(btype, btype)
            confirmed = self.tf.update(sid, btype, True, beh.get("confidence", 0.7))
            if confirmed:
                self.risk.add_event(sid, risk_key)
                if sid in student_summaries:
                    student_summaries[sid]["alerts"].append({
                        "type":       btype,
                        "label":      beh["label"],
                        "severity":   beh["severity"],
                        "confidence": beh.get("confidence", 0.7),
                    })

        # ── Risk events: Wrist velocity ───────────────────────────────
        for alert in vel_result["alerts"]:
            sid = alert["student_id"]
            confirmed = self.tf.update(sid, alert["type"], True,
                                       alert.get("confidence", 0.7))
            if confirmed:
                self.risk.add_event(sid, alert["type"])
                if sid in student_summaries:
                    student_summaries[sid]["alerts"].append(alert)

        # ── Risk events: Emotion ─────────────────────────────────────
        for evt in emotion_events:
            sid = evt["student_id"]
            self.risk.add_event(sid, evt["event_type"])
            if sid in student_summaries:
                student_summaries[sid]["alerts"].append({
                    "type":       evt["event_type"],
                    "label":      evt["label"],
                    "severity":   evt["severity"],
                    "confidence": evt["confidence"],
                })

        # ── Risk events: Absence ─────────────────────────────────────
        global_alerts: list = []
        absent_ids: set = set()
        for ab in absences:
            sid = ab["id"]
            absent_ids.add(sid)
            self.risk.add_event(sid, "face_absent")
            if sid not in student_summaries:
                student_summaries[sid] = {
                    "student_id": sid, "risk_score": 0, "severity": "safe",
                    "gaze": "N/A", "emotion": "N/A", "alerts": [],
                    "absent": True, "absent_seconds": ab["absent_seconds"],
                }
            student_summaries[sid]["absent"]         = True
            student_summaries[sid]["absent_seconds"] = ab["absent_seconds"]
            global_alerts.append({
                "type":  "face_absent",
                "label": f"{sid} Absent {ab['absent_seconds']:.0f}s",
                "severity": "high", "student_id": sid,
            })

        # ── Risk events: Proxy ────────────────────────────────────────
        if proxy_alert:
            for sid in all_sids:
                self.risk.add_event(sid, "proxy")
            global_alerts.append({
                "type": "proxy", "label": "Proxy / Impersonation Alert",
                "severity": "critical", "student_id": None,
            })

        # ── Risk events: Coordination ─────────────────────────────────
        for alert in coord_result["alerts"]:
            for sid in alert.get("student_ids", []):
                risk_key = _COORD_TYPE_TO_RISK.get(alert["type"], "coordinated_movement")
                self.risk.add_event(sid, risk_key)
            global_alerts.append(alert)

        # ── Finalise per-student risk ─────────────────────────────────
        all_scores = self.risk.get_all_scores()
        for sid, summary in student_summaries.items():
            summary["risk_score"] = all_scores.get(sid, 0)
            summary["severity"]   = self.risk.get_severity(sid)

        # ── Adaptive frame budget ─────────────────────────────────────
        frame_ms = (time.perf_counter() - t0) * 1000
        self._adapt_skips(frame_ms)

        return {
            "students":      list(student_summaries.values()),
            "global_alerts": global_alerts,
            "proxy_alert":   proxy_alert,
            "frame_ms":      round(frame_ms, 2),
            "frame_number":  self._frame_count,
        }

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def get_risk_report(self) -> list:
        """Full session risk report from RiskScorer."""
        return self.risk.get_report()

    def cleanup(self) -> None:
        """Release all MediaPipe resources."""
        self.face.cleanup()
        self.obj.cleanup()
        self.pose.cleanup()

    # ------------------------------------------------------------------
    # Adaptive frame budget
    # ------------------------------------------------------------------

    def _adapt_skips(self, frame_ms: float) -> None:
        """Adjust frame skips on slow modules based on recent frame times."""
        self._frame_ms_history.append(frame_ms)
        if len(self._frame_ms_history) > 10:
            self._frame_ms_history.pop(0)
        if len(self._frame_ms_history) < 5:
            return

        avg_ms = sum(self._frame_ms_history) / len(self._frame_ms_history)

        if avg_ms > _BUDGET_HIGH_MS:
            # Slow down heavy modules
            new_obj  = min(_MAX_SLOW_SKIP, self._obj_skip  + 1)
            new_pose = min(_MAX_SLOW_SKIP, self._pose_skip + 1)
            if new_obj != self._obj_skip:
                self.obj.frame_skip  = new_obj
                self._obj_skip       = new_obj
            if new_pose != self._pose_skip:
                self.pose.frame_skip = new_pose
                self._pose_skip      = new_pose

        elif avg_ms < _BUDGET_LOW_MS:
            # Speed up if we have headroom
            new_obj  = max(0, self._obj_skip  - 1)
            new_pose = max(0, self._pose_skip - 1)
            if new_obj != self._obj_skip:
                self.obj.frame_skip  = new_obj
                self._obj_skip       = new_obj
            if new_pose != self._pose_skip:
                self.pose.frame_skip = new_pose
                self._pose_skip      = new_pose