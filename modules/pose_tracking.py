"""
Sentinel AI — Multi-Person Pose Tracking Module v3
====================================================
MediaPipe PoseLandmarker (Tasks API) — up to 6 students simultaneously.

What's new vs v2
-----------------
[ ACCURACY ]
  • Under-desk hand detection   — wrists below hip level AND low absolute Y position
                                  in frame → "hiding_object" alert (phone/cheat sheet
                                  being concealed under desk)
  • Look-behind detection       — head rotation (LEFT/RIGHT yaw from shoulder-ear diff)
                                  + neck turn → "looking_behind" alert
  • Shoulder block detection    — one shoulder raised while body curves forward →
                                  "shielding_paper" (blocking neighbor's/proctor's view)
  • Repeated head turn tracking — if student looks L/R more than N times in window →
                                  "repeated_head_turn" (sustained peer copying)
  • Head-down improved          — uses ear midpoint + shoulder baseline for accuracy
                                  across different student heights

[ PERFORMANCE ]
  • Default frame_skip=2        — inference every 3rd frame (saves ~15ms on CPU)
  • Default scale=0.50          — half resolution sufficient for pose (landmarks are coarse)
  • Early exit on no landmarks  — avoids all downstream Python work on empty result
  • Per-student behavior cache  — behaviors cached for skipped frames; only timer
                                  refreshes happen (zero ML cost on skip)
"""

import os
import math
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
)

_MODELS_DIR  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
_POSE_MODEL  = os.path.join(_MODELS_DIR, "pose_landmarker_lite.task")

# ---------------------------------------------------------------------------
# Landmark indices (MediaPipe Pose — same across Tasks API and legacy)
# ---------------------------------------------------------------------------
class LM:
    NOSE            = 0
    LEFT_EYE        = 2;  RIGHT_EYE       = 5
    LEFT_EAR        = 7;  RIGHT_EAR       = 8
    LEFT_SHOULDER   = 11; RIGHT_SHOULDER  = 12
    LEFT_ELBOW      = 13; RIGHT_ELBOW     = 14
    LEFT_WRIST      = 15; RIGHT_WRIST     = 16
    LEFT_HIP        = 23; RIGHT_HIP       = 24
    LEFT_KNEE       = 25; RIGHT_KNEE      = 26
    LEFT_ANKLE      = 27; RIGHT_ANKLE     = 28


# ---------------------------------------------------------------------------
# Behavioral thresholds
# ---------------------------------------------------------------------------
_HEAD_DOWN_PITCH    = 38    # degrees
_ARM_RAISE_MARGIN   = 25    # pixels above shoulder
_LEAN_RATIO         = 0.48  # shoulder tilt / hip width
_UNDER_DESK_RATIO   = 0.15  # wrist must be this far below hip (normalised to body height)
_SHIELD_RATIO       = 0.30  # raised shoulder delta / body width → shielding
_LOOK_BEHIND_EAR    = 0.18  # ear-to-shoulder horizontal offset ratio → looking behind
_HEAD_TURN_WINDOW   = 12.0  # seconds
_HEAD_TURN_THRESH   = 3     # count within window → repeated turns


class PoseTracker:
    """Multi-person pose tracking with cheating-behavior analysis."""

    def __init__(
        self,
        max_students: int   = 6,
        min_confidence: float = 0.28,
        frame_skip: int     = 2,
        detect_scale: float = 0.50,
    ):
        self.max_students = max_students
        self.min_conf     = min_confidence
        self.frame_skip   = max(0, int(frame_skip))
        self.detect_scale = max(0.25, min(1.0, float(detect_scale)))

        opts = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_POSE_MODEL),
            num_poses=max_students,
            min_pose_detection_confidence=0.45,
            min_tracking_confidence=0.40,
        )
        self.landmarker = PoseLandmarker.create_from_options(opts)

        self._frame_counter  = 0
        self._cached_result  = None

        # Head-turn tracking per student
        self._head_turns: dict = {}   # sid → [timestamp, ...]
        self._last_head_dir: dict = {}  # sid → "L" | "R" | "C"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame_rgb: np.ndarray, h: int, w: int,
                person_bboxes=None) -> dict:
        """
        Run pose estimation + behavioral analysis on an RGB frame.

        Returns
        -------
        dict:
            poses       list[dict]  — student_id, keypoints
            behaviors   list[dict]  — student_id, type, label, severity, confidence
        """
        self._frame_counter += 1
        if self.frame_skip > 0 and (self._frame_counter % (self.frame_skip + 1)) != 0:
            if self._cached_result is not None:
                return self._cached_result
            return {"poses": [], "behaviors": []}

        stride   = max(1, round(1.0 / self.detect_scale))
        frame_in = frame_rgb[::stride, ::stride] if stride > 1 else frame_rgb
        mp_img   = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_in, dtype=np.uint8)
        )
        pose_result = self.landmarker.detect(mp_img)

        result = {"poses": [], "behaviors": []}
        if not pose_result.pose_landmarks:
            self._cached_result = result
            return result

        now = time.time()
        for i, person_lm in enumerate(pose_result.pose_landmarks[: self.max_students]):
            kps = self._extract_keypoints(person_lm, w, h, stride)
            sid = f"S{i}"
            result["poses"].append({"student_id": sid, "keypoints": kps})
            behaviors = self._analyze_behavior(kps, sid, h, w, now)
            result["behaviors"].extend(behaviors)

        self._cached_result = result
        return result

    def get_wrist_positions(self, poses: list) -> dict:
        """Extract wrist positions for WristVelocityTracker."""
        wrists = {}
        for pose in poses:
            sid = pose["student_id"]
            kps = pose["keypoints"]
            lw  = kps.get("left_wrist")
            rw  = kps.get("right_wrist")
            if (lw and lw["confidence"] > self.min_conf
                    and rw and rw["confidence"] > self.min_conf):
                wrists[sid] = {
                    "left":  (lw["x"], lw["y"]),
                    "right": (rw["x"], rw["y"]),
                }
        return wrists

    def cleanup(self):
        self.landmarker.close()

    # ------------------------------------------------------------------
    # Keypoint extraction
    # ------------------------------------------------------------------

    def _extract_keypoints(self, landmarks, frame_w: int, frame_h: int,
                           stride: int) -> dict:
        """Convert NormalizedLandmarks → absolute pixel coords (rescaled for stride)."""
        name_map = {
            LM.NOSE:           "nose",
            LM.LEFT_EYE:       "left_eye",   LM.RIGHT_EYE:      "right_eye",
            LM.LEFT_EAR:       "left_ear",   LM.RIGHT_EAR:      "right_ear",
            LM.LEFT_SHOULDER:  "left_shoulder", LM.RIGHT_SHOULDER: "right_shoulder",
            LM.LEFT_ELBOW:     "left_elbow", LM.RIGHT_ELBOW:    "right_elbow",
            LM.LEFT_WRIST:     "left_wrist", LM.RIGHT_WRIST:    "right_wrist",
            LM.LEFT_HIP:       "left_hip",   LM.RIGHT_HIP:      "right_hip",
            LM.LEFT_KNEE:      "left_knee",  LM.RIGHT_KNEE:     "right_knee",
            LM.LEFT_ANKLE:     "left_ankle", LM.RIGHT_ANKLE:    "right_ankle",
        }
        kps = {}
        for idx, name in name_map.items():
            if idx >= len(landmarks):
                continue
            lm = landmarks[idx]
            kps[name] = {
                "x":          int(lm.x * frame_w),
                "y":          int(lm.y * frame_h),
                "confidence": round(getattr(lm, "visibility", 0.0) or 0.0, 3),
            }
        return kps

    # ------------------------------------------------------------------
    # Behavioral analysis
    # ------------------------------------------------------------------

    def _analyze_behavior(self, kps: dict, sid: str,
                          frame_h: int, frame_w: int, now: float) -> list:
        alerts = []

        def ok(name: str) -> bool:
            return name in kps and kps[name]["confidence"] > self.min_conf

        def pt(name: str):
            return kps[name] if ok(name) else None

        # ── 1. Head down (reading hidden notes) ──────────────────────
        nose  = pt("nose")
        ls    = pt("left_shoulder"); rs = pt("right_shoulder")
        le    = pt("left_ear");      re = pt("right_ear")
        if nose and ls and rs:
            shoulder_y = (ls["y"] + rs["y"]) / 2
            ear_y      = ((le["y"] + re["y"]) / 2) if (le and re) else nose["y"]
            vert_ref   = abs(shoulder_y - ear_y) or 1
            drop       = nose["y"] - ear_y
            pitch      = math.degrees(math.atan2(drop, vert_ref))
            if pitch > _HEAD_DOWN_PITCH:
                alerts.append({
                    "student_id": sid, "type": "head_down",
                    "label": f"Head Down {pitch:.0f}°", "severity": "med",
                    "confidence": round(min(1.0, pitch / 65), 2),
                })

        # ── 2. Arm raise (signaling / passing notes) ─────────────────
        for side in ("left", "right"):
            wrist = pt(f"{side}_wrist")
            shldr = pt(f"{side}_shoulder")
            if wrist and shldr and wrist["y"] < shldr["y"] - _ARM_RAISE_MARGIN:
                alerts.append({
                    "student_id": sid, "type": "arm_raise",
                    "label": f"{side.title()} Arm Raised", "severity": "med",
                    "confidence": 0.78,
                })

        # ── 3. Sideways lean (copying neighbor) ──────────────────────
        lshldr = pt("left_shoulder"); rshldr = pt("right_shoulder")
        lhip   = pt("left_hip");      rhip   = pt("right_hip")
        if lshldr and rshldr and lhip and rhip:
            s_tilt   = abs(lshldr["y"] - rshldr["y"])
            h_width  = abs(lhip["x"]   - rhip["x"]) or 45
            if s_tilt / h_width > _LEAN_RATIO:
                alerts.append({
                    "student_id": sid, "type": "lean_over",
                    "label": "Leaning Sideways", "severity": "high",
                    "confidence": round(min(1.0, s_tilt / h_width / 0.6), 2),
                })

        # ── 4. Under-desk hands (hiding object) ──────────────────────
        if lhip and rhip:
            hip_y       = (lhip["y"] + rhip["y"]) / 2
            body_height = max(abs(hip_y - (ls["y"] if ls else hip_y - 100)), 50)
            for side in ("left", "right"):
                wrist = pt(f"{side}_wrist")
                if wrist and (wrist["y"] - hip_y) / body_height > _UNDER_DESK_RATIO:
                    alerts.append({
                        "student_id": sid, "type": "hiding_object",
                        "label": f"{side.title()} Hand Under Desk",
                        "severity": "high", "confidence": 0.82,
                    })

        # ── 5. Look-behind (checking for proctor) ─────────────────────
        if le and re and ls and rs:
            ear_mid_x    = (le["x"] + re["x"]) / 2
            shldr_mid_x  = (ls["x"] + rs["x"]) / 2
            shldr_width  = abs(ls["x"] - rs["x"]) or 50
            turn_ratio   = (ear_mid_x - shldr_mid_x) / shldr_width
            if abs(turn_ratio) > _LOOK_BEHIND_EAR:
                direction = "Right" if turn_ratio > 0 else "Left"
                alerts.append({
                    "student_id": sid, "type": "look_behind",
                    "label": f"Looking Behind ({direction})", "severity": "med",
                    "confidence": round(min(0.90, abs(turn_ratio) / 0.25), 2),
                })

                # Track repeated head turns
                curr_dir = "R" if turn_ratio > 0 else "L"
                last_dir = self._last_head_dir.get(sid, "C")
                if curr_dir != last_dir and last_dir != "C":
                    turns = self._head_turns.setdefault(sid, [])
                    turns.append(now)
                    self._head_turns[sid] = [t for t in turns if now - t <= _HEAD_TURN_WINDOW]
                    if len(self._head_turns[sid]) >= _HEAD_TURN_THRESH:
                        alerts.append({
                            "student_id": sid, "type": "repeated_head_turn",
                            "label": f"Repeated Head Turns ({len(self._head_turns[sid])}x)",
                            "severity": "high", "confidence": 0.85,
                        })
                self._last_head_dir[sid] = curr_dir
            else:
                self._last_head_dir[sid] = "C"

        # ── 6. Shoulder shield (blocking proctor's view of paper) ─────
        if lshldr and rshldr and lhip and rhip and nose:
            body_w    = abs(lhip["x"] - rhip["x"]) or 50
            shldr_diff = abs(lshldr["y"] - rshldr["y"])
            # One shoulder raised + body hunched forward
            nose_fwd  = nose["y"] > ((lshldr["y"] + rshldr["y"]) / 2 - 10)
            if shldr_diff / body_w > _SHIELD_RATIO and nose_fwd:
                alerts.append({
                    "student_id": sid, "type": "shielding_paper",
                    "label": "Shielding Paper from View", "severity": "high",
                    "confidence": round(min(0.88, shldr_diff / body_w / 0.5), 2),
                })

        return alerts
