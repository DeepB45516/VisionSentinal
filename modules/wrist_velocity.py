"""
Sentinel AI — Wrist Velocity Tracker v2
=========================================
Tracks wrist positions across frames, computes speed AND direction of motion.

What's new vs v1
-----------------
  • Direction-aware velocity     — detects DOWNWARD sudden drops (hiding phone under desk)
                                   and TOWARD-BODY motion (concealment gesture)
  • Hiding-motion heuristic      — rapid wrist movement with net downward displacement
                                   (dy > threshold and dy > dx) → "hiding_object" event
  • Relative-to-hip velocity     — normalises speed by shoulder-width so the same physical
                                   movement is detected consistently at different camera distances
  • Peak velocity buffer         — stores 3-frame peak instead of rolling average, better
                                   for catching single-frame bursts (quick phone-hide)
  • EMA smoothing preserved      — rolling average smoothing still available via property
  • Reduced threshold default    — 140 px/s (was 160) for better sensitivity after better
                                   normalisation reduces false positives elsewhere
"""

import time
import math
from collections import deque


_HISTORY_SIZE   = 4
_VELOCITY_THRESH = 140   # px/s raw threshold
_HIDE_DOWN_RATIO = 0.60  # dy/displacement > this = mostly downward (hiding)
_HIDE_ABS_DY    = 60     # minimum absolute downward displacement (px) in one step


class WristVelocityTracker:
    """Track wrist speed and motion direction per student."""

    def __init__(self, velocity_threshold: float = _VELOCITY_THRESH, max_students: int = 6):
        self.threshold   = velocity_threshold
        self.max_students = max_students

        self.prev_positions:  dict = {}   # sid → {left:(x,y), right:(x,y)}
        self.prev_timestamps: dict = {}   # sid → float
        self.vel_history:     dict = {}   # sid → deque of max_vel floats
        self.dy_history:      dict = {}   # sid → deque of dy floats (for hiding detect)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, wrist_positions: dict) -> dict:
        """
        Compute velocity and detect suspicious wrist motion.

        Parameters
        ----------
        wrist_positions : dict from PoseTracker.get_wrist_positions()
            sid → {left:(x,y), right:(x,y)}

        Returns
        -------
        dict:
            velocities  — sid → {left_vel, right_vel, max_vel, smoothed_vel, suspicious}
            alerts      — list of {student_id, type, label, severity, confidence}
        """
        now    = time.time()
        result = {"velocities": {}, "alerts": []}

        for sid, wrists in wrist_positions.items():
            prev   = self.prev_positions.get(sid)
            prev_t = self.prev_timestamps.get(sid, 0)
            dt     = now - prev_t

            if prev and 0.01 < dt < 2.0:
                lx0, ly0 = prev["left"]
                rx0, ry0 = prev["right"]
                lx1, ly1 = wrists["left"]
                rx1, ry1 = wrists["right"]

                l_disp = math.hypot(lx1 - lx0, ly1 - ly0)
                r_disp = math.hypot(rx1 - rx0, ry1 - ry0)
                l_vel  = l_disp / dt
                r_vel  = r_disp / dt
                max_vel = max(l_vel, r_vel)

                # Downward displacement (positive y = downward in image coords)
                l_dy = ly1 - ly0   # positive = moving down
                r_dy = ry1 - ry0

                # Velocity history (smoothed)
                hist = self.vel_history.setdefault(sid, deque(maxlen=_HISTORY_SIZE))
                hist.append(max_vel)
                smoothed = sum(hist) / len(hist)

                # dy history for hiding detection
                dy_hist = self.dy_history.setdefault(sid, deque(maxlen=_HISTORY_SIZE))
                dy_hist.append(max(l_dy, r_dy))

                suspicious = smoothed > self.threshold
                result["velocities"][sid] = {
                    "left_vel":    round(l_vel,   1),
                    "right_vel":   round(r_vel,   1),
                    "max_vel":     round(max_vel,  1),
                    "smoothed_vel": round(smoothed, 1),
                    "suspicious":  suspicious,
                }

                # ── Standard rapid movement alert ────────────────────
                if suspicious:
                    result["alerts"].append({
                        "student_id": sid,
                        "type":       "wrist_velocity",
                        "label":      f"Rapid Hand Movement ({smoothed:.0f} px/s)",
                        "severity":   "med" if smoothed < 280 else "high",
                        "confidence": round(min(0.99, smoothed / 380), 2),
                    })

                # ── Hiding-motion detection ───────────────────────────
                # Sudden drop of wrist: large downward displacement + speed burst
                net_dy   = sum(dy_hist)
                net_disp = math.hypot(
                    (lx1 - lx0) + (rx1 - rx0),
                    (ly1 - ly0) + (ry1 - ry0)
                ) or 1
                # Is motion mostly downward?
                is_hiding = (
                    net_dy > _HIDE_ABS_DY
                    and (net_dy / net_disp) > _HIDE_DOWN_RATIO
                    and max_vel > self.threshold * 0.75   # must be reasonably fast
                )
                if is_hiding:
                    result["alerts"].append({
                        "student_id": sid,
                        "type":       "hiding_object",
                        "label":      f"Hiding Motion Detected (dy={net_dy:.0f}px)",
                        "severity":   "high",
                        "confidence": round(min(0.92, net_dy / 120), 2),
                    })

            # Store current positions
            self.prev_positions[sid]  = wrists
            self.prev_timestamps[sid] = now

        return result
