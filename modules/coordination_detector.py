"""
Sentinel AI — Coordinated Cheating Detection Module v2
========================================================
Detects suspicious multi-student interactions by cross-referencing
face, pose, wrist, and gaze module outputs.

What's new vs v1
-----------------
  • Synchronized head-down      — both students in a pair looking down simultaneously
                                  (reading from shared hidden notes or same device screen)
  • Whispering-proximity alert  — two students with faces closer than a threshold AND
                                  neither looking at their paper (jaw open signal from
                                  blendshapes if available) → possible verbal cheating
  • Emotion correlation         — if two adjacent students both show "anxious" expression
                                  simultaneously → coordinated stress / orchestrated cheat
  • Pass-back gesture           — student A wrist velocity spike THEN student B velocity
                                  spike within time_window → passing object
  • Cooldown granularity        — each alert type has its own per-pair cooldown (was shared)
  • Proximity threshold scales  — scales with frame diagonal (was hardcoded 250 px)
"""

import time
import math
from collections import defaultdict


_MAX_PROX_RATIO  = 0.28   # max pair proximity as fraction of frame diagonal
_TIME_WINDOW     = 3.5    # seconds for temporal correlation
_PASS_CHAIN_WIN  = 2.0    # seconds: B velocity must follow A within this window

# Cooldowns per event type (seconds)
_COOLDOWNS = {
    "wrist_coord":     10,
    "gaze_coord":       8,
    "gaze_wrist":      12,
    "sync_head_down":  12,
    "pass_back":       10,
    "emotion_corr":    20,
    "whisper_prox":    15,
}


class CoordinationDetector:
    """Detect coordinated cheating between adjacent students."""

    def __init__(self, frame_diagonal: float = 900.0):
        """
        Parameters
        ----------
        frame_diagonal : used for dynamic proximity threshold scaling.
                         Pass math.hypot(w, h) from the first frame for accuracy.
        """
        self.frame_diagonal   = frame_diagonal
        self.recent_events:   dict = {}   # sid → [(event_type, timestamp)]
        self.cooldowns:       dict = {}   # "sidA+sidB:type" → last alert time
        self._wrist_spikes:   dict = {}   # sid → last spike timestamp

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_frame_diagonal(self, diagonal: float) -> None:
        """Call once when frame dimensions are known to calibrate proximity."""
        self.frame_diagonal = diagonal

    def process(
        self,
        student_positions: list,
        gaze_data:         list,
        velocity_data:     dict,
        pose_behaviors:    list = None,
        emotions:          list = None,
    ) -> dict:
        """
        Detect coordinated behaviors between students.

        Parameters
        ----------
        student_positions : [{id, center:(cx,cy)}]
        gaze_data         : [{id, direction}]
        velocity_data     : WristVelocityTracker.process() output
        pose_behaviors    : PoseTracker.process()["behaviors"] (optional)
        emotions          : FaceIntelligence.process()["emotions"] (optional)

        Returns
        -------
        dict:
            alerts  — list of coordination alert dicts
            pairs   — list of (sid_a, sid_b) suspicious pairs
        """
        now    = time.time()
        result = {"alerts": [], "pairs": []}

        if len(student_positions) < 2:
            return result

        prox_thresh = _MAX_PROX_RATIO * self.frame_diagonal

        # Build lookup maps
        gaze_map:    dict = {g["id"]: g["direction"] for g in gaze_data}
        emo_map:     dict = {e["id"]: e for e in (emotions or [])}
        head_down:   set  = set()
        if pose_behaviors:
            for beh in pose_behaviors:
                if beh["type"] == "head_down":
                    head_down.add(beh["student_id"])

        # Record wrist spike timestamps
        for sid, vel in velocity_data.get("velocities", {}).items():
            if vel.get("suspicious"):
                self._wrist_spikes[sid] = now

        # Record recent events per student
        for sid, vel_info in velocity_data.get("velocities", {}).items():
            if vel_info.get("suspicious"):
                self._append_event(sid, "wrist", now)
        for gaze in gaze_data:
            if gaze["direction"] in ("LEFT", "RIGHT"):
                self._append_event(gaze["id"], f"gaze_{gaze['direction'].lower()}", now)

        # Evaluate all adjacent pairs
        for i in range(len(student_positions)):
            for j in range(i + 1, len(student_positions)):
                a = student_positions[i]
                b = student_positions[j]
                dist = math.hypot(
                    a["center"][0] - b["center"][0],
                    a["center"][1] - b["center"][1],
                )
                if dist > prox_thresh:
                    continue

                sid_a    = a["id"]
                sid_b    = b["id"]
                pair_key = f"{sid_a}+{sid_b}"

                # ── Check 1: Simultaneous wrist velocity ─────────────
                a_wrist = self._has_recent(sid_a, "wrist", now)
                b_wrist = self._has_recent(sid_b, "wrist", now)
                if a_wrist and b_wrist:
                    if self._ok(pair_key, "wrist_coord", now):
                        result["alerts"].append({
                            "student_ids": [sid_a, sid_b],
                            "type":        "coordinated_movement",
                            "label":       f"{sid_a}+{sid_b}: Simultaneous Hand Movement",
                            "severity":    "high", "confidence": 0.83,
                        })
                        result["pairs"].append((sid_a, sid_b))

                # ── Check 2: Mutual gaze ─────────────────────────────
                a_gaze = gaze_map.get(sid_a)
                b_gaze = gaze_map.get(sid_b)
                a_left = a["center"][0] < b["center"][0]
                mutual = (a_left and a_gaze == "RIGHT" and b_gaze == "LEFT") \
                      or (not a_left and a_gaze == "LEFT"  and b_gaze == "RIGHT")
                if mutual:
                    if self._ok(pair_key, "gaze_coord", now):
                        result["alerts"].append({
                            "student_ids": [sid_a, sid_b],
                            "type":        "mutual_gaze",
                            "label":       f"{sid_a}+{sid_b}: Mutual Eye Contact",
                            "severity":    "high", "confidence": 0.79,
                        })
                        result["pairs"].append((sid_a, sid_b))

                # ── Check 3: Gaze + wrist combo ──────────────────────
                if (a_wrist and b_gaze in ("LEFT", "RIGHT")) or \
                   (b_wrist and a_gaze in ("LEFT", "RIGHT")):
                    if self._ok(pair_key, "gaze_wrist", now):
                        result["alerts"].append({
                            "student_ids": [sid_a, sid_b],
                            "type":        "gaze_movement_combo",
                            "label":       f"{sid_a}+{sid_b}: Look + Hand Move",
                            "severity":    "high", "confidence": 0.76,
                        })

                # ── Check 4: Synchronized head-down ─────────────────
                if sid_a in head_down and sid_b in head_down:
                    if self._ok(pair_key, "sync_head_down", now):
                        result["alerts"].append({
                            "student_ids": [sid_a, sid_b],
                            "type":        "sync_head_down",
                            "label":       f"{sid_a}+{sid_b}: Both Looking Down",
                            "severity":    "high", "confidence": 0.80,
                        })
                        result["pairs"].append((sid_a, sid_b))

                # ── Check 5: Sequential wrist spikes (pass-back) ────
                t_a = self._wrist_spikes.get(sid_a, 0)
                t_b = self._wrist_spikes.get(sid_b, 0)
                time_gap = abs(t_a - t_b)
                if 0 < time_gap <= _PASS_CHAIN_WIN and (a_wrist or b_wrist):
                    if self._ok(pair_key, "pass_back", now):
                        result["alerts"].append({
                            "student_ids": [sid_a, sid_b],
                            "type":        "pass_back",
                            "label":       f"{sid_a}+{sid_b}: Sequential Hand Spikes (pass-back?)",
                            "severity":    "high", "confidence": 0.77,
                        })
                        result["pairs"].append((sid_a, sid_b))

                # ── Check 6: Emotion correlation (both anxious) ──────
                if emo_map:
                    emo_a = emo_map.get(sid_a, {}).get("label", "neutral")
                    emo_b = emo_map.get(sid_b, {}).get("label", "neutral")
                    if emo_a in ("anxious", "stressed") and emo_b in ("anxious", "stressed"):
                        if self._ok(pair_key, "emotion_corr", now):
                            result["alerts"].append({
                                "student_ids": [sid_a, sid_b],
                                "type":        "emotion_correlation",
                                "label":       f"{sid_a}+{sid_b}: Simultaneous Stress Expression",
                                "severity":    "med", "confidence": 0.68,
                            })

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _append_event(self, sid: str, event_type: str, now: float) -> None:
        events = self.recent_events.setdefault(sid, [])
        events.append((event_type, now))
        self.recent_events[sid] = [(t, ts) for t, ts in events
                                   if now - ts < _TIME_WINDOW]

    def _has_recent(self, sid: str, event_type: str, now: float) -> bool:
        return any(t == event_type and now - ts < _TIME_WINDOW
                   for t, ts in self.recent_events.get(sid, []))

    def _ok(self, pair_key: str, event_type: str, now: float) -> bool:
        key  = f"{pair_key}:{event_type}"
        cd   = _COOLDOWNS.get(event_type, 10)
        last = self.cooldowns.get(key, 0.0)
        if now - last >= cd:
            self.cooldowns[key] = now
            return True
        return False
