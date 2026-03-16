"""
Sentinel AI — Risk Scoring Module v3
======================================
Per-student risk score (0–100) with behavioral event accumulation and time decay.

What's new vs v2
-----------------
  • New event types added:
      - hiding_object          +28  (under-desk hand / hiding motion)
      - look_behind            +12  (checking for proctor)
      - repeated_head_turn     +22  (sustained peer observation)
      - shielding_paper        +20  (blocking view)
      - anxious_expression     +10  (emotion signal)
      - surprised_expression   +18  (caught-reaction signal)
      - stressed_expression    +8   (background anxiety)
      - suspicious_blink       +8   (elevated blink rate / asymmetry)
      - concentration_overload +12  (focused reading → hidden material)
      - smartwatch             +35
      - notepad                +32
      - emotion_correlation    +15  (coordinated stress pair)
      - pass_back              +28
      - sync_head_down         +22

  • Composite risk events:
      If phone_detected + gaze_down fire in the same 3-second window → composite
      "phone_gaze_combo" adds +20 extra on top of individual scores.

  • Severity-weighted decay:
      Students with high risk (≥70) decay 40% slower than low-risk students,
      keeping them flagged until a proctor reviews.

  • Session summary with top-risk student tracking.

Score ranges:
    0–19  : Safe
    20–39 : Watch
    40–69 : Suspicious
    70–89 : High Risk
    90–100: Critical
"""

import time
from collections import defaultdict


# ---------------------------------------------------------------------------
# Base event weights
# ---------------------------------------------------------------------------
EVENT_WEIGHTS: dict = {
    # Object detections
    "cell phone":             40,
    "earphone":               42,
    "earbuds":                42,
    "headphones":             35,
    "laptop":                 35,
    "tablet":                 35,
    "smartwatch":             35,
    "book":                   30,
    "notepad":                32,
    "paper":                  22,
    "device":                 18,
    "water_bottle":            8,
    "scissors":                5,
    "knife":                  10,
    # Gaze
    "gaze_left":              10,
    "gaze_right":             10,
    "gaze_down":               8,
    "gaze_up":                 5,
    # Pose behaviors
    "head_down":              10,
    "arm_raise":              12,
    "lean_over":              20,
    "hiding_object":          28,
    "look_behind":            12,
    "repeated_head_turn":     22,
    "shielding_paper":        20,
    # Wrist
    "wrist_velocity":         15,
    # Coordination
    "coordinated_movement":   30,
    "mutual_gaze":            25,
    "gaze_movement_combo":    20,
    "sync_head_down":         22,
    "pass_back":              28,
    "emotion_correlation":    15,
    # Absence / identity
    "face_absent":            25,
    "proxy":                  45,
    # Emotion
    "anxious_expression":     10,
    "surprised_expression":   18,
    "stressed_expression":     8,
    "suspicious_blink":        8,
    "concentration_overload": 12,
    # Composite
    "phone_gaze_combo":       20,
    # Fallback
    "unknown":                 5,
}

# Composite trigger pairs: (event_type_A, event_type_B) → composite_event, window_secs
_COMPOSITE_RULES: list = [
    (("cell phone", "gaze_down"),       "phone_gaze_combo", 3.0),
    (("book",       "gaze_down"),       "phone_gaze_combo", 3.0),
    (("notepad",    "gaze_down"),       "phone_gaze_combo", 3.0),
    (("hiding_object", "wrist_velocity"), "phone_gaze_combo", 2.0),
]

# Severity labels
_LABELS = [
    (90, "critical"),
    (70, "high"),
    (40, "suspicious"),
    (20, "watch"),
    (0,  "safe"),
]


class RiskScorer:
    """Per-student behavioral risk scoring with composite rules and adaptive decay."""

    def __init__(self, max_score: float = 100.0, decay_rate: float = 0.995):
        """
        Parameters
        ----------
        max_score  : hard ceiling for risk score
        decay_rate : multiplicative decay per 100ms tick (default ~1%/s)
                     High-risk students use 0.997 (slower decay) automatically.
        """
        self.max_score  = max_score
        self.decay_rate = decay_rate

        self.scores:      dict = {}   # sid → float
        self.last_update: dict = {}   # sid → float timestamp
        self.event_log:   dict = {}   # sid → [(event_type, timestamp, delta)]

        # For composite detection: last event times per student
        self._event_times: dict = defaultdict(dict)  # sid → {event_type: last_time}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_event(self, student_id: str, event_type: str) -> float:
        """
        Record a risk event. Returns the delta score added.
        Automatically evaluates composite rules.
        """
        now   = time.time()
        self._decay(student_id, now)

        delta = EVENT_WEIGHTS.get(event_type, EVENT_WEIGHTS["unknown"])
        self._apply_delta(student_id, event_type, delta, now)

        # Record event time for composite evaluation
        self._event_times[student_id][event_type] = now

        # Check composite rules
        for (type_a, type_b), composite, window in _COMPOSITE_RULES:
            if event_type not in (type_a, type_b):
                continue
            other = type_b if event_type == type_a else type_a
            last_other = self._event_times[student_id].get(other, 0)
            if now - last_other <= window:
                comp_delta = EVENT_WEIGHTS.get(composite, 0)
                self._apply_delta(student_id, composite, comp_delta, now)

        return delta

    def add_events_bulk(self, student_id: str, event_types: list) -> None:
        """Add multiple events for one student in one call (avoids repeated decay calls)."""
        now = time.time()
        self._decay(student_id, now)
        for event_type in event_types:
            delta = EVENT_WEIGHTS.get(event_type, EVENT_WEIGHTS["unknown"])
            self._apply_delta(student_id, event_type, delta, now)
            self._event_times[student_id][event_type] = now

    def get_score(self, student_id: str) -> int:
        self._decay(student_id, time.time())
        return round(self.scores.get(student_id, 0))

    def get_all_scores(self) -> dict:
        now = time.time()
        out = {}
        for sid in list(self.scores):
            self._decay(sid, now)
            out[sid] = round(self.scores[sid])
        return out

    def get_severity(self, student_id: str) -> str:
        score = self.get_score(student_id)
        for threshold, label in _LABELS:
            if score >= threshold:
                return label
        return "safe"

    def get_report(self) -> list:
        """Full session report sorted by descending risk."""
        scores = self.get_all_scores()
        report = []
        for sid, score in sorted(scores.items(), key=lambda x: -x[1]):
            events  = self.event_log.get(sid, [])
            report.append({
                "student_id":    sid,
                "risk_score":    score,
                "severity":      self.get_severity(sid),
                "event_count":   len(events),
                "recent_events": [
                    {"type": t, "delta": d}
                    for t, _, d in events[-8:]
                ],
            })
        return report

    def top_risk_student(self) -> tuple:
        """Return (student_id, score) for the highest-risk student, or (None, 0)."""
        scores = self.get_all_scores()
        if not scores:
            return None, 0
        sid = max(scores, key=scores.__getitem__)
        return sid, scores[sid]

    def reset_student(self, student_id: str) -> None:
        """Manually reset a student's score (e.g. after proctor review)."""
        self.scores.pop(student_id, None)
        self.last_update.pop(student_id, None)
        self.event_log.pop(student_id, None)
        self._event_times.pop(student_id, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_delta(self, sid: str, event_type: str, delta: float, now: float) -> None:
        current           = self.scores.get(sid, 0.0)
        self.scores[sid]  = min(self.max_score, current + delta)
        self.last_update[sid] = now

        log = self.event_log.setdefault(sid, [])
        log.append((event_type, now, delta))
        if len(log) > 60:
            self.event_log[sid] = log[-60:]

    def _decay(self, student_id: str, now: float) -> None:
        if student_id not in self.scores:
            return
        last    = self.last_update.get(student_id, now)
        elapsed = now - last
        if elapsed <= 0:
            return
        # High-risk students decay slower (stay flagged longer for review)
        score   = self.scores[student_id]
        rate    = 0.997 if score >= 70 else self.decay_rate
        ticks   = int(elapsed * 10)
        if ticks > 0:
            self.scores[student_id] = score * (rate ** ticks)
            if self.scores[student_id] < 0.5:
                self.scores[student_id] = 0.0
            self.last_update[student_id] = now
