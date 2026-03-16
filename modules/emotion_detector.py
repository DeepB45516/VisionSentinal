"""
Sentinel AI — Emotion Detector Module  (NEW in v3)
====================================================
Processes blendshape data already produced by FaceIntelligence (zero extra
inference cost) to generate higher-level exam-integrity emotion events.

Why blendshapes work for exam cheating:
  Cheating students exhibit measurable micro-expressions:
    • Anxiety     — raised inner brow, brow furrow
    • Surprise    — wide eyes + jaw open (caught / startled by proctor)
    • Stress      — pressed lips, nose sneer, cheek squint
    • Deceptive   — asymmetric mouth, rapid eye blink (cognitive load spike)
    • Concentration Overload — sustained squint + brow down (reading hidden notes)

None of these require a separate "emotion" model — they fall directly out of
the FaceLandmarker blendshape coefficients already computed by FaceIntelligence.

Usage
-----
    from modules.emotion_detector import EmotionDetector
    detector = EmotionDetector()
    events   = detector.process(face_intel_result["emotions"])

Output event types (fed to RiskScorer):
    "anxious_expression"    → risk +15
    "surprised_expression"  → risk +18
    "stressed_expression"   → risk +12
    "suspicious_blink"      → risk +10
    "concentration_overload"→ risk +8
"""

import time
from collections import deque


# ---------------------------------------------------------------------------
# Thresholds (tune per environment lighting)
# ---------------------------------------------------------------------------
_ANXIETY_THRESH       = 0.36
_SURPRISE_THRESH      = 0.48
_STRESS_THRESH        = 0.38
_DECEPTIVE_ASYM_THRESH = 0.22   # |smile_L - smile_R| > this = asymmetric mouth
_BLINK_RATE_THRESH    = 4       # blinks per 5-second window (elevated = stress)
_CONC_OVERLOAD_THRESH = 0.45    # sustained squint+brow_down = hiding notes focus

# Cooldown: minimum seconds between same event for same student
_COOLDOWN: dict = {
    "anxious_expression":     8,
    "surprised_expression":   5,
    "stressed_expression":    10,
    "suspicious_blink":       12,
    "concentration_overload": 15,
}


class EmotionDetector:
    """
    Stateful emotion event generator from FaceIntelligence blendshape output.

    Maintains per-student:
      - Smoothed blendshape scores (EMA)
      - Blink rate window
      - Event cooldown timers
    """

    def __init__(self):
        self._ema:      dict = {}   # sid → {bs_name: smoothed_score}
        self._blinks:   dict = {}   # sid → deque of blink timestamps
        self._cooldowns: dict = {}  # sid:event_type → last_alert_time
        self._ema_alpha = 0.40

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, emotions: list, raw_blendshapes_map: dict = None) -> list:
        """
        Generate emotion-based risk events.

        Parameters
        ----------
        emotions : list[dict]
            Output of FaceIntelligence.process()["emotions"].
            Each entry: {id, anxiety, surprise, stress, label}

        raw_blendshapes_map : dict (optional)
            sid → raw blendshape dict {name: score} for advanced features.
            If None, advanced features (blink rate, deceptive asymmetry) are skipped.

        Returns
        -------
        list[dict]:
            Each: {student_id, event_type, label, severity, confidence}
        """
        now    = time.time()
        events = []

        for emo in emotions:
            sid      = emo["id"]
            anxiety  = emo.get("anxiety",  0.0)
            surprise = emo.get("surprise", 0.0)
            stress   = emo.get("stress",   0.0)

            # EMA-smooth the incoming scores
            s = self._ema.setdefault(sid, {})
            a  = self._ema_alpha
            s["anxiety"]  = a * anxiety  + (1 - a) * s.get("anxiety",  0.0)
            s["surprise"] = a * surprise + (1 - a) * s.get("surprise", 0.0)
            s["stress"]   = a * stress   + (1 - a) * s.get("stress",   0.0)

            # ── Anxiety ──────────────────────────────────────────────
            if s["anxiety"] >= _ANXIETY_THRESH:
                if self._check_cooldown(sid, "anxious_expression", now):
                    events.append(self._make_event(
                        sid, "anxious_expression",
                        "Anxious Expression", "med",
                        min(0.95, s["anxiety"] / _ANXIETY_THRESH * 0.7)
                    ))

            # ── Surprise (caught reaction) ────────────────────────────
            if s["surprise"] >= _SURPRISE_THRESH:
                if self._check_cooldown(sid, "surprised_expression", now):
                    events.append(self._make_event(
                        sid, "surprised_expression",
                        "Surprised / Caught Expression", "high",
                        min(0.95, s["surprise"] / _SURPRISE_THRESH * 0.75)
                    ))

            # ── Stress / Tension ──────────────────────────────────────
            if s["stress"] >= _STRESS_THRESH:
                if self._check_cooldown(sid, "stressed_expression", now):
                    events.append(self._make_event(
                        sid, "stressed_expression",
                        "Stressed Expression", "low",
                        min(0.90, s["stress"] / _STRESS_THRESH * 0.65)
                    ))

            # Advanced features require raw blendshapes
            if raw_blendshapes_map and sid in raw_blendshapes_map:
                bs = raw_blendshapes_map[sid]
                events.extend(self._advanced_events(sid, bs, now))

        return events

    def update_blink(self, student_id: str) -> None:
        """
        Call each time a blink is detected (e.g. from eyeBlinkLeft/Right blendshape
        crossing threshold). Tracked internally for blink-rate analysis.
        """
        now = time.time()
        dq  = self._blinks.setdefault(student_id, deque())
        dq.append(now)
        # Keep only last 5-second window
        while dq and now - dq[0] > 5.0:
            dq.popleft()

    # ------------------------------------------------------------------
    # Advanced event analysis (uses raw blendshape scores)
    # ------------------------------------------------------------------

    def _advanced_events(self, sid: str, bs: dict, now: float) -> list:
        """
        Additional events from raw blendshape dict {name: score}.
        """
        events = []

        def g(k: str) -> float:
            return bs.get(k, 0.0)

        # ── Deceptive facial asymmetry (asymmetric forced smile) ─────
        smile_asym = abs(g("mouthSmileLeft") - g("mouthSmileRight"))
        if smile_asym > _DECEPTIVE_ASYM_THRESH:
            if self._check_cooldown(sid, "suspicious_blink", now, override_cd=15):
                events.append(self._make_event(
                    sid, "suspicious_blink",
                    "Deceptive Facial Asymmetry", "low",
                    round(min(0.75, smile_asym / 0.4), 2)
                ))

        # ── Elevated blink rate ───────────────────────────────────────
        blink_l = g("eyeBlinkLeft")
        blink_r = g("eyeBlinkRight")
        if (blink_l > 0.55 or blink_r > 0.55):
            self.update_blink(sid)

        blink_window = self._blinks.get(sid, deque())
        if len(blink_window) >= _BLINK_RATE_THRESH:
            if self._check_cooldown(sid, "suspicious_blink", now):
                events.append(self._make_event(
                    sid, "suspicious_blink",
                    f"High Blink Rate ({len(blink_window)}/5s)", "low",
                    round(min(0.80, len(blink_window) / 8.0), 2)
                ))

        # ── Concentration overload (sustained reading focus) ──────────
        # Student staring intently at something below frame (cheat sheet)
        squint  = (g("eyeSquintLeft") + g("eyeSquintRight")) / 2
        brow_dn = (g("browDownLeft") + g("browDownRight")) / 2
        conc    = 0.55 * squint + 0.45 * brow_dn
        if conc >= _CONC_OVERLOAD_THRESH:
            if self._check_cooldown(sid, "concentration_overload", now):
                events.append(self._make_event(
                    sid, "concentration_overload",
                    "Sustained Focus (possible cheat sheet)", "med",
                    round(min(0.85, conc / _CONC_OVERLOAD_THRESH * 0.7), 2)
                ))

        return events

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_event(sid: str, etype: str, label: str,
                    severity: str, confidence: float) -> dict:
        return {
            "student_id": sid,
            "event_type": etype,
            "label":      label,
            "severity":   severity,
            "confidence": round(confidence, 3),
        }

    def _check_cooldown(self, sid: str, event_type: str,
                        now: float, override_cd: int = None) -> bool:
        key      = f"{sid}:{event_type}"
        cd       = override_cd if override_cd is not None else _COOLDOWN.get(event_type, 10)
        last     = self._cooldowns.get(key, 0.0)
        if now - last >= cd:
            self._cooldowns[key] = now
            return True
        return False
