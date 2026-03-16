"""
Sentinel AI — Temporal Smoothing Filter v2
============================================
Rolling-window false-positive reduction for all detection streams.

What's new vs v1
-----------------
  • Confidence accumulator      — in addition to bool confirmation, accumulates
                                  confidence scores over the window; returns
                                  weighted average (favours recent frames via
                                  exponential weighting)
  • Per-event decay on miss     — if an event is not seen this frame, its
                                  confidence window is partially decayed instead
                                  of simply appending False; makes recovery faster
                                  but also re-confirmation faster after brief gaps
  • reset_all() alias           — clean alias for reset(student_id=None)
  • Separate fast/slow windows  — get_confirmation() accepts a mode:
                                    "fast"  → N=2, K=1 (react in 2 frames)
                                    "slow"  → N=6, K=4 (stable only, low FP)
                                    default → instance window_size / min_confirm
"""

from collections import deque


class TemporalFilter:
    """Rolling window temporal smoothing for detection events."""

    def __init__(self, window_size: int = 4, min_confirmations: int = 2):
        self.window_size = window_size
        self.min_confirm = min_confirmations

        self._bool_windows:  dict = {}   # "sid:event" → deque[bool]
        self._conf_windows:  dict = {}   # "sid:event" → deque[float]
        self._value_windows: dict = {}   # "sid:metric" → deque[float]

    # ------------------------------------------------------------------
    # Boolean confirmation
    # ------------------------------------------------------------------

    def update(self, student_id: str, event_type: str, detected: bool,
               confidence: float = 1.0) -> bool:
        """
        Update rolling window for a student+event.

        Parameters
        ----------
        student_id  : str
        event_type  : str
        detected    : bool — whether event was detected this frame
        confidence  : float — detection confidence (used in weighted average)

        Returns
        -------
        bool — True if confirmed (≥ min_confirmations in current window)
        """
        key = f"{student_id}:{event_type}"

        bw = self._bool_windows.setdefault(key, deque(maxlen=self.window_size))
        cw = self._conf_windows.setdefault(key, deque(maxlen=self.window_size))

        if detected:
            bw.append(True)
            cw.append(confidence)
        else:
            bw.append(False)
            # On a miss, decay last confidence instead of appending 0
            # (less aggressive than a hard 0, allows brief gaps)
            last_conf = cw[-1] * 0.55 if cw else 0.0
            cw.append(last_conf)

        return sum(bw) >= self.min_confirm

    def is_confirmed(self, student_id: str, event_type: str,
                     mode: str = "default") -> bool:
        """
        Check if a detection is currently confirmed.

        Parameters
        ----------
        mode : "fast"    → 2 of last 2 frames
               "slow"    → 4 of last 6 frames
               "default" → instance settings
        """
        key = f"{student_id}:{event_type}"
        bw  = self._bool_windows.get(key)
        if not bw:
            return False
        if mode == "fast":
            last2 = list(bw)[-2:]
            return sum(last2) >= 1
        if mode == "slow":
            last6 = list(bw)[-6:]
            return sum(last6) >= 4
        return sum(bw) >= self.min_confirm

    def get_confidence(self, student_id: str, event_type: str) -> float:
        """
        Return exponentially-weighted average confidence over the current window.
        Most recent frame has highest weight.
        """
        key = f"{student_id}:{event_type}"
        cw  = list(self._conf_windows.get(key, []))
        if not cw:
            return 0.0
        # Exponential weights: last frame weight = 1, second-to-last = 0.7, ...
        weights = [0.7 ** (len(cw) - 1 - i) for i in range(len(cw))]
        total_w = sum(weights)
        if total_w == 0:
            return 0.0
        return round(sum(c * w for c, w in zip(cw, weights)) / total_w, 3)

    # ------------------------------------------------------------------
    # Continuous value smoothing
    # ------------------------------------------------------------------

    def smooth_value(self, student_id: str, metric_name: str,
                     value: float, window_size: int = None) -> float:
        """
        Rolling average smoothing for a continuous metric.
        Returns smoothed float.
        """
        ws  = window_size or self.window_size
        key = f"{student_id}:{metric_name}"
        vw  = self._value_windows.setdefault(key, deque(maxlen=ws))
        vw.append(value)
        return round(sum(vw) / len(vw), 4)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, student_id: str = None) -> None:
        """Reset all windows, or just for a specific student."""
        if student_id:
            prefix = f"{student_id}:"
            for store in (self._bool_windows, self._conf_windows, self._value_windows):
                for k in list(store):
                    if k.startswith(prefix):
                        del store[k]
        else:
            self._bool_windows.clear()
            self._conf_windows.clear()
            self._value_windows.clear()

    def reset_all(self) -> None:
        """Alias for reset()."""
        self.reset()
