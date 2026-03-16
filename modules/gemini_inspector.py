"""
Sentinel AI — Gemini Deep Inspector  (NEW in v4)
=================================================
Uses Google Gemini Flash (free API tier) to perform periodic deep-scan
analysis of each student's desk region.

Why Gemini for object detection?
---------------------------------
  YOLO/EfficientDet are trained on COCO-80 — a general-purpose dataset.
  Items like earpieces, handwritten cheat sheets, hidden notes under paper,
  folded papers, or very small phones partially hidden by hands score very
  low confidence in any standard object detector because they don't appear
  as clean isolated objects in COCO training data.

  Gemini Flash (gemini-1.5-flash) is a vision-language model that can
  reason about partial occlusion, context ("that small object near the ear
  looks like an earpiece"), and spatial relationships that standard detectors
  miss entirely.

Free tier limits (as of 2025):
  • 15 requests / minute
  • 1,000,000 tokens / day
  → For 6 students scanned every 5 seconds: 6 × 12 = 72 RPM worst case,
    so the module defaults to 10-second scan intervals per student
    (6 × 6 = 36 RPM — safely within limits).

Setup:
  1. Get free API key from https://aistudio.google.com/app/apikey
  2. Set environment variable:
         export GEMINI_API_KEY="your_key_here"
     OR pass api_key= to GeminiInspector.__init__()

Install:
    pip install google-generativeai pillow

Output format
-------------
Returns a list of detections compatible with ObjectDetector output:
    [{
        class:      "earphone",
        confidence: 0.85,
        bbox:       None,          # Gemini gives no pixel bbox — set to None
        center:     None,
        severity:   "CRITICAL",
        student_id: "S0",
        source:     "gemini",      # distinguishes from YOLO detections
        description: "Small wireless earpiece visible in left ear"
    }, ...]

Integration
-----------
GeminiInspector runs on a separate scan cycle (every N seconds per student).
BehaviorAggregator calls it in process() and merges its output with YOLO detections.
"""

import os
import io
import time
import base64
import threading
from typing import Optional

import numpy as np

try:
    import google.generativeai as genai
    from PIL import Image as PILImage
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False
    print("[Sentinel] Gemini inspector disabled. "
          "Install: pip install google-generativeai pillow")

# ---------------------------------------------------------------------------
# Severity map for items Gemini may detect
# ---------------------------------------------------------------------------
_SEVERITY_MAP: dict = {
    "earpiece":         "CRITICAL",
    "earphone":         "CRITICAL",
    "earbud":           "CRITICAL",
    "airpod":           "CRITICAL",
    "wireless earphone":"CRITICAL",
    "bluetooth earphone":"CRITICAL",
    "cell phone":       "CRITICAL",
    "mobile phone":     "CRITICAL",
    "smartphone":       "CRITICAL",
    "phone":            "CRITICAL",
    "cheat sheet":      "CRITICAL",
    "hidden note":      "CRITICAL",
    "note paper":       "HIGH",
    "handwritten note": "CRITICAL",
    "laptop":           "HIGH",
    "tablet":           "HIGH",
    "smartwatch":       "HIGH",
    "watch":            "HIGH",
    "book":             "MEDIUM",
    "textbook":         "MEDIUM",
    "notebook":         "MEDIUM",
    "paper":            "MEDIUM",
    "calculator":       "MEDIUM",
    "headphones":       "HIGH",
    "bluetooth headset":"HIGH",
}

# Canonical label normalization
_LABEL_NORM: dict = {
    "mobile phone": "cell phone",
    "smartphone":   "cell phone",
    "phone":        "cell phone",
    "airpod":       "earphone",
    "earbud":       "earphone",
    "wireless earphone": "earphone",
    "bluetooth earphone": "earphone",
    "textbook":     "book",
    "notebook":     "book",
    "cheat sheet":  "cheat_sheet",
    "hidden note":  "cheat_sheet",
    "note paper":   "cheat_sheet",
    "handwritten note": "cheat_sheet",
    "calculator":   "calculator",
    "smartwatch":   "smartwatch",
    "watch":        "smartwatch",
    "bluetooth headset": "headphones",
}

# Gemini prompt — structured to get JSON back reliably
_SYSTEM_PROMPT = """You are a strict exam-room proctor AI. 
Analyze the image of a student at their desk and identify ANY of these exam violations:

CRITICAL violations (must flag immediately):
- Cell phone / mobile phone (even if partially hidden, face down, or in pocket)
- Earpiece / earphone / AirPod / bluetooth earphone (in or near ear)
- Cheat sheet / hidden notes / handwritten notes (folded paper, paper under book, etc.)

HIGH violations:
- Laptop or tablet on desk
- Smartwatch or digital watch
- Headphones / bluetooth headset
- Backpack open on desk

MEDIUM violations:
- Open book or textbook
- Physical notes or notebook
- Calculator (if not permitted)

Respond ONLY with a JSON array (no markdown, no explanation):
[
  {
    "item": "<exact item name>",
    "confidence": <0.0 to 1.0>,
    "location": "<brief location description, e.g. 'right ear', 'under textbook', 'left hand'>",
    "severity": "<CRITICAL|HIGH|MEDIUM|LOW>",
    "evidence": "<one sentence describing what you see>"
  }
]

If nothing suspicious is found, respond with exactly: []
Be STRICT. Better to flag a false positive than miss real cheating.
"""


class GeminiInspector:
    """
    Periodic Gemini-powered deep scan of student desk regions.
    Thread-safe: scan requests are queued and results are buffered.
    """

    def __init__(
        self,
        api_key:         str   = None,
        scan_interval:   float = 10.0,   # seconds between scans per student
        model_name:      str   = "gemini-1.5-flash",
        max_image_size:  int   = 512,    # resize crop to this max dimension before sending
    ):
        """
        Parameters
        ----------
        api_key        : Gemini API key (or set GEMINI_API_KEY env var)
        scan_interval  : seconds between scans per student; 10s = safe for free tier
        model_name     : Gemini model to use (gemini-1.5-flash recommended)
        max_image_size : max pixel dimension of crop sent to Gemini (saves tokens)
        """
        self.scan_interval  = scan_interval
        self.max_image_size = max_image_size
        self._enabled       = False

        if not _GEMINI_AVAILABLE:
            return

        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            print("[Sentinel] GeminiInspector: no API key found. "
                  "Set GEMINI_API_KEY env var or pass api_key=. "
                  "Deep scanning disabled.")
            return

        genai.configure(api_key=key)
        self._model   = genai.GenerativeModel(model_name)
        self._enabled = True
        print(f"[Sentinel] GeminiInspector: enabled ({model_name}, "
              f"scan every {scan_interval}s per student)")

        # Per-student scan state (thread-safe via lock)
        self._last_scan:    dict = {}    # sid → timestamp of last scan
        self._scan_results: dict = {}    # sid → list of detection dicts
        self._lock          = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    def scan_student(self, student_id: str, frame_rgb: np.ndarray,
                     bbox: tuple = None) -> list:
        """
        Scan a student region synchronously if the scan interval has elapsed.
        Returns cached results otherwise (zero cost on most frames).

        Parameters
        ----------
        student_id : str
        frame_rgb  : full frame numpy array
        bbox       : (x, y, w, h) of student region. If None, uses full frame.

        Returns
        -------
        list of detection dicts (empty if no violations found or not yet scanned)
        """
        if not self._enabled:
            return []

        now = time.time()
        with self._lock:
            last = self._last_scan.get(student_id, 0)
            if now - last < self.scan_interval:
                return self._scan_results.get(student_id, [])

        # Perform scan (outside lock to avoid blocking)
        crop = self._crop(frame_rgb, bbox)
        results = self._call_gemini(crop, student_id)

        with self._lock:
            self._last_scan[student_id]    = now
            self._scan_results[student_id] = results

        return results

    def scan_student_async(self, student_id: str, frame_rgb: np.ndarray,
                           bbox: tuple = None) -> None:
        """
        Non-blocking version: fires a background thread for the Gemini call.
        Results are stored in the buffer and available on next scan_student() call.
        Use this in real-time video loops to avoid blocking the main thread.
        """
        if not self._enabled:
            return

        now = time.time()
        with self._lock:
            last = self._last_scan.get(student_id, 0)
            if now - last < self.scan_interval:
                return   # not due yet
            self._last_scan[student_id] = now   # mark as in-progress

        crop = self._crop(frame_rgb, bbox)
        t = threading.Thread(
            target=self._async_scan_worker,
            args=(student_id, crop),
            daemon=True,
        )
        t.start()

    def get_cached_results(self, student_id: str) -> list:
        """Return buffered Gemini results for a student (non-blocking)."""
        with self._lock:
            return self._scan_results.get(student_id, [])

    def get_all_cached(self) -> dict:
        """Return all buffered results: sid → list[detection]."""
        with self._lock:
            return dict(self._scan_results)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _async_scan_worker(self, student_id: str, crop: np.ndarray) -> None:
        results = self._call_gemini(crop, student_id)
        with self._lock:
            self._scan_results[student_id] = results

    def _call_gemini(self, crop: np.ndarray, student_id: str) -> list:
        """Send crop to Gemini and parse response into detection dicts."""
        try:
            pil_img  = PILImage.fromarray(crop)
            img_bytes = io.BytesIO()
            pil_img.save(img_bytes, format="JPEG", quality=85)
            img_bytes.seek(0)

            response = self._model.generate_content(
                [_SYSTEM_PROMPT, PILImage.open(img_bytes)],
                generation_config=genai.GenerationConfig(
                    temperature=0.0,          # deterministic output
                    max_output_tokens=512,
                ),
            )
            raw_text = response.text.strip()
            return self._parse_response(raw_text, student_id)

        except Exception as e:
            print(f"[Sentinel] Gemini scan error for {student_id}: {e}")
            return []

    def _parse_response(self, raw_text: str, student_id: str) -> list:
        """Parse Gemini's JSON array response into detection dicts."""
        import json

        if not raw_text or raw_text == "[]":
            return []

        # Strip markdown fences if model adds them
        text = raw_text
        for fence in ("```json", "```JSON", "```"):
            text = text.replace(fence, "")
        text = text.strip()

        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            # Attempt to extract a JSON array with a simple search
            start = text.find("[")
            end   = text.rfind("]")
            if start != -1 and end != -1:
                try:
                    items = json.loads(text[start:end + 1])
                except Exception:
                    return []
            else:
                return []

        if not isinstance(items, list):
            return []

        detections = []
        for item in items:
            if not isinstance(item, dict):
                continue

            raw_name  = str(item.get("item", "")).lower().strip()
            if not raw_name:
                continue

            # Normalize label
            label = _LABEL_NORM.get(raw_name, raw_name)
            if len(label) > 30:
                label = label[:30]

            confidence = float(item.get("confidence", 0.7))
            if confidence < 0.45:     # ignore Gemini low-confidence guesses
                continue

            severity = str(item.get("severity", "LOW")).upper()
            if severity not in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
                severity = _SEVERITY_MAP.get(raw_name, "LOW")

            detections.append({
                "class":       label,
                "raw_class":   raw_name,
                "confidence":  round(confidence, 2),
                "bbox":        None,
                "center":      None,
                "severity":    severity,
                "student_id":  student_id,
                "under_table": False,
                "confirmed":   True,    # Gemini is high-level; treat as confirmed
                "source":      "gemini",
                "location":    str(item.get("location", "")),
                "description": str(item.get("evidence", "")),
            })

        return detections

    def _crop(self, frame_rgb: np.ndarray, bbox: tuple) -> np.ndarray:
        """Crop student region and resize to max_image_size for efficiency."""
        h, w = frame_rgb.shape[:2]

        if bbox:
            x, y, bw, bh = bbox
            x1 = max(0, x - 20)
            y1 = max(0, y - 20)
            x2 = min(w, x + bw + 20)
            y2 = min(h, y + bh + 20)
            crop = frame_rgb[y1:y2, x1:x2]
        else:
            crop = frame_rgb

        # Resize to max dimension
        ch, cw = crop.shape[:2]
        if max(ch, cw) > self.max_image_size:
            scale  = self.max_image_size / max(ch, cw)
            new_w  = max(1, int(cw * scale))
            new_h  = max(1, int(ch * scale))
            # Resize using PIL (available since we imported it above)
            pil    = PILImage.fromarray(crop)
            pil    = pil.resize((new_w, new_h), PILImage.LANCZOS)
            crop   = np.array(pil)

        return crop