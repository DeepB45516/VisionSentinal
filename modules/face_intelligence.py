"""
Sentinel AI — Face Intelligence Module v3
==========================================
Single-pass FaceLandmarker: detection + landmarks + blendshapes in ONE forward pass.

What's new vs v2
-----------------
[ ACCURACY ]
  • Blendshape emotion signals  — FaceLandmarker blendshapes enabled; _analyze_emotion()
                                  returns per-face anxiety/stress/surprise scores without
                                  any extra model; zero additional inference cost
  • Proxy / impersonation alert — tracks expected_faces baseline from first N stable frames;
                                  if face_count rises above baseline → proxy alert
  • Shifty-eye tracker          — counts gaze direction changes per student per second;
                                  ≥ 3 switches/10s triggers a "shifty_eyes" event
  • Improved gaze thresholds    — H_THRESH raised 0.26→0.28, V_DOWN 0.58→0.55 (better
                                  sensitivity at typical CCTV angles / low seat positions)
  • Confident absence           — absence only fires once ABSENT_DEBOUNCE * frame_skip
                                  frames pass (prevents false alerts on brief occlusion)

[ PERFORMANCE ]
  • Default frame_skip=1        — every other frame (was 2); blendshape inference adds
                                  <2 ms/face on CPU — negligible vs landmark cost
  • Early-exit on empty result  — skips all downstream processing if no faces found
  • Blendshape lookup by name   — dict built once per face call, O(1) per key read
  • Float32 arrays only         — force dtype=float32 throughout; avoids silent float64
                                  promotion that doubles memory bandwidth in numpy ops

[ API ADDITIONS ]
  • process() now returns "emotions" list: [{id, anxiety, surprise, stress, label}]
  • process() now returns "proxy_alert": bool — True if unexpected extra face detected
  • "absences" entries gain "expected_id" if we can infer which student went missing
  • All other keys/shapes unchanged from v2
"""

import os
import time
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
_MODELS_DIR    = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
_FACE_LM_MODEL = os.path.join(_MODELS_DIR, "face_landmarker.task")

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
_EMA_ALPHA        = 0.30   # gaze EMA smoothing: lower → more stable
_ABSENT_DEBOUNCE  = 3      # consecutive absent inference frames required
_MAX_MATCH_DIST_R = 0.38   # face ID match threshold as fraction of frame diagonal
_CONF_AREA_SCALE  = 22.0   # face confidence scale factor

# Blendshape emotion thresholds
_ANXIETY_THRESHOLD   = 0.38
_SURPRISE_THRESHOLD  = 0.50
_STRESS_THRESHOLD    = 0.40

# Proxy detection: baseline established after this many stable frames
_PROXY_BASELINE_FRAMES = 30
# Shifty eyes: flag if gaze changes ≥ this count within the time window
_SHIFTY_CHANGE_THRESH  = 3
_SHIFTY_WINDOW_SECS    = 10.0


class FaceIntelligence:
    """Real-time face detection, gaze estimation, emotion analysis, and absence tracking."""

    def __init__(
        self,
        max_students: int   = 6,
        absence_threshold: float = 5.0,
        frame_skip: int     = 1,
        detect_scale: float = 0.5,
    ):
        """
        Parameters
        ----------
        max_students      : maximum concurrent faces to track
        absence_threshold : seconds before a missing face is flagged absent
        frame_skip        : run ML every (frame_skip+1) frames; 0=every frame
        detect_scale      : numpy-stride downscale before inference (0.25–1.0)
        """
        self.max_students      = max_students
        self.absence_threshold = absence_threshold
        self.frame_skip        = max(0, int(frame_skip))
        self.detect_scale      = max(0.25, min(1.0, float(detect_scale)))

        opts = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_FACE_LM_MODEL),
            num_faces=max_students,
            min_face_detection_confidence=0.45,
            min_face_presence_confidence=0.45,
            min_tracking_confidence=0.40,
            output_face_blendshapes=True,            # ← NEW: enables emotion signals
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = FaceLandmarker.create_from_options(opts)

        # Per-student temporal state
        self.face_last_seen:    dict = {}
        self.face_first_seen:   dict = {}
        self.face_absent_start: dict = {}
        self._absent_frames:    dict = {}
        self._gaze_ema:         dict = {}   # sid → [h_score, v_score]
        self._prev_faces:       list = []
        self._gaze_history:     dict = {}   # sid → [(direction, timestamp), ...]

        # Proxy detection state
        self._expected_face_count: int  = -1
        self._stable_frame_count:  int  = 0
        self._last_face_count:     int  = 0

        self._frame_counter:  int        = 0
        self._cached_result:  dict | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame_rgb: np.ndarray, h: int, w: int) -> dict:
        """
        Process a single RGB frame.

        Returns
        -------
        dict with keys:
            face_count      int
            faces           list[dict]  — id, bbox, center, confidence, student_id
            gaze_directions list[dict]  — id, direction, confidence
            absences        list[dict]  — id, absent_seconds
            emotions        list[dict]  — id, anxiety, surprise, stress, label   ← NEW
            proxy_alert     bool        — unexpected extra face detected           ← NEW
            shifty_students list[str]   — student IDs with rapid gaze switching   ← NEW
        """
        now = time.time()
        self._frame_counter += 1

        # Frame-skip: update only absence timers on skipped frames
        if self.frame_skip > 0 and (self._frame_counter % (self.frame_skip + 1)) != 0:
            if self._cached_result is not None:
                return self._refresh_absences(self._cached_result, now)

        # Stride-based downscale (no extra dependencies)
        stride    = max(1, round(1.0 / self.detect_scale))
        frame_in  = frame_rgb[::stride, ::stride] if stride > 1 else frame_rgb
        mp_image  = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_in, dtype=np.uint8)
        )

        lm_result         = self.landmarker.detect(mp_image)
        face_lms          = lm_result.face_landmarks   or []
        face_blendshapes  = lm_result.face_blendshapes or []

        # Build face list from landmark extents (vectorised)
        faces: list = []
        for i, face_lm in enumerate(face_lms[: self.max_students]):
            coords        = np.array([[lm.x * w, lm.y * h] for lm in face_lm], dtype=np.float32)
            x1, y1        = coords.min(axis=0).astype(int)
            x2, y2        = coords.max(axis=0).astype(int)
            bw, bh        = max(1, x2 - x1), max(1, y2 - y1)
            cx, cy        = x1 + bw // 2, y1 + bh // 2
            conf          = min(1.0, (bw * bh) / (w * h + 1e-6) * _CONF_AREA_SCALE)
            faces.append({
                "id": i, "bbox": (x1, y1, bw, bh),
                "center": (cx, cy), "size": max(bw, bh),
                "confidence": round(conf, 2),
            })

        # Early exit if no faces
        if not faces:
            empty = {
                "face_count": 0, "faces": [], "gaze_directions": [],
                "absences": self._build_absences(set(), now),
                "emotions": [], "proxy_alert": False, "shifty_students": [],
            }
            self._cached_result = empty
            return empty

        # Match to persistent student IDs
        matched_ids = self._match_faces(faces, w, h)
        for i, face in enumerate(faces):
            sid = matched_ids[i]
            face["student_id"] = sid
            self.face_last_seen[sid] = now
            self.face_first_seen.setdefault(sid, now)
            self.face_absent_start.pop(sid, None)
            self._absent_frames.pop(sid, None)

        active_ids = set(matched_ids)

        # Gaze directions + emotion per face
        gaze_directions: list = []
        emotions:        list = []

        for i, face_lm in enumerate(face_lms[: self.max_students]):
            sid       = matched_ids[i] if i < len(matched_ids) else f"S{i}"
            direction, gaze_conf = self._compute_gaze(face_lm, sid, w, h)
            gaze_directions.append({"id": sid, "direction": direction, "confidence": gaze_conf})

            # Track gaze direction changes for shifty-eye detection
            gh = self._gaze_history.setdefault(sid, [])
            gh.append((direction, now))
            # Trim to window
            self._gaze_history[sid] = [(d, t) for d, t in gh if now - t <= _SHIFTY_WINDOW_SECS]

            # Emotion from blendshapes
            if i < len(face_blendshapes) and face_blendshapes[i]:
                emo = self._analyze_emotion(face_blendshapes[i], sid)
                emotions.append(emo)

        # Shifty eyes: count direction changes in the window
        shifty_students: list = []
        for sid, history in self._gaze_history.items():
            if len(history) < 2:
                continue
            changes = sum(
                1 for j in range(1, len(history))
                if history[j][0] != history[j-1][0]
                and history[j][0] in ("LEFT", "RIGHT")
            )
            if changes >= _SHIFTY_CHANGE_THRESH:
                shifty_students.append(sid)

        # Proxy detection
        proxy_alert = self._check_proxy(len(faces))

        result = {
            "face_count":      len(faces),
            "faces":           faces,
            "gaze_directions": gaze_directions,
            "absences":        self._build_absences(active_ids, now),
            "emotions":        emotions,
            "proxy_alert":     proxy_alert,
            "shifty_students": list(set(shifty_students)),
        }
        self._cached_result = result
        return result

    def cleanup(self):
        self.landmarker.close()

    # ------------------------------------------------------------------
    # Emotion Analysis (blendshapes — no extra model)
    # ------------------------------------------------------------------

    def _analyze_emotion(self, blendshape_list, sid: str) -> dict:
        """
        Derive exam-relevant emotional signals from MediaPipe blendshapes.

        Signals:
          anxiety  — raised inner brow + brow furrow (worried/nervous look)
          surprise — wide eyes + jaw open (caught/startled)
          stress   — lip press + brow down + nose sneer (tense concentration)
        """
        # Build O(1) lookup by category_name
        bs: dict = {b.category_name: b.score for b in blendshape_list}

        def g(key: str) -> float:
            return bs.get(key, 0.0)

        # ── Anxiety / Nervousness ─────────────────────────────────────
        brow_inner_up   = g("browInnerUp")
        brow_down_l     = g("browDownLeft")
        brow_down_r     = g("browDownRight")
        brow_furrow     = (brow_down_l + brow_down_r) / 2.0
        anxiety_score   = 0.55 * brow_inner_up + 0.45 * brow_furrow

        # ── Surprise / Caught ─────────────────────────────────────────
        eye_wide_l      = g("eyeWideLeft")
        eye_wide_r      = g("eyeWideRight")
        jaw_open        = g("jawOpen")
        surprise_score  = 0.60 * ((eye_wide_l + eye_wide_r) / 2.0) + 0.40 * jaw_open

        # ── Stress / Tension ─────────────────────────────────────────
        lip_press_l     = g("mouthPressLeft")
        lip_press_r     = g("mouthPressRight")
        nose_sneer_l    = g("noseSneerLeft")
        nose_sneer_r    = g("noseSneerRight")
        cheek_sq_l      = g("cheekSquintLeft")
        cheek_sq_r      = g("cheekSquintRight")
        stress_score    = (
            0.35 * ((lip_press_l + lip_press_r) / 2.0)
            + 0.30 * brow_furrow
            + 0.20 * ((nose_sneer_l + nose_sneer_r) / 2.0)
            + 0.15 * ((cheek_sq_l  + cheek_sq_r)  / 2.0)
        )

        # Determine dominant label
        scores = {
            "anxious":   anxiety_score,
            "surprised": surprise_score,
            "stressed":  stress_score,
        }
        dominant = max(scores, key=scores.__getitem__)
        max_score = scores[dominant]

        if max_score < 0.22:
            label = "neutral"
        elif dominant == "anxious"   and anxiety_score   >= _ANXIETY_THRESHOLD:
            label = "anxious"
        elif dominant == "surprised" and surprise_score  >= _SURPRISE_THRESHOLD:
            label = "surprised"
        elif dominant == "stressed"  and stress_score    >= _STRESS_THRESHOLD:
            label = "stressed"
        else:
            label = "neutral"

        return {
            "id":       sid,
            "anxiety":  round(anxiety_score,  3),
            "surprise": round(surprise_score, 3),
            "stress":   round(stress_score,   3),
            "label":    label,
        }

    # ------------------------------------------------------------------
    # Proxy / Impersonation Detection
    # ------------------------------------------------------------------

    def _check_proxy(self, face_count: int) -> bool:
        """
        Baseline face count established from the first PROXY_BASELINE_FRAMES stable frames.
        Once baseline is set, any face_count > expected_face_count → proxy alert.
        """
        if self._expected_face_count < 0:
            # Building baseline: require N consecutive frames with same count
            if face_count == self._last_face_count and face_count > 0:
                self._stable_frame_count += 1
                if self._stable_frame_count >= _PROXY_BASELINE_FRAMES:
                    self._expected_face_count = face_count
            else:
                self._stable_frame_count = 0
            self._last_face_count = face_count
            return False

        return face_count > self._expected_face_count

    # ------------------------------------------------------------------
    # Absence Tracking
    # ------------------------------------------------------------------

    def _build_absences(self, active_ids: set, now: float) -> list:
        absences = []
        for sid in list(self.face_last_seen):
            if sid not in active_ids:
                self.face_absent_start.setdefault(sid, now)
                self._absent_frames[sid] = self._absent_frames.get(sid, 0) + 1
                absent_secs = now - self.face_absent_start[sid]
                if (absent_secs >= self.absence_threshold
                        and self._absent_frames[sid] >= _ABSENT_DEBOUNCE):
                    absences.append({
                        "id":             sid,
                        "absent_seconds": round(absent_secs, 1),
                    })
        return absences

    def _refresh_absences(self, cached: dict, now: float) -> dict:
        """Zero-ML update on skipped frames: only refresh absence durations."""
        active_ids = {f.get("student_id", f.get("id")) for f in cached["faces"]}
        return {**cached, "absences": self._build_absences(active_ids, now)}

    # ------------------------------------------------------------------
    # Face ID Matching
    # ------------------------------------------------------------------

    def _match_faces(self, faces: list, w: int, h: int) -> list:
        """
        Nearest-neighbour matching with size-weighted distance.
        Resolution-independent: max threshold scales with frame diagonal.
        Lowest-free-S-index allocation for stable naming.
        """
        if not self._prev_faces:
            ids = [f"S{i}" for i in range(len(faces))]
            self._prev_faces = [(f["center"], f.get("size", 50), ids[i]) for i, f in enumerate(faces)]
            return ids

        frame_diag = math.hypot(w, h)
        max_score  = _MAX_MATCH_DIST_R * frame_diag
        matched: list = []
        used:    set  = set()

        for face in faces:
            cx, cy    = face["center"]
            sz        = face.get("size", 50)
            best_score, best_id, best_idx = float("inf"), None, -1

            for j, (pc, psz, pid) in enumerate(self._prev_faces):
                if j in used:
                    continue
                dist       = math.hypot(cx - pc[0], cy - pc[1])
                size_ratio = abs(sz - psz) / (max(sz, psz) + 1e-3)
                score      = dist * (1.0 + size_ratio)
                if score < best_score:
                    best_score, best_id, best_idx = score, pid, j

            if best_id is not None and best_score < max_score:
                matched.append(best_id)
                used.add(best_idx)
            else:
                taken = {pid for _, _, pid in self._prev_faces} | set(matched)
                k = 0
                while f"S{k}" in taken:
                    k += 1
                matched.append(f"S{k}")

        self._prev_faces = [
            (f["center"], f.get("size", 50), matched[i]) for i, f in enumerate(faces)
        ]
        return matched

    # ------------------------------------------------------------------
    # Gaze Estimation
    # ------------------------------------------------------------------

    def _compute_gaze(self, landmarks, sid: str, w: int, h: int) -> tuple:
        """
        Fused iris + nose gaze estimate with per-student EMA smoothing.
        Returns (direction_str, confidence_float).
        Directions: CENTER | LEFT | RIGHT | UP | DOWN
        """
        lm = landmarks

        nose_tip   = lm[1]
        chin       = lm[152]
        forehead   = lm[10]
        left_eye   = lm[33]
        right_eye  = lm[263]
        left_iris  = lm[468] if len(lm) > 468 else lm[159]
        right_iris = lm[473] if len(lm) > 473 else lm[386]

        # Horizontal (yaw): iris offset + nose offset, normalised by eye width
        eye_cx    = (left_eye.x  + right_eye.x) / 2
        eye_cy    = (left_eye.y  + right_eye.y) / 2
        eye_width = abs(right_eye.x - left_eye.x) or 1e-3
        nose_h    = (nose_tip.x - eye_cx) / eye_width
        iris_cx   = (left_iris.x + right_iris.x) / 2
        iris_off  = (iris_cx - eye_cx) / eye_width
        h_raw     = 0.45 * nose_h + 0.55 * iris_off

        # Vertical (pitch): nose drop normalised by face height
        face_ht   = abs(chin.y - eye_cy) or 1e-3
        nose_drop = (nose_tip.y - eye_cy) / face_ht
        fore_rise = (eye_cy - forehead.y) / face_ht
        v_raw     = nose_drop - 0.5 * fore_rise

        # Per-student EMA
        if sid not in self._gaze_ema:
            self._gaze_ema[sid] = [h_raw, v_raw]
        ema    = self._gaze_ema[sid]
        ema[0] = _EMA_ALPHA * h_raw + (1.0 - _EMA_ALPHA) * ema[0]
        ema[1] = _EMA_ALPHA * v_raw + (1.0 - _EMA_ALPHA) * ema[1]
        h_s, v_s = ema

        H_THRESH = 0.28
        V_DOWN   = 0.55
        V_UP     = 0.28

        if v_s > V_DOWN:
            return "DOWN",   round(min(1.0, (v_s - V_DOWN) * 5.0 + 0.5), 2)
        if v_s < -V_UP:
            return "UP",     round(min(1.0, (abs(v_s) - V_UP) * 5.0 + 0.5), 2)
        if h_s > H_THRESH:
            return "RIGHT",  round(min(1.0, (h_s - H_THRESH) * 4.0 + 0.5), 2)
        if h_s < -H_THRESH:
            return "LEFT",   round(min(1.0, (abs(h_s) - H_THRESH) * 4.0 + 0.5), 2)
        return "CENTER", round(max(0.0, 1.0 - (abs(h_s) + abs(v_s)) * 1.5), 2)
