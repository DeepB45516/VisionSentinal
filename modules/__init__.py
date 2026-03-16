# Sentinel AI — Module Package v3
#
# New in v3:
#   • emotion_detector.py     — blendshape-based emotion analysis (no extra model)
#   • behavior_aggregator.py  — single-call driver for all modules
#
# Quick start (recommended):
#   from modules import BehaviorAggregator
#   agg    = BehaviorAggregator()
#   result = agg.process(frame_rgb)
#
# Or use modules individually:
#   from modules import FaceIntelligence, ObjectDetector, ...

from .face_intelligence     import FaceIntelligence
from .emotion_detector      import EmotionDetector
from .object_detection      import ObjectDetector
from .gemini_inspector      import GeminiInspector
from .pose_tracking         import PoseTracker
from .wrist_velocity        import WristVelocityTracker
from .coordination_detector import CoordinationDetector
from .risk_scoring          import RiskScorer
from .temporal_filter       import TemporalFilter
from .behavior_aggregator   import BehaviorAggregator

__all__ = [
    "FaceIntelligence",
    "EmotionDetector",
    "ObjectDetector",
    "GeminiInspector",
    "PoseTracker",
    "WristVelocityTracker",
    "CoordinationDetector",
    "RiskScorer",
    "TemporalFilter",
    "BehaviorAggregator",
]