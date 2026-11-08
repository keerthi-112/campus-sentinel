"""
Campus Sentinel — Vision & Zone Logic Track (Member 1).

Core modules:
- capture: unified video ingest (file, webcam, RTSP)
- detect: YOLO person detector (class 0)
- tracker: centroid tracking for identity persistence
- zones: polygon restricted zone & dwell timer
- crowd: crowd formation persistence logic
- features: event feature vector computation & export
- pipeline: end-to-end continuous vision pipeline
- benchmark: latency, throughput (FPS), and evaluation harness
"""

from .capture import CameraConfig, VideoCapture
from .detect import PersonDetector, Detection
from .tracker import CentroidTracker, TrackedPerson
from .zones import ZoneConfig, ZoneMonitor, ZoneIncidentCandidate
from .crowd import CrowdConfig, CrowdMonitor, CrowdIncidentCandidate
from .features import compute_features, export_feature_vector
from .pipeline import run_pipeline

__all__ = [
    "CameraConfig",
    "VideoCapture",
    "PersonDetector",
    "Detection",
    "CentroidTracker",
    "TrackedPerson",
    "ZoneConfig",
    "ZoneMonitor",
    "ZoneIncidentCandidate",
    "CrowdConfig",
    "CrowdMonitor",
    "CrowdIncidentCandidate",
    "compute_features",
    "export_feature_vector",
    "run_pipeline",
]
