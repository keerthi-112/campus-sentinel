"""
pipeline.py — Wires capture → detect → track → zone → features into one
continuous loop.

This is the main entry point for running the vision track.  It prints
events to stdout and (later) will POST incident candidates to the backend.
"""

from __future__ import annotations

import json
import time
import cv2

from .capture import CameraConfig, VideoCapture
from .detect import PersonDetector, Detection
from .tracker import CentroidTracker, TrackedPerson
from .zones import ZoneConfig, ZoneMonitor, ZoneIncidentCandidate
from .features import compute_features


def run_pipeline(
    camera: CameraConfig,
    zones: list[ZoneConfig],
    *,
    model_name: str = "yolo11n.pt",
    confidence: float = 0.40,
    device: str = "cpu",
    show_window: bool = True,
) -> None:
    """
    Main vision loop.

    Parameters
    ----------
    camera : CameraConfig
        Which video source to read.
    zones : list[ZoneConfig]
        Restricted zones drawn on this camera's view.
    model_name : str
        YOLO checkpoint to use.
    confidence : float
        Detection confidence threshold.
    device : str
        Inference device (``"cpu"`` / ``"cuda"`` / ``"mps"``).
    show_window : bool
        Whether to display an annotated OpenCV window (set ``False`` for
        headless / Docker).
    """
    detector = PersonDetector(model_name=model_name,
                              confidence_threshold=confidence,
                              device=device)
    tracker = CentroidTracker()
    zone_monitor = ZoneMonitor(zones)

    prev_centers: dict[int, tuple[int, int]] = {}

    with VideoCapture(camera) as cap:
        for info in cap.frames():
            frame = info["frame"]
            cam_name = info["camera_name"]

            # --- (3) detect people ---
            detections = detector.detect(frame)

            # --- (4) track ---
            tracks = tracker.update(detections)

            # --- (4) zone / crowd logic ---
            candidates = zone_monitor.update(tracks, cam_name)

            # --- (5) feature vectors (one per zone per frame) ---
            for zone in zones:
                # Find the longest current dwell & highest conf for this zone
                dwell = 0.0
                max_conf = 0.0
                for pid, person in tracks.items():
                    if zone_monitor._point_in_zone(person.center, zone):
                        entry = zone_monitor._dwells[zone.zone_name].get(pid)
                        if entry:
                            elapsed = time.time() - entry.enter_time
                            dwell = max(dwell, elapsed)
                            max_conf = max(max_conf, entry.max_confidence)

                fv = compute_features(
                    detections=detections,
                    tracks=tracks,
                    zone=zone,
                    camera_name=cam_name,
                    frame_shape=(frame.shape[0], frame.shape[1]),
                    dwell_seconds=dwell,
                    max_confidence=max_conf,
                    prev_centers=prev_centers,
                )

                # Print feature vectors periodically (every ~1 s)
                if info["frame_idx"] % 30 == 0:
                    print(f"[features] {json.dumps(fv)}")

            # --- (5) incident candidates ---
            for c in candidates:
                _print_candidate(c)

            # --- draw + display ---
            if show_window:
                _annotate(frame, detections, tracks, zone_monitor)
                cv2.imshow(f"Sentinel — {cam_name}", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[pipeline] quit requested.")
                    break

            # remember centres for motion score
            prev_centers = {pid: p.center for pid, p in tracks.items()}

    if show_window:
        cv2.destroyAllWindows()
    print("[pipeline] done.")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _annotate(
    frame,
    detections: list[Detection],
    tracks: dict[int, TrackedPerson],
    zone_monitor: ZoneMonitor,
) -> None:
    """Draw bounding boxes, IDs, and zone polygons onto the frame."""
    zone_monitor.draw_zones(frame)

    for pid, person in tracks.items():
        x1, y1, x2, y2 = person.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"ID {pid}  {person.confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def _print_candidate(c: ZoneIncidentCandidate) -> None:
    print(
        f"\n{'='*60}\n"
        f"[INCIDENT CANDIDATE]\n"
        f"  Person ID   : {c.person_id}\n"
        f"  Zone        : {c.zone_name}\n"
        f"  Camera      : {c.camera_name}\n"
        f"  Dwell       : {c.dwell_seconds:.1f}s\n"
        f"  Confidence  : {c.max_confidence:.2f}\n"
        f"  Timestamp   : {c.timestamp}\n"
        f"{'='*60}\n"
    )
