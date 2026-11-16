"""
pipeline.py — Wires capture → detect → track → zone/crowd → features into one
continuous loop.

Main entry point for running the vision track pipeline.
Emits schema-compliant event feature vectors for federated learning and
incident candidates for RAG & backend.
"""

from __future__ import annotations

import json
import time
from typing import Callable
from pathlib import Path
import urllib.request
import urllib.error
import cv2

from .capture import CameraConfig, VideoCapture
from .detect import PersonDetector, Detection
from .tracker import CentroidTracker, TrackedPerson
from .zones import ZoneConfig, ZoneMonitor, ZoneIncidentCandidate
from .crowd import CrowdConfig, CrowdMonitor, CrowdIncidentCandidate
from .features import compute_features, export_feature_vector


def post_incident_to_rag(
    incident_dict: dict,
    rag_url: str = "http://127.0.0.1:8000/reports",
    timeout: float = 5.0,
) -> dict | None:
    """
    Deliver an incident record to the RAG service endpoint (e.g. POST /reports).
    The RAG service will queue the incident and generate a policy-cited report.

    Uses the Python standard library (urllib.request) so no extra dependencies
    are required by the vision track.
    """
    data = json.dumps(incident_dict).encode("utf-8")
    req = urllib.request.Request(
        rag_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            resp_body = response.read().decode("utf-8")
            res_json = json.loads(resp_body)
            print(
                f"[pipeline -> rag] Incident '{incident_dict.get('incident_ref')}' queued successfully "
                f"(HTTP {response.status}, job_id: {res_json.get('job_id')}, status: {res_json.get('status')})"
            )
            return res_json
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        print(f"[pipeline -> rag] HTTP {e.code} error from RAG endpoint ({rag_url}): {err_body}")
        return None
    except urllib.error.URLError as e:
        print(f"[pipeline -> rag] Warning: Could not reach RAG endpoint ({rag_url}): {e.reason}")
        return None


def run_pipeline(
    camera: CameraConfig,
    zones: list[ZoneConfig],
    *,
    model_name: str = "yolo11n.pt",
    confidence: float = 0.40,
    device: str = "cpu",
    crowd_config: CrowdConfig | None = None,
    show_window: bool = True,
    on_incident: Callable[[dict], None] | None = None,
    rag_url: str | None = None,
    export_features_path: str | Path | None = None,
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
        YOLO checkpoint to use (e.g. "yolo11n.pt", "yolov8n.pt", or "yolo26n.pt").
    confidence : float
        Detection confidence threshold.
    device : str
        Inference device ("cpu" / "cuda" / "mps").
    crowd_config : CrowdConfig | None
        Optional crowd threshold parameters.
    show_window : bool
        Whether to display an annotated OpenCV window.
    on_incident : Callable[[dict], None] | None
        Optional callback invoked with a schema-compliant incident dict
        whenever a zone entry or crowd incident is confirmed.
    rag_url : str | None
        Optional URL to the RAG endpoint (e.g. "http://127.0.0.1:8000/reports").
        When set, automatically queues confirmed incidents for RAG report generation.
    export_features_path : str | Path | None
        Optional file path to persist feature vectors as JSON-Lines.
    """
    detector = PersonDetector(
        model_name=model_name,
        confidence_threshold=confidence,
        device=device,
    )
    tracker = CentroidTracker()
    zone_monitor = ZoneMonitor(zones)
    crowd_monitor = CrowdMonitor(crowd_config)

    prev_centers: dict[int, tuple[int, int]] = {}

    with VideoCapture(camera) as cap:
        for info in cap.frames():
            frame = info["frame"]
            cam_name = info["camera_name"]

            # --- (1) Detect people ---
            detections = detector.detect(frame)

            # --- (2) Track persons across frames ---
            tracks = tracker.update(detections)

            # --- (3) Zone persistence logic ---
            zone_candidates = zone_monitor.update(tracks, cam_name)

            # --- (4) Crowd persistence logic ---
            frame_area = frame.shape[0] * frame.shape[1]
            bbox_area = sum(d.area for d in detections)
            density = round(bbox_area / frame_area, 4) if frame_area else 0.0
            crowd_candidates = crowd_monitor.update(
                person_count=len(detections),
                density=density,
                camera_name=cam_name,
                zone_name=zones[0].zone_name if zones else "General Area",
            )

            # Combine all incident candidates
            all_candidates = list(zone_candidates) + list(crowd_candidates)

            # --- (5) Compute & optionally export feature vectors ---
            for zone in zones:
                dwell, max_conf = zone_monitor.get_zone_dwell_info(zone, tracks)

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

                if export_features_path:
                    export_feature_vector(fv, export_features_path)

                # Periodic terminal heartbeat (~1 per second)
                if info["frame_idx"] % 30 == 0:
                    print(f"[features] {json.dumps(fv)}")

            # --- (6) Process incidents ---
            for c in all_candidates:
                inc_dict = c.to_incident_dict()
                _print_candidate(c, inc_dict)
                if rag_url:
                    post_incident_to_rag(inc_dict, rag_url=rag_url)
                if on_incident:
                    on_incident(inc_dict)

            # --- (7) Display view ---
            if show_window:
                _annotate(frame, detections, tracks, zone_monitor)
                cv2.imshow(f"Sentinel — {cam_name}", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[pipeline] Quit requested by operator.")
                    break

            prev_centers = {pid: p.center for pid, p in tracks.items()}

    if show_window:
        cv2.destroyAllWindows()
    print("[pipeline] Stream processing complete.")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _annotate(
    frame,
    detections: list[Detection],
    tracks: dict[int, TrackedPerson],
    zone_monitor: ZoneMonitor,
) -> None:
    """Draw bounding boxes, track IDs, and zone boundaries onto the frame."""
    zone_monitor.draw_zones(frame)

    for pid, person in tracks.items():
        x1, y1, x2, y2 = person.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"ID {pid} ({person.confidence:.2f})"
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )


def _print_candidate(c, incident_dict: dict) -> None:
    print(
        f"\n{'='*60}\n"
        f"[INCIDENT CONFIRMED — {incident_dict['event_type'].upper()}]\n"
        f"  Ref         : {incident_dict['incident_ref']}\n"
        f"  Camera      : {incident_dict['camera_name']}\n"
        f"  Zone        : {incident_dict['zone_name']}\n"
        f"  Severity    : {incident_dict['severity'].upper()}\n"
        f"  Duration    : {incident_dict['duration_seconds']}s\n"
        f"  Confidence  : {incident_dict['confidence']}\n"
        f"  Timestamp   : {incident_dict['timestamp']}\n"
        f"  JSON Output : {json.dumps(incident_dict)}\n"
        f"{'='*60}\n"
    )
