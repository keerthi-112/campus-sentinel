# vision/

Owns: video/camera input, YOLO person detection, centroid tracking, restricted-zone monitoring, crowd formation detection, feature vector extraction, and incident dispatching.

- **Reads:** video file, webcam, or RTSP stream.
- **Produces (Contract 1):** `docs/schemas/event_feature_vector.json`-compliant records handed to `federated/` for model training and live decision.
- **Produces (Contract 2):** `docs/schemas/incident.json`-compliant records handed to `rag/` (for report generation) and `backend/` (for persistence).

---

## Architecture & Data Flow

```
Video Stream (capture.py)
       │
       ▼
Person Detection (detect.py - YOLO class 0)
       │
       ▼
Centroid Tracking (tracker.py - stable IDs across frames)
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
Restricted Zones (zones.py)        Crowd Detection (crowd.py)
- Point-in-polygon check           - Count & density threshold
- Dwell persistence check          - Persistence dwell check
- Risk tiering (Tier 1/2/3)        - Zone-aware context
       │                                 │
       ▼                                 ▼
   ZoneIncidentCandidate             CrowdIncidentCandidate
       │                                 │
       └──────────────┬──────────────────┘
                      │ .to_incident_dict()
                      ▼
         Incident Record (matches docs/schemas/incident.json & RAG Incident)
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
   RAG Service               Event Features (features.py)
   (POST /reports)           - Count, density, occupancy, motion score
   via post_incident_to_rag  - Matches docs/schemas/event_feature_vector.json
                             - Exportable to data/features.jsonl
```

---

## Setup

```bash
cd campus-sentinel
python3 -m venv .venv
source .venv/bin/activate
pip install -r vision/requirements.txt
```

---

## Running the Pipeline

### 1. Interactive Runner / Demo
```bash
# Webcam (default)
python -m vision.run_demo

# Video file
python -m vision.run_demo --source data/videos/clip.mp4 --type file

# Apple Silicon GPU acceleration
python -m vision.run_demo --device mps

# Stream incidents directly to RAG service for automatic report generation
python -m vision.run_demo --rag-url http://127.0.0.1:8000/reports

# Export features to JSONL for Federated Learning track
python -m vision.run_demo --export-features data/features.jsonl

# Configure zone tier (1 = Low, 2 = Medium, 3 = High)
python -m vision.run_demo --tier 3 --dwell 3.0
```
*Press **q** in the OpenCV window to exit.*

### 2. Performance & Benchmark Suite
Measures inference latency, FPS throughput, and tracking/zone overhead:
```bash
python -m vision.benchmark --frames 50
```

### 3. Automated Test Suite
Runs 46 comprehensive tests checking module logic, schema compliance, and RAG dispatch:
```bash
python -m vision.test_vision
```

---

## Module Layout

| File | Purpose |
|---|---|
| `capture.py` | Unified video capture wrapper supporting files, webcams, and RTSP streams. |
| `detect.py` | Ultralytics YOLO person detector (class 0 only) with configurable confidence threshold. |
| `tracker.py` | Centroid tracker that matches detections across frames and assigns consistent IDs. |
| `zones.py` | Polygon boundary definition, point-in-polygon math, risk tiers, and dwell persistence check. |
| `crowd.py` | Crowd formation monitoring based on person density and count persistence. |
| `features.py` | Computes density, motion score, occupancy, and exports `event_feature_vector.json`. |
| `pipeline.py` | Wires capture → detect → track → zone/crowd → features, and dispatches incidents to RAG. |
| `benchmark.py` | Latency (ms), FPS, and pipeline overhead measurement harness. |
| `test_vision.py` | Unit and integration test suite with synthetic verification. |
| `run_demo.py` | CLI entry point with configurable camera, zone, crowd, and RAG parameters. |
| `requirements.txt` | Core vision dependencies (`opencv-python`, `ultralytics`, `numpy`). |

---

## Shared Contracts Compliance

### 1. Federated Learning Track (`docs/schemas/event_feature_vector.json`)
`features.compute_features()` produces:
```json
{
  "camera_name": "CAM 01",
  "zone_name": "Restricted Zone Alpha",
  "timestamp": "2026-11-08T16:00:47.123456+00:00",
  "person_count": 3,
  "density": 0.42,
  "occupancy": true,
  "motion_score": 0.18,
  "zone_dwell_seconds": 4.2,
  "max_confidence": 0.94
}
```

### 2. RAG & Backend Tracks (`docs/schemas/incident.json` & `rag/api.py:Incident`)
`candidate.to_incident_dict()` produces:
```json
{
  "incident_ref": "restricted_area_entry_cam01_restricted_zone_alpha_20261116_085411",
  "event_type": "restricted_area_entry",
  "camera_name": "CAM 01",
  "zone_name": "Restricted Zone Alpha",
  "timestamp": "2026-11-16T08:54:11Z",
  "confidence": 0.92,
  "duration_seconds": 4.5,
  "severity": "high",
  "status": "open"
}
```
*(For crowd events, `event_type` is `"crowd_formation"`.)*
