# vision/

Owns: video/camera input, YOLO person detection, tracking, restricted-zone
and crowd logic.

Reads: a file, webcam, or (later) RTSP stream.
Produces: `docs/schemas/event_feature_vector.json`-shaped records, handed to
`federated/` for training and to the live event-decision stage.

## Setup

```bash
cd campus-sentinel
python3 -m venv .venv
source .venv/bin/activate
pip install -r vision/requirements.txt
```

## Quick demo

```bash
# Webcam (default)
python -m vision.run_demo

# Video file
python -m vision.run_demo --source path/to/clip.mp4 --type file

# Headless (no OpenCV window)
python -m vision.run_demo --source path/to/clip.mp4 --type file --no-window
```

Press **q** in the OpenCV window to stop.

## Module layout

```
vision/
  __init__.py
  capture.py       # OpenCV input: file / webcam / RTSP
  detect.py        # YOLO person detection (class 0 only)
  tracker.py       # Centroid tracker — stable IDs across frames
  zones.py         # Polygon zones, dwell timer, incident candidates
  features.py      # count / density / occupancy / motion → feature vector
  pipeline.py      # Wires everything into one continuous loop
  run_demo.py      # CLI demo entry point
  requirements.txt
```

## How it connects to other tracks

1. **→ federated/**  — `features.py` produces event-feature-vector dicts
   matching `docs/schemas/event_feature_vector.json`. The federated track
   consumes these for classifier training.
2. **→ backend/**  — `zones.py` emits `ZoneIncidentCandidate` objects.
   Once the event classifier confirms them, they become incidents matching
   `docs/schemas/incident.json` and get POSTed to the backend API.
3. **→ frontend/**  — The annotated video frame (bounding boxes + zone
   polygons) can be streamed to the React dashboard.

## Configuration

Zone polygons are currently defined in code (see `run_demo.py`).  A later
milestone will load them from a config file or from the backend API so the
dashboard can draw/edit zones interactively.

## Notes

- **YOLO model:** Ships with `yolo11n.pt` (reliable, widely tested).  Switch
  to `yolov8n.pt` or a future `yolo26n.pt` by passing `--model`.
- **Device:** `--device mps` for Apple Silicon GPU, `--device cuda` for
  NVIDIA.
- **Dwell threshold:** Default 4.0 s.  Configurable per zone via `ZoneConfig`.
