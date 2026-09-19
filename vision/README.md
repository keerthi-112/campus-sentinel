# vision/

Owns: video/camera input, YOLO26n person detection, tracking, restricted-zone
and crowd logic.

Reads: a file, webcam, or (later) RTSP stream.
Produces: `docs/schemas/event_feature_vector.json`-shaped records, handed to
`federated/` for training and to the live event-decision stage.

## Planned layout

```
vision/
  capture.py       # OpenCV input: file / webcam / RTSP
  detect.py        # YOLO26n inference
  tracker.py       # lightweight tracking to dedupe detections
  zones.py         # polygon/rectangle zone config + dwell timer
  features.py      # count / density / occupancy / motion -> feature vector
```
