#!/usr/bin/env python3
"""
test_vision.py — Automated test suite for the vision track.

Tests all modules WITHOUT requiring a webcam, display, or active video source.
Uses synthetic data (mock detections, simulated tracker states) to validate logic
and verify strict compliance with shared schemas.
"""

import sys
import time
import json
import tempfile
from pathlib import Path
import numpy as np

PASS = 0
FAIL = 0
RESULTS = []

def report(name, passed, detail=""):
    global PASS, FAIL
    if passed:
        PASS += 1
        RESULTS.append(f"  ✅ {name}")
    else:
        FAIL += 1
        RESULTS.append(f"  ❌ {name} — {detail}")


# ==================================================================
# TEST 1: capture.py — CameraConfig construction
# ==================================================================
print("\n[TEST] capture.py ...")
try:
    from vision.capture import CameraConfig, VideoCapture

    c1 = CameraConfig(name="Gate 1", source="test.mp4", source_type="file")
    report("CameraConfig(file)", c1.name == "Gate 1" and c1.source == "test.mp4")

    c2 = CameraConfig(name="Webcam", source="0", source_type="webcam")
    report("CameraConfig(webcam) source→int", isinstance(c2.source, int) and c2.source == 0)

    c3 = CameraConfig(name="FPS test", source="x.mp4", fps_limit=15)
    report("CameraConfig fps_limit", c3.fps_limit == 15)

    vc = VideoCapture(c1)
    opened = vc.open()
    report("VideoCapture.open() fails gracefully for missing file", not opened)
    vc.release()

except Exception as e:
    report("capture.py import/run", False, str(e))


# ==================================================================
# TEST 2: detect.py — Detection dataclass
# ==================================================================
print("[TEST] detect.py ...")
try:
    from vision.detect import Detection

    d = Detection(bbox=(100, 200, 200, 400), confidence=0.85, center=(150, 300))
    report("Detection.width", d.width == 100)
    report("Detection.height", d.height == 200)
    report("Detection.area", d.area == 20000)
    report("Detection.center", d.center == (150, 300))

except Exception as e:
    report("detect.py import/run", False, str(e))


# ==================================================================
# TEST 3: tracker.py — CentroidTracker
# ==================================================================
print("[TEST] tracker.py ...")
try:
    from vision.tracker import CentroidTracker
    from vision.detect import Detection

    tracker = CentroidTracker(max_disappeared=5, max_distance=80)

    dets1 = [
        Detection(bbox=(10, 10, 50, 50), confidence=0.9, center=(30, 30)),
        Detection(bbox=(200, 200, 250, 250), confidence=0.8, center=(225, 225)),
    ]
    tracks = tracker.update(dets1)
    report("Tracker registers 2 people", len(tracks) == 2)
    ids1 = set(tracks.keys())

    dets2 = [
        Detection(bbox=(12, 12, 52, 52), confidence=0.9, center=(32, 32)),
        Detection(bbox=(202, 202, 252, 252), confidence=0.8, center=(227, 227)),
    ]
    tracks = tracker.update(dets2)
    ids2 = set(tracks.keys())
    report("Tracker maintains IDs across continuous frames", ids1 == ids2)

    dets3 = [
        Detection(bbox=(14, 14, 54, 54), confidence=0.9, center=(34, 34)),
    ]
    tracks = tracker.update(dets3)
    report("Tracker retains disappeared track within grace period", len(tracks) == 2)

    for _ in range(6):
        tracks = tracker.update(dets3)
    report("Tracker evicts disappeared track after max_disappeared", len(tracks) == 1)

    for _ in range(10):
        tracks = tracker.update([])
    report("Tracker handles empty frames gracefully", len(tracks) == 0)

    dets_new = [
        Detection(bbox=(300, 300, 350, 350), confidence=0.7, center=(325, 325)),
    ]
    tracks = tracker.update(dets_new)
    report("Tracker assigns ID for newcomer", len(tracks) == 1)

except Exception as e:
    report("tracker.py import/run", False, str(e))


# ==================================================================
# TEST 4: zones.py — ZoneConfig, ZoneMonitor, and Incident Compliance
# ==================================================================
print("[TEST] zones.py ...")
try:
    from vision.zones import ZoneConfig, ZoneMonitor, ZoneIncidentCandidate
    from vision.tracker import TrackedPerson

    zone = ZoneConfig(
        zone_name="Test Zone",
        polygon=[(100, 100), (300, 100), (300, 300), (100, 300)],
        dwell_threshold=0.3,
    )
    report("ZoneConfig creation", zone.zone_name == "Test Zone")

    monitor = ZoneMonitor([zone])

    tracks_outside = {
        0: TrackedPerson(person_id=0, center=(50, 50), bbox=(30, 30, 70, 70), confidence=0.9)
    }
    candidates = monitor.update(tracks_outside, "CAM-TEST")
    report("No candidate for person outside zone", len(candidates) == 0)

    tracks_inside = {
        1: TrackedPerson(person_id=1, center=(200, 200), bbox=(180, 180, 220, 220), confidence=0.95)
    }
    candidates = monitor.update(tracks_inside, "CAM-TEST")
    report("No immediate candidate (dwell threshold pending)", len(candidates) == 0)

    time.sleep(0.35)

    candidates = monitor.update(tracks_inside, "CAM-TEST")
    report("Candidate emitted once dwell threshold elapsed", len(candidates) == 1)

    if candidates:
        c = candidates[0]
        inc_dict = c.to_incident_dict()

        # Validate against docs/schemas/incident.json contract
        INCIDENT_SCHEMA_FIELDS = {
            "event_type": str,
            "camera_name": str,
            "zone_name": str,
            "timestamp": str,
            "confidence": (int, float),
            "duration_seconds": (int, float),
            "severity": str,
            "status": str,
        }
        all_incident_fields_valid = True
        for k, expected_type in INCIDENT_SCHEMA_FIELDS.items():
            if k not in inc_dict:
                report(f"Incident schema field '{k}'", False, "Missing")
                all_incident_fields_valid = False
            elif not isinstance(inc_dict[k], expected_type):
                report(f"Incident schema field '{k}' type", False, f"Expected {expected_type}")
                all_incident_fields_valid = False

        if all_incident_fields_valid:
            report("to_incident_dict() 100% compliant with incident.json schema", True)

        report("incident.event_type == 'restricted_area_entry'", inc_dict["event_type"] == "restricted_area_entry")
        report("incident.status == 'open'", inc_dict["status"] == "open")
        report("incident.timestamp is ISO string with 'Z'", inc_dict["timestamp"].endswith("Z"))
        report("incident.severity in valid set ('low','medium','high')", inc_dict["severity"] in ["low", "medium", "high"])

    # Duplicate suppression check
    candidates_dup = monitor.update(tracks_inside, "CAM-TEST")
    report("No duplicate incident candidate on successive frames", len(candidates_dup) == 0)

    # Public helper methods
    dwell, conf = monitor.get_zone_dwell_info(zone, tracks_inside)
    report("Public get_zone_dwell_info() works", dwell >= 0.3 and conf == 0.95)
    report("Public is_point_in_zone() inside", monitor.is_point_in_zone((200, 200), zone))
    report("Public is_point_in_zone() outside", not monitor.is_point_in_zone((20, 20), zone))

    # Frame drawing
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    annotated = monitor.draw_zones(dummy_frame)
    report("draw_zones() renders without error", annotated is not None)

except Exception as e:
    report("zones.py import/run", False, str(e))


# ==================================================================
# TEST 5: crowd.py — CrowdMonitor and Crowd Candidates
# ==================================================================
print("[TEST] crowd.py ...")
try:
    from vision.crowd import CrowdConfig, CrowdMonitor, CrowdIncidentCandidate

    cfg = CrowdConfig(threshold_count=3, dwell_threshold=0.2)
    c_mon = CrowdMonitor(cfg)

    # Below threshold
    res = c_mon.update(person_count=2, density=0.10, camera_name="CAM-02")
    report("CrowdMonitor suppresses sub-threshold counts", len(res) == 0)

    # Exceed threshold
    res = c_mon.update(person_count=4, density=0.25, camera_name="CAM-02")
    report("CrowdMonitor waits for dwell duration", len(res) == 0)

    time.sleep(0.25)
    res = c_mon.update(person_count=4, density=0.25, camera_name="CAM-02")
    report("Crowd candidate emitted after dwell elapsed", len(res) == 1)

    if res:
        c_dict = res[0].to_incident_dict()
        report("crowd event_type == 'crowd_formation'", c_dict["event_type"] == "crowd_formation")
        report("crowd incident schema matches incident.json", all(k in c_dict for k in [
            "event_type", "camera_name", "zone_name", "timestamp", "confidence",
            "duration_seconds", "severity", "status"
        ]))

except Exception as e:
    report("crowd.py import/run", False, str(e))


# ==================================================================
# TEST 6: features.py — Feature Vector Schema Compliance & Export
# ==================================================================
print("[TEST] features.py ...")
try:
    from vision.features import compute_features, export_feature_vector
    from vision.detect import Detection
    from vision.tracker import TrackedPerson
    from vision.zones import ZoneConfig

    zone = ZoneConfig(
        zone_name="Lab 2 restricted area",
        polygon=[(0, 0), (640, 0), (640, 480), (0, 480)],
    )

    dets = [
        Detection(bbox=(100, 100, 200, 300), confidence=0.94, center=(150, 200)),
        Detection(bbox=(300, 100, 400, 300), confidence=0.87, center=(350, 200)),
        Detection(bbox=(500, 200, 580, 400), confidence=0.72, center=(540, 300)),
    ]

    tracks = {
        0: TrackedPerson(person_id=0, center=(150, 200), bbox=(100, 100, 200, 300), confidence=0.94),
        1: TrackedPerson(person_id=1, center=(350, 200), bbox=(300, 100, 400, 300), confidence=0.87),
        2: TrackedPerson(person_id=2, center=(540, 300), bbox=(500, 200, 580, 400), confidence=0.72),
    }

    prev_centers = {0: (145, 198), 1: (348, 199), 2: (535, 295)}

    fv = compute_features(
        detections=dets,
        tracks=tracks,
        zone=zone,
        camera_name="CAM 03",
        frame_shape=(480, 640),
        dwell_seconds=4.2,
        max_confidence=0.94,
        prev_centers=prev_centers,
    )

    FEATURE_SCHEMA_FIELDS = {
        "camera_name": str,
        "zone_name": str,
        "timestamp": str,
        "person_count": int,
        "density": (int, float),
        "occupancy": bool,
        "motion_score": (int, float),
        "zone_dwell_seconds": (int, float),
        "max_confidence": (int, float),
    }

    all_fv_valid = True
    for field, expected_type in FEATURE_SCHEMA_FIELDS.items():
        if field not in fv:
            report(f"Feature field '{field}'", False, "Missing")
            all_fv_valid = False
        elif not isinstance(fv[field], expected_type):
            report(f"Feature field '{field}' type", False, f"Expected {expected_type}")
            all_fv_valid = False

    if all_fv_valid:
        report("Feature vector 100% compliant with event_feature_vector.json", True)

    report("person_count == 3", fv["person_count"] == 3)
    report("occupancy is True (crowd/multi-person)", fv["occupancy"] is True)
    report("motion_score > 0 with displacement", fv["motion_score"] > 0)

    # Test export to JSON-Lines file for Federated track ingestion
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        tmp_path = tmp.name

    export_feature_vector(fv, tmp_path)
    with open(tmp_path, "r", encoding="utf-8") as f:
        line = f.readline()
        loaded = json.loads(line)
        report("export_feature_vector() creates valid JSONL record", loaded == fv)
    Path(tmp_path).unlink(missing_ok=True)

except Exception as e:
    report("features.py import/run", False, str(e))


# ==================================================================
# TEST 7: benchmark.py & pipeline helpers
# ==================================================================
print("[TEST] benchmark.py and pipeline helpers ...")
try:
    from vision.benchmark import benchmark_pipeline_overhead
    from vision.pipeline import _annotate, _print_candidate
    from vision.zones import ZoneIncidentCandidate

    bench_res = benchmark_pipeline_overhead(num_iterations=20)
    report("benchmark_pipeline_overhead() runs cleanly", "overhead_ms" in bench_res)

    cand = ZoneIncidentCandidate(
        person_id=99,
        zone_name="Restricted Zone Alpha",
        camera_name="CAM 01",
        enter_time=time.time() - 4.5,
        dwell_seconds=4.5,
        max_confidence=0.92,
        timestamp=time.time(),
    )
    _print_candidate(cand, cand.to_incident_dict())
    report("pipeline._print_candidate handles incident dict format", True)

except Exception as e:
    report("benchmark / pipeline helpers", False, str(e))


# ==================================================================
# TEST 8: run_demo.py importable
# ==================================================================
print("[TEST] run_demo.py ...")
try:
    import importlib.util
    spec = importlib.util.find_spec("vision.run_demo")
    report("vision.run_demo is importable via CLI", spec is not None)
except Exception as e:
    report("run_demo.py import", False, str(e))


# ==================================================================
# SUMMARY
# ==================================================================
print("\n" + "=" * 60)
print("CAMPUS SENTINEL — VISION TRACK TEST SUITE SUMMARY")
print("=" * 60)
for r in RESULTS:
    print(r)
print("-" * 60)
print(f"  PASSED: {PASS}   FAILED: {FAIL}   TOTAL: {PASS + FAIL}")
if FAIL == 0:
    print("  🎉 ALL TESTS PASSED STRICT COMPLIANCE CHECKS")
else:
    print(f"  ⚠️  {FAIL} FAILURE(S) — check details above")
print("=" * 60)

sys.exit(1 if FAIL > 0 else 0)
