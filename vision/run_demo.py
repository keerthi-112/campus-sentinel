#!/usr/bin/env python3
"""
run_demo.py — Quick demo of the full vision pipeline.

Usage
-----
# With webcam (default):
    python -m vision.run_demo

# With a video file:
    python -m vision.run_demo --source data/videos/gate1.mp4 --type file

# With Apple Silicon GPU acceleration:
    python -m vision.run_demo --device mps

# Export feature vectors for Federated Learning:
    python -m vision.run_demo --export-features data/features.jsonl

Press 'q' in the OpenCV window to stop.
"""

import argparse
from pathlib import Path

from vision.capture import CameraConfig
from vision.zones import ZoneConfig
from vision.crowd import CrowdConfig
from vision.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Campus Sentinel — Vision demo")
    parser.add_argument("--source", default="0",
                        help="Video file path, webcam index, or RTSP URL")
    parser.add_argument("--type", default="webcam",
                        choices=["file", "webcam", "rtsp"],
                        help="Source type")
    parser.add_argument("--camera-name", default="CAM 01 - Main Gate",
                        help="Logical camera name")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="YOLO model checkpoint")
    parser.add_argument("--confidence", type=float, default=0.40,
                        help="Detection confidence threshold")
    parser.add_argument("--device", default="cpu",
                        help="Inference device: cpu, cuda, mps")
    parser.add_argument("--dwell", type=float, default=4.0,
                        help="Zone dwell persistence threshold in seconds")
    parser.add_argument("--crowd-count", type=int, default=5,
                        help="Minimum person count to trigger crowd alert")
    parser.add_argument("--export-features", default=None,
                        help="Path to export event feature vectors (e.g. data/features.jsonl)")
    parser.add_argument("--no-window", action="store_true",
                        help="Disable the OpenCV display window (for headless/server mode)")
    args = parser.parse_args()

    # -----------------------------------------------------------------
    # Camera config
    # -----------------------------------------------------------------
    source = int(args.source) if args.type == "webcam" else args.source
    camera = CameraConfig(
        name=args.camera_name,
        source=source,
        source_type=args.type,
    )

    # -----------------------------------------------------------------
    # Zone config — default demo zone (a rectangle in the centre of a
    # 640×480 frame).  Custom polygons can be loaded dynamically.
    # -----------------------------------------------------------------
    demo_zone = ZoneConfig(
        zone_name="Restricted Zone Alpha",
        polygon=[(160, 120), (480, 120), (480, 360), (160, 360)],
        dwell_threshold=args.dwell,
    )

    crowd_cfg = CrowdConfig(
        threshold_count=args.crowd_count,
        dwell_threshold=3.0,
    )

    print("=" * 60)
    print("CAMPUS SENTINEL — VISION PIPELINE RUNNER")
    print("=" * 60)
    print(f"  Source          : {camera.source} ({camera.source_type})")
    print(f"  Camera          : {camera.name}")
    print(f"  Zone            : {demo_zone.zone_name}")
    print(f"  Dwell Threshold : {demo_zone.dwell_threshold}s")
    print(f"  Crowd Alert     : >= {crowd_cfg.threshold_count} people")
    print(f"  Model           : {args.model} ({args.device})")
    if args.export_features:
        print(f"  Feature Export  : {args.export_features}")
    print("  Controls        : Press 'q' in video window to exit.")
    print("=" * 60)

    run_pipeline(
        camera=camera,
        zones=[demo_zone],
        model_name=args.model,
        confidence=args.confidence,
        device=args.device,
        crowd_config=crowd_cfg,
        show_window=not args.no_window,
        export_features_path=args.export_features,
    )


if __name__ == "__main__":
    main()
