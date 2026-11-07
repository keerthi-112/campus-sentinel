#!/usr/bin/env python3
"""
run_demo.py — Quick demo of the full vision pipeline.

Usage
-----
# With a video file:
    python -m vision.run_demo --source videos/gate1.mp4

# With webcam (index 0):
    python -m vision.run_demo --source 0 --type webcam

Press 'q' in the OpenCV window to stop.
"""

import argparse

from vision.capture import CameraConfig
from vision.zones import ZoneConfig
from vision.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Campus Sentinel — Vision demo")
    parser.add_argument("--source", default="0",
                        help="Video file path, webcam index, or RTSP URL")
    parser.add_argument("--type", default="webcam",
                        choices=["file", "webcam", "rtsp"],
                        help="Source type")
    parser.add_argument("--camera-name", default="Demo Cam",
                        help="Logical camera name")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="YOLO model checkpoint")
    parser.add_argument("--confidence", type=float, default=0.40,
                        help="Detection confidence threshold")
    parser.add_argument("--device", default="cpu",
                        help="Inference device: cpu, cuda, mps")
    parser.add_argument("--no-window", action="store_true",
                        help="Disable the OpenCV display window")
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
    # 640×480 frame).  Replace with real coordinates for your test video.
    # -----------------------------------------------------------------
    demo_zone = ZoneConfig(
        zone_name="Demo Restricted Area",
        polygon=[(160, 120), (480, 120), (480, 360), (160, 360)],
        dwell_threshold=4.0,
    )

    print("=" * 60)
    print("Campus Sentinel — Vision Pipeline Demo")
    print(f"  Source : {camera.source}  ({camera.source_type})")
    print(f"  Camera : {camera.name}")
    print(f"  Zone   : {demo_zone.zone_name}")
    print(f"  Dwell  : {demo_zone.dwell_threshold}s")
    print(f"  Model  : {args.model}")
    print("  Press 'q' to quit.")
    print("=" * 60)

    run_pipeline(
        camera=camera,
        zones=[demo_zone],
        model_name=args.model,
        confidence=args.confidence,
        device=args.device,
        show_window=not args.no_window,
    )


if __name__ == "__main__":
    main()
