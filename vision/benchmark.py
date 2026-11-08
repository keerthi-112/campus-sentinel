"""
benchmark.py — Latency, throughput (FPS), and performance evaluation harness.

Evaluates:
- YOLO inference latency (mean, p95, min, max in ms) and effective FPS
- End-to-end pipeline latency (capture -> detect -> track -> zone -> features)
- Resource metrics for reporting in project dossier (§09 Evaluation)
"""

from __future__ import annotations

import argparse
import time
import numpy as np

from .detect import PersonDetector, Detection
from .tracker import CentroidTracker
from .zones import ZoneConfig, ZoneMonitor
from .features import compute_features


def benchmark_detector(
    model_name: str = "yolo11n.pt",
    device: str = "cpu",
    num_frames: int = 50,
    frame_size: tuple[int, int] = (480, 640),
) -> dict:
    """
    Measures pure inference latency and FPS over synthetic/mock frames.
    """
    print(f"\n[benchmark] Loading model '{model_name}' on device '{device}'...")
    detector = PersonDetector(model_name=model_name, device=device)

    # Warmup
    dummy = np.zeros((*frame_size, 3), dtype=np.uint8)
    for _ in range(5):
        detector.detect(dummy)

    latencies_ms: list[float] = []

    print(f"[benchmark] Running inference over {num_frames} frames...")
    for _ in range(num_frames):
        # Create synthetic frame with noise
        frame = np.random.randint(0, 255, (*frame_size, 3), dtype=np.uint8)
        start = time.perf_counter()
        _ = detector.detect(frame)
        duration_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(duration_ms)

    mean_ms = float(np.mean(latencies_ms))
    p95_ms = float(np.percentile(latencies_ms, 95))
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    return {
        "model": model_name,
        "device": device,
        "frames_tested": num_frames,
        "mean_latency_ms": round(mean_ms, 2),
        "p95_latency_ms": round(p95_ms, 2),
        "fps": round(fps, 1),
    }


def benchmark_pipeline_overhead(num_iterations: int = 200) -> dict:
    """
    Measures the non-YOLO overhead (tracking, zone point-in-poly, feature computation).
    """
    tracker = CentroidTracker()
    zone = ZoneConfig("BenchZone", [(50, 50), (250, 50), (250, 250), (50, 250)])
    monitor = ZoneMonitor([zone])

    dets = [
        Detection(bbox=(60, 60, 100, 150), confidence=0.88, center=(80, 105)),
        Detection(bbox=(300, 100, 360, 200), confidence=0.91, center=(330, 150)),
    ]

    latencies_us: list[float] = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        tracks = tracker.update(dets)
        _ = monitor.update(tracks, "BENCH-CAM")
        dwell, max_conf = monitor.get_zone_dwell_info(zone, tracks)
        _ = compute_features(
            detections=dets,
            tracks=tracks,
            zone=zone,
            camera_name="BENCH-CAM",
            frame_shape=(480, 640),
            dwell_seconds=dwell,
            max_confidence=max_conf,
        )
        duration_us = (time.perf_counter() - start) * 1_000_000.0
        latencies_us.append(duration_us)

    mean_us = float(np.mean(latencies_us))
    return {
        "tracking_and_zone_overhead_us": round(mean_us, 2),
        "overhead_ms": round(mean_us / 1000.0, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Campus Sentinel Vision Benchmark")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO weights identifier")
    parser.add_argument("--device", default="cpu", help="cpu / mps / cuda")
    parser.add_argument("--frames", type=int, default=30, help="Number of benchmark frames")
    args = parser.parse_args()

    print("=" * 60)
    print("CAMPUS SENTINEL — VISION BENCHMARK HARNESS")
    print("=" * 60)

    det_metrics = benchmark_detector(
        model_name=args.model,
        device=args.device,
        num_frames=args.frames,
    )

    overhead = benchmark_pipeline_overhead()

    print("\n" + "-" * 60)
    print("PERFORMANCE METRICS SUMMARY")
    print("-" * 60)
    print(f"Model checkpoint       : {det_metrics['model']}")
    print(f"Inference device       : {det_metrics['device']}")
    print(f"Mean detection latency : {det_metrics['mean_latency_ms']} ms")
    print(f"95th percentile (P95)  : {det_metrics['p95_latency_ms']} ms")
    print(f"Throughput (FPS)       : {det_metrics['fps']} frames/sec")
    print(f"Zone/Tracker overhead  : {overhead['overhead_ms']} ms ({overhead['tracking_and_zone_overhead_us']} µs)")
    total_latency = det_metrics['mean_latency_ms'] + overhead['overhead_ms']
    print(f"Estimated End-to-End   : {round(total_latency, 2)} ms/frame")
    print("=" * 60)


if __name__ == "__main__":
    main()
