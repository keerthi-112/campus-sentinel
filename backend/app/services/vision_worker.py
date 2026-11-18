"""
vision_worker.py — Runs the Vision & Zone track live inside the backend, one
thread per active camera, and streams annotated frames to the dashboard.

It reuses the vision track's own building blocks unchanged:
    PersonDetector → CentroidTracker → ZoneMonitor / CrowdMonitor → compute_features

What it adds on top (application concerns):
  * start / stop per camera from the dashboard, camera + zone config from the DB
  * real-time pacing for video files (frames are skipped to keep up with the
    clip's own FPS), so dwell timers measure video seconds on any machine
  * optional looping of demo clips
  * each candidate is passed with its feature vector to the incident decision
    stage (federated classifier → incident → RAG report)
  * the latest annotated JPEG is kept in memory for the MJPEG stream; frames
    are never written to disk or the database
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ..config import DEVICE, YOLO_MODEL

log = logging.getLogger("sentinel.vision")

try:
    from vision.crowd import CrowdConfig, CrowdMonitor
    from vision.detect import PersonDetector
    from vision.features import compute_features
    from vision.tracker import CentroidTracker
    from vision.zones import ZoneConfig, ZoneMonitor

    VISION_AVAILABLE, VISION_ERROR = True, None
except Exception as exc:  # pragma: no cover - depends on environment
    VISION_AVAILABLE, VISION_ERROR = False, f"{type(exc).__name__}: {exc}"


@dataclass
class CameraSpec:
    id: int
    camera_name: str
    zone_name: str
    source_type: str
    source: str
    restricted_region: list
    tier: int
    dwell_threshold: float
    confidence_threshold: float
    crowd_count: int
    loop_video: bool

    @classmethod
    def from_row(cls, row) -> "CameraSpec":
        return cls(row.id, row.camera_name, row.zone_name, row.source_type, row.source,
                   list(row.restricted_region or []), row.tier, row.dwell_threshold,
                   row.confidence_threshold, row.crowd_count, row.loop_video)


@dataclass
class Runtime:
    state: str = "starting"  # starting | running | stopped | error | ended
    error: str | None = None
    fps: float = 0.0
    frames: int = 0
    persons: int = 0
    in_zone: int = 0
    incidents: int = 0
    width: int = 0
    height: int = 0
    frame_ms: float = 0.0  # processing time per frame (YOLO + tracking + zones), moving average
    live: dict = field(default_factory=dict)  # latest feature vector + classifier score (for the zone card)
    started_at: float = field(default_factory=time.time)


def placeholder_jpeg(text: str, w: int = 640, h: int = 360) -> bytes:
    img = np.full((h, w, 3), 24, dtype=np.uint8)
    cv2.putText(img, text, (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    return cv2.imencode(".jpg", img)[1].tobytes()


def open_source(spec_type: str, source: str) -> cv2.VideoCapture:
    src = int(source) if spec_type == "webcam" else source
    return cv2.VideoCapture(src)


def default_region(w: int, h: int) -> list[list[int]]:
    """Centre rectangle (same idea as vision.run_demo) when no polygon was drawn."""
    return [[w // 4, h // 4], [3 * w // 4, h // 4], [3 * w // 4, 3 * h // 4], [w // 4, 3 * h // 4]]


class CameraWorker(threading.Thread):
    def __init__(self, spec: CameraSpec, on_candidate) -> None:
        super().__init__(name=f"camera-{spec.id}", daemon=True)
        self.spec = spec
        self.on_candidate = on_candidate
        self.runtime = Runtime()
        self._stop = threading.Event()
        self._jpeg: bytes | None = None
        self._raw: np.ndarray | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._stop.set()

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def latest_raw(self) -> np.ndarray | None:
        with self._lock:
            return None if self._raw is None else self._raw.copy()

    # ------------------------------------------------------------------
    def run(self) -> None:
        try:
            self._loop()
        except Exception as exc:
            log.exception("camera %s crashed", self.spec.camera_name)
            self.runtime.state, self.runtime.error = "error", str(exc)
            with self._lock:
                self._jpeg = placeholder_jpeg(f"error: {str(exc)[:60]}")

    def _loop(self) -> None:
        s = self.spec
        cap = open_source(s.source_type, s.source)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open source '{s.source}' ({s.source_type})")
        is_file = s.source_type == "file"
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if not (1.0 <= src_fps <= 120.0):
            src_fps = 25.0

        detector = PersonDetector(model_name=YOLO_MODEL, confidence_threshold=s.confidence_threshold,
                                  device=DEVICE)
        tracker = CentroidTracker()
        crowd = CrowdMonitor(CrowdConfig(threshold_count=s.crowd_count, dwell_threshold=3.0))
        zone_monitor: ZoneMonitor | None = None
        zone: ZoneConfig | None = None
        prev_centers: dict = {}

        t_start = time.time()
        frame_pos = 0  # index of the next frame to read (files)
        last_tick, ticks = time.time(), 0
        self.runtime.state = "running"

        while not self._stop.is_set():
            if is_file:
                # Real-time pacing: drop frames we are too slow for, wait if too fast.
                target = int((time.time() - t_start) * src_fps)
                if target < frame_pos:
                    time.sleep(min((frame_pos - target) / src_fps, 0.05))
                    continue
                while frame_pos < target and cap.grab():
                    frame_pos += 1
            ok, frame = cap.read()
            frame_pos += 1
            if not ok:
                if is_file and s.loop_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    t_start, frame_pos = time.time(), 0
                    tracker = CentroidTracker()  # new pass of the clip = new people
                    prev_centers = {}
                    continue
                self.runtime.state = "ended"
                break

            h, w = frame.shape[:2]
            if zone_monitor is None:
                poly = s.restricted_region if len(s.restricted_region) >= 3 else default_region(w, h)
                zone = ZoneConfig(zone_name=s.zone_name, polygon=[tuple(p) for p in poly],
                                  dwell_threshold=s.dwell_threshold, tier=s.tier)
                zone_monitor = ZoneMonitor([zone])
                self.runtime.width, self.runtime.height = w, h

            t_frame = time.perf_counter()
            detections = detector.detect(frame)
            tracks = tracker.update(detections)
            zone_cands = zone_monitor.update(tracks, s.camera_name)
            density = round(sum(d.area for d in detections) / float(w * h), 4)
            crowd_cands = crowd.update(len(detections), density, s.camera_name, zone_name=s.zone_name)

            for cand in list(zone_cands) + list(crowd_cands):
                is_zone = hasattr(cand, "person_id")
                dwell = cand.dwell_seconds  # zone: time in zone; crowd: how long the crowd has persisted
                max_conf = cand.max_confidence if is_zone else max(
                    (d.confidence for d in detections), default=0.0)
                fv = compute_features(detections, tracks, zone, s.camera_name, (h, w),
                                      dwell_seconds=dwell, max_confidence=max_conf,
                                      prev_centers=prev_centers)
                incident = cand.to_incident_dict()
                try:
                    # several people crossing in the same second share one incident_ref → one incident
                    if self.on_candidate(incident, fv, s.id):
                        self.runtime.incidents += 1
                except Exception:
                    log.exception("candidate handling failed")

            in_zone = sum(1 for p in tracks.values() if zone_monitor.is_point_in_zone(p.center, zone))
            ms = (time.perf_counter() - t_frame) * 1000
            self.runtime.frame_ms = round(ms if not self.runtime.frame_ms else 0.9 * self.runtime.frame_ms + 0.1 * ms, 1)
            if self.runtime.frames % 15 == 0:  # clean frame for the zone editor (memory only)
                with self._lock:
                    self._raw = frame.copy()
            self._annotate(frame, detections, tracks, zone_monitor, zone, in_zone)
            ok_jpg, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            with self._lock:
                if ok_jpg:
                    self._jpeg = buf.tobytes()
            self.runtime.persons, self.runtime.in_zone = len(detections), in_zone
            if self.runtime.frames % 5 == 0:
                self._update_live(detections, tracks, zone_monitor, zone, (h, w), prev_centers)
            prev_centers = {pid: p.center for pid, p in tracks.items()}

            ticks += 1
            self.runtime.frames += 1
            if time.time() - last_tick >= 1.0:
                self.runtime.fps = round(ticks / (time.time() - last_tick), 1)
                last_tick, ticks = time.time(), 0

        cap.release()
        if self.runtime.state == "running":
            self.runtime.state = "stopped"
        with self._lock:
            self._jpeg = placeholder_jpeg(f"{s.camera_name}: {self.runtime.state}")

    # ------------------------------------------------------------------
    def _update_live(self, detections, tracks, zone_monitor, zone, shape, prev_centers) -> None:
        """Live read-out for the dashboard zone card: the same feature vector the
        classifier sees, plus what the federated model would say right now."""
        try:
            from .incidents import classifier

            dwell, max_conf = zone_monitor.get_zone_dwell_info(zone, tracks)
            fv = compute_features(detections, tracks, zone, self.spec.camera_name, shape,
                                  dwell_seconds=dwell, max_confidence=max_conf, prev_centers=prev_centers)
            crowd = fv["person_count"] >= self.spec.crowd_count
            et = "crowd_formation" if crowd else "restricted_area_entry"
            in_zone = self.runtime.in_zone
            score = classifier.score(fv, et) if (in_zone or crowd) else None
            if crowd or dwell >= zone.dwell_threshold:
                state = "alert"
            elif in_zone:
                state = "watching"
            else:
                state = "clear"
            self.runtime.live = {**fv, "event_type": et, "score": None if score is None else round(score, 3),
                                 "state": state, "dwell_threshold": zone.dwell_threshold, "tier": zone.tier}
        except Exception:  # never let the read-out break monitoring
            log.debug("live feature update failed", exc_info=True)

    def _annotate(self, frame, detections, tracks, zone_monitor, zone, in_zone) -> None:
        zone_monitor.draw_zones(frame)
        dwells = zone_monitor._dwells.get(zone.zone_name, {})
        now = time.time()
        for pid, p in tracks.items():
            if p.frames_since_seen:
                continue
            x1, y1, x2, y2 = p.bbox
            inside = pid in dwells
            color = (0, 0, 255) if inside else (0, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID {pid} {p.confidence:.2f}"
            if inside:
                label += f" | {now - dwells[pid].enter_time:.1f}s"
            cv2.putText(frame, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        hud = f"{self.spec.camera_name} | people {len(detections)} | in zone {in_zone} | {self.runtime.fps:.0f} fps"
        cv2.rectangle(frame, (0, frame.shape[0] - 24), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.putText(frame, hud, (8, frame.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


class VisionManager:
    """Owns all camera workers."""

    def __init__(self) -> None:
        self.workers: dict[int, CameraWorker] = {}
        self._lock = threading.Lock()
        self.on_candidate = None  # set by main.py

    @property
    def available(self) -> bool:
        return VISION_AVAILABLE

    def start(self, spec: CameraSpec) -> None:
        if not VISION_AVAILABLE:
            raise RuntimeError(f"vision track unavailable: {VISION_ERROR}")
        with self._lock:
            old = self.workers.get(spec.id)
            if old and old.is_alive():
                old.stop()
                old.join(timeout=5)
            w = CameraWorker(spec, self.on_candidate)
            self.workers[spec.id] = w
            w.start()

    def stop(self, camera_id: int) -> None:
        with self._lock:
            w = self.workers.get(camera_id)
        if w:
            w.stop()
            w.join(timeout=5)

    def stop_all(self) -> None:
        for cid in list(self.workers):
            self.stop(cid)

    def runtime(self, camera_id: int) -> dict | None:
        w = self.workers.get(camera_id)
        if not w:
            return None
        r = w.runtime
        state = r.state if w.is_alive() or r.state in ("error", "ended") else "stopped"
        return {"state": state, "error": r.error, "fps": r.fps, "frames": r.frames, "persons": r.persons,
                "in_zone": r.in_zone, "incidents": r.incidents, "width": r.width, "height": r.height,
                "frame_ms": r.frame_ms,
                "live": r.live if state == "running" else {}}

    def jpeg(self, camera_id: int) -> bytes | None:
        w = self.workers.get(camera_id)
        return w.latest_jpeg() if w else None

    def raw_frame(self, camera_id: int) -> np.ndarray | None:
        w = self.workers.get(camera_id)
        return w.latest_raw() if w and w.is_alive() else None

    def is_running(self, camera_id: int) -> bool:
        w = self.workers.get(camera_id)
        return bool(w and w.is_alive())


def grab_snapshot(source_type: str, source: str) -> tuple[bytes, int, int]:
    """One clean frame (no overlays) for the polygon editor."""
    cap = open_source(source_type, source)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"cannot open source '{source}'")
        frame = None
        for _ in range(5):  # webcams often return black first frames
            ok, f = cap.read()
            if ok:
                frame = f
                if source_type == "file":
                    break
        if frame is None:
            raise RuntimeError("source returned no frames")
        h, w = frame.shape[:2]
        return cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes(), w, h
    finally:
        cap.release()


vision_manager = VisionManager()
