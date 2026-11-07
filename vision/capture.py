"""
capture.py — Unified video input: file / webcam / RTSP.

Reads frames from any OpenCV-compatible source and yields them one at a time.
The rest of the vision pipeline never knows (or cares) where the frames came
from.
"""

from __future__ import annotations

import cv2
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator


@dataclass
class CameraConfig:
    """Configuration for a single camera / video source."""

    name: str  # logical name shown in the UI  e.g. "Gate 1"
    source: str | int  # file path, RTSP URL, or webcam index (0, 1, …)
    source_type: str = "file"  # "file" | "webcam" | "rtsp"
    fps_limit: int | None = None  # optional cap on processing rate

    def __post_init__(self) -> None:
        if self.source_type == "webcam":
            self.source = int(self.source)


class VideoCapture:
    """
    Wraps cv2.VideoCapture with retry logic and FPS limiting.

    Usage
    -----
    >>> cam = CameraConfig(name="Gate 1", source="videos/gate1.mp4")
    >>> for frame_info in VideoCapture(cam).frames():
    ...     process(frame_info)
    """

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._cap: cv2.VideoCapture | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> bool:
        """Open the underlying capture device. Returns True on success."""
        self._cap = cv2.VideoCapture(self.config.source)
        if not self._cap.isOpened():
            print(f"[capture] ERROR  cannot open {self.config.name} "
                  f"({self.config.source})")
            return False
        print(f"[capture] opened {self.config.name}  "
              f"source={self.config.source}  type={self.config.source_type}")
        return True

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------
    # Frame generator
    # ------------------------------------------------------------------

    def frames(self) -> Generator[dict, None, None]:
        """
        Yield dicts with keys: ``frame`` (BGR ndarray), ``frame_idx`` (int),
        ``timestamp`` (float, seconds since epoch), ``camera_name`` (str).
        """
        if self._cap is None and not self.open():
            return

        frame_idx = 0
        min_interval = 1.0 / self.config.fps_limit if self.config.fps_limit else 0
        last_time = 0.0

        while True:
            ok, frame = self._cap.read()
            if not ok:
                print(f"[capture] {self.config.name} — stream ended or lost.")
                break

            now = time.time()
            if now - last_time < min_interval:
                continue
            last_time = now

            yield {
                "frame": frame,
                "frame_idx": frame_idx,
                "timestamp": now,
                "camera_name": self.config.name,
            }
            frame_idx += 1

        self.release()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.release()
