"""
detect.py — YOLO26n person detection.

Wraps the Ultralytics YOLO model.  Only class 0 ("person") is kept;
everything else is discarded.  Output is a list of Detection objects
ready for tracking and zone logic.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from ultralytics import YOLO


# COCO class-id for "person"
_PERSON_CLS = 0


@dataclass
class Detection:
    """One detected person in a single frame."""

    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) pixel coords
    confidence: float
    center: tuple[int, int]  # (cx, cy)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return self.width * self.height


class PersonDetector:
    """
    Thin wrapper around YOLO for person-only detection.

    Parameters
    ----------
    model_name : str
        Ultralytics model identifier.  Default ``"yolo11n.pt"`` (reliable
        fallback).  Switch to ``"yolov8n.pt"`` or a future ``"yolo26n.pt"``
        once available and benchmarked.
    confidence_threshold : float
        Minimum confidence to accept a detection (default 0.40).
    device : str
        ``"cpu"``, ``"cuda"``, ``"mps"`` — passed straight to Ultralytics.
    """

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence_threshold: float = 0.40,
        device: str = "cpu",
    ) -> None:
        self.model = YOLO(model_name)
        self.conf_threshold = confidence_threshold
        self.device = device
        print(f"[detect] loaded {model_name}  conf≥{confidence_threshold}  "
              f"device={device}")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference on one BGR frame.

        Returns a list of ``Detection`` objects for every person found above
        the confidence threshold.
        """
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            classes=[_PERSON_CLS],
            device=self.device,
            verbose=False,
        )

        detections: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    center=(cx, cy),
                ))

        return detections
