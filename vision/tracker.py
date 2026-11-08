"""
tracker.py — Lightweight centroid tracker to de-duplicate detections across
frames.

Assigns a stable integer ID to each person for as long as they stay visible.
The tracker is intentionally simple (centroid distance) — the heavy lifting
is done by YOLO and the zone dwell logic.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field

from .detect import Detection


@dataclass
class TrackedPerson:
    """State held for one tracked individual."""

    person_id: int
    center: tuple[int, int]
    bbox: tuple[int, int, int, int]
    confidence: float
    frames_since_seen: int = 0


class CentroidTracker:
    """
    Match new detections to existing tracks using Euclidean distance on
    centroids.

    Parameters
    ----------
    max_disappeared : int
        Frames a track may go unmatched before being dropped (default 30).
    max_distance : float
        Maximum pixel distance to accept a centroid match (default 80).
    """

    def __init__(
        self,
        max_disappeared: int = 30,
        max_distance: float = 80.0,
    ) -> None:
        self._next_id = 0
        self._tracks: OrderedDict[int, TrackedPerson] = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    @property
    def tracks(self) -> dict[int, TrackedPerson]:
        return dict(self._tracks)

    def update(self, detections: list[Detection]) -> dict[int, TrackedPerson]:
        """
        Accept new detections, match / register / deregister as needed.
        Returns the current track dictionary.
        """
        # --- nothing detected this frame ---
        if not detections:
            for pid in list(self._tracks):
                self._tracks[pid].frames_since_seen += 1
                if self._tracks[pid].frames_since_seen > self.max_disappeared:
                    del self._tracks[pid]
            return self.tracks

        # --- first frame / no existing tracks ---
        if not self._tracks:
            for det in detections:
                self._register(det)
            return self.tracks

        # --- match existing tracks to new detections (greedy nearest) ---
        track_ids = list(self._tracks.keys())
        track_centers = [self._tracks[tid].center for tid in track_ids]

        det_centers = [d.center for d in detections]
        used_det: set[int] = set()
        used_trk: set[int] = set()

        # build pair-wise distances
        pairs: list[tuple[float, int, int]] = []
        for ti, tc in enumerate(track_centers):
            for di, dc in enumerate(det_centers):
                dist = math.hypot(tc[0] - dc[0], tc[1] - dc[1])
                pairs.append((dist, ti, di))
        pairs.sort()

        for dist, ti, di in pairs:
            if ti in used_trk or di in used_det:
                continue
            if dist > self.max_distance:
                break
            tid = track_ids[ti]
            det = detections[di]
            self._tracks[tid].center = det.center
            self._tracks[tid].bbox = det.bbox
            self._tracks[tid].confidence = det.confidence
            self._tracks[tid].frames_since_seen = 0
            used_trk.add(ti)
            used_det.add(di)

        # deregister unmatched tracks
        for ti in range(len(track_ids)):
            if ti not in used_trk:
                tid = track_ids[ti]
                self._tracks[tid].frames_since_seen += 1
                if self._tracks[tid].frames_since_seen > self.max_disappeared:
                    del self._tracks[tid]

        # register new detections
        for di in range(len(detections)):
            if di not in used_det:
                self._register(detections[di])

        return self.tracks

    # ------------------------------------------------------------------

    def _register(self, det: Detection) -> None:
        self._tracks[self._next_id] = TrackedPerson(
            person_id=self._next_id,
            center=det.center,
            bbox=det.bbox,
            confidence=det.confidence,
        )
        self._next_id += 1

