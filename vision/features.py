"""
features.py — Compute the event feature vector from current detections and
zone state.

Produces records matching ``docs/schemas/event_feature_vector.json``.
These records feed the federated event classifier (``federated/``) and the
live decision stage.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np

from .detect import Detection
from .tracker import TrackedPerson
from .zones import ZoneConfig


def compute_features(
    detections: list[Detection],
    tracks: dict[int, TrackedPerson],
    zone: ZoneConfig,
    camera_name: str,
    frame_shape: tuple[int, int],  # (height, width) of the frame
    dwell_seconds: float = 0.0,
    max_confidence: float = 0.0,
    prev_centers: dict[int, tuple[int, int]] | None = None,
) -> dict:
    """
    Build one feature-vector dict matching the shared schema.

    Parameters
    ----------
    detections : list[Detection]
        Raw detections for the current frame.
    tracks : dict[int, TrackedPerson]
        Current tracker state.
    zone : ZoneConfig
        The zone being evaluated.
    camera_name : str
        Logical camera identifier.
    frame_shape : tuple[int, int]
        (height, width) used to normalize density.
    dwell_seconds : float
        Longest current dwell time inside this zone.
    max_confidence : float
        Highest detection confidence among people inside this zone.
    prev_centers : dict | None
        Previous-frame centres keyed by person_id — used to compute
        motion score.  Pass ``None`` on the first frame.

    Returns
    -------
    dict
        Event feature vector ready for JSON serialization.
    """
    person_count = len(detections)

    # ------------------------------------------------------------------
    # density  =  total person-bbox area  /  frame area
    # ------------------------------------------------------------------
    frame_area = frame_shape[0] * frame_shape[1]
    bbox_area = sum(d.area for d in detections)
    density = round(bbox_area / frame_area, 4) if frame_area else 0.0

    # ------------------------------------------------------------------
    # occupancy  =  more than one person in the zone
    # ------------------------------------------------------------------
    occupancy = person_count > 1

    # ------------------------------------------------------------------
    # motion_score  =  mean centroid displacement (normalised by frame
    #                   diagonal) between this frame and the last
    # ------------------------------------------------------------------
    motion_score = 0.0
    if prev_centers and tracks:
        diag = np.hypot(*frame_shape)
        displacements: list[float] = []
        for pid, person in tracks.items():
            if pid in prev_centers:
                dx = person.center[0] - prev_centers[pid][0]
                dy = person.center[1] - prev_centers[pid][1]
                displacements.append(np.hypot(dx, dy) / diag)
        if displacements:
            motion_score = round(float(np.mean(displacements)), 4)

    return {
        "camera_name": camera_name,
        "zone_name": zone.zone_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "person_count": person_count,
        "density": density,
        "occupancy": occupancy,
        "motion_score": motion_score,
        "zone_dwell_seconds": round(dwell_seconds, 2),
        "max_confidence": round(max_confidence, 4),
    }
