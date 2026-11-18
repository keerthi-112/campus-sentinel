"""
features.py — Turn an event feature vector (docs/schemas/event_feature_vector.json)
into the fixed-length numeric input of the event classifier.

The scaling constants below are FIXED and agreed in advance by every zone.
They are deliberately not computed from data: computing a global mean/std would
require zones to share statistics about their local data, which is exactly the
kind of leakage federated learning is meant to avoid.
"""

from __future__ import annotations

import math

import numpy as np

# Order matters — every client and the server must use the same order.
#
# zone_dwell_seconds is in the feature-vector schema but NOT a classifier input:
# the classifier runs at the moment the vision rule fires, when the dwell (or
# crowd persistence) has just reached its threshold, so it always has the same
# value there and carries no information. The rule already enforces it.
FEATURE_NAMES: list[str] = [
    "person_count",        # people detected in the frame
    "density",             # person-bbox area / frame area
    "occupancy",           # more than one person present (0/1)
    "motion_score",        # mean centroid displacement per frame / frame diagonal
    "max_confidence",      # highest YOLO confidence (in the zone, or in the frame for crowds)
    "is_crowd_event",      # 1 if the candidate is crowd_formation, else 0
]

N_FEATURES = len(FEATURE_NAMES)

CROWD_EVENT = "crowd_formation"
ZONE_EVENT = "restricted_area_entry"


def to_vector(fv: dict, event_type: str | None = None) -> np.ndarray:
    """Map one feature-vector record to a scaled float32 vector in roughly [0, 1]."""
    et = event_type or fv.get("event_type") or ZONE_EVENT
    person_count = float(fv.get("person_count", 0) or 0)
    density = float(fv.get("density", 0.0) or 0.0)
    occupancy = 1.0 if fv.get("occupancy") else 0.0
    motion = float(fv.get("motion_score", 0.0) or 0.0)
    conf = float(fv.get("max_confidence", 0.0) or 0.0)

    # Scales match what the vision pipeline really produces (measured on the
    # evaluation clips): density ~0.02–0.3, motion ~0–0.03 per frame.
    return np.array(
        [
            min(math.log1p(max(person_count, 0.0)) / math.log1p(30.0), 1.5),
            min(max(density, 0.0) / 0.3, 1.5),
            occupancy,
            min(max(motion, 0.0) / 0.02, 1.5),
            min(max(conf, 0.0), 1.0),
            1.0 if et == CROWD_EVENT else 0.0,
        ],
        dtype=np.float32,
    )


def records_to_xy(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Convert labelled records (feature vector + 'label') to (X, y)."""
    if not records:
        return np.zeros((0, N_FEATURES), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    X = np.stack([to_vector(r) for r in records])
    y = np.array([float(r["label"]) for r in records], dtype=np.float32)
    return X, y
