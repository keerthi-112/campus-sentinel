"""
data.py — Per-zone (client) datasets for federated training.

Two sources of local data:

1. Synthetic (default) — `generate_all()` simulates 3-5 campus zones whose
   event candidates look different from each other (non-IID): the main gate
   sees many crowds, the lab almost only restricted-zone entries, the parking
   lot is dark so YOLO confidence is lower, and so on. Each record has exactly
   the shape of docs/schemas/event_feature_vector.json plus `event_type` and a
   `label` (1 = real incident worth escalating, 0 = false alarm to ignore).

2. Real vision exports — `label_vision_features.py` turns a
   `vision.run_demo --export-features` JSONL file into the same format.

Every client only ever reads its own file; no raw records leave a zone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .features import CROWD_EVENT, ZONE_EVENT

FED_DIR = Path(__file__).resolve().parent
DATA_DIR = FED_DIR / "data"
CLIENT_DIR = DATA_DIR / "clients"
TEST_PATH = DATA_DIR / "test" / "global_test.jsonl"


@dataclass(frozen=True)
class ZoneProfile:
    zone_id: str
    camera_name: str
    zone_name: str
    n_samples: int          # how much local data this zone has
    positive_rate: float    # share of candidates that are real incidents
    crowd_share: float      # share of candidates that are crowd events
    conf_shift: float       # lighting / camera quality effect on YOLO confidence
    count_scale: float      # how busy the area is


# Deliberately heterogeneous: different sizes, label balance and feature shifts.
ZONE_PROFILES: list[ZoneProfile] = [
    ZoneProfile("gate_1", "CAM 01", "Main Gate", 1200, 0.25, 0.55, 0.00, 1.4),
    ZoneProfile("lab_2", "CAM 03", "Lab 2 restricted area", 180, 0.55, 0.03, 0.03, 0.6),
    ZoneProfile("parking", "CAM 05", "Parking Lot B", 420, 0.35, 0.15, -0.12, 0.8),
    ZoneProfile("library", "CAM 07", "Library Atrium", 320, 0.30, 0.35, 0.02, 1.1),
    ZoneProfile("hostel", "CAM 09", "Hostel Block C", 260, 0.40, 0.20, -0.05, 0.9),
]


def _beta(rng, mean, k=20.0):
    mean = float(np.clip(mean, 0.02, 0.98))
    return float(rng.beta(mean * k, (1 - mean) * k))


def _sample(rng: np.random.Generator, prof: ZoneProfile, label_noise: float) -> dict:
    """Draw one event candidate for a zone, with its ground-truth label.

    A candidate is what the vision rule emits at the moment it fires, so the
    dwell (zone) or persistence (crowd) has just reached its threshold. Value
    ranges match what the real pipeline produces (measured on the evaluation
    clips): density ~0.02–0.3, motion ~0–0.03 of the frame diagonal per frame.
    """
    is_crowd = rng.random() < prof.crowd_share
    real = rng.random() < prof.positive_rate
    conf_hi = lambda: _beta(rng, 0.85 + prof.conf_shift, k=40)

    if not is_crowd:
        # ---- restricted-area entry candidates ----
        dwell = 4.0 + float(rng.uniform(0, 0.3))
        if real:   # unauthorised person standing / loitering / working in the zone
            count, conf = int(rng.choice([1, 1, 1, 2])), conf_hi()
            motion = abs(rng.normal(0.0015, 0.0015))
        else:
            kind = rng.choice(["pass_through", "misdetection", "staff"], p=[0.45, 0.30, 0.25])
            if kind == "pass_through":     # walks slowly through, just crosses the dwell timer
                count, conf = int(rng.choice([1, 1, 2])), conf_hi()
                motion = abs(rng.normal(0.012, 0.005))
            elif kind == "misdetection":   # shadow, poster, reflection, partial occlusion
                count, conf = 1, _beta(rng, 0.50 + prof.conf_shift, k=15)
                motion = abs(rng.normal(0.0008, 0.0008))
            else:                          # authorised staff at work: the hard, ambiguous case
                count, conf = int(rng.choice([1, 2])), conf_hi()
                motion = abs(rng.normal(0.004, 0.003))
        density = float(np.clip(count * rng.normal(0.035, 0.012), 0.005, 0.4))
    else:
        # ---- crowd formation candidates ----
        dwell = 3.0 + float(rng.uniform(0, 0.3))
        if real:
            if rng.random() < 0.6:         # gathering that stays (protest, fight circle, blockage)
                count = max(5, int(round(rng.normal(7, 2) * prof.count_scale)))
                motion = abs(rng.normal(0.0015, 0.0015))
            else:                          # surge / rush
                count = max(5, int(round(rng.normal(12, 3) * prof.count_scale)))
                motion = abs(rng.normal(0.014, 0.006))
            density = float(np.clip(count * rng.uniform(0.013, 0.024), 0.05, 0.9))
            conf = _beta(rng, 0.82 + prof.conf_shift, k=40)
        else:
            if rng.random() < 0.6:         # people streaming past (class change, gate rush hour)
                count = max(5, int(round(rng.normal(6, 1))))
                density = float(np.clip(count * rng.uniform(0.009, 0.015), 0.03, 0.5))
                motion = abs(rng.normal(0.013, 0.005))
                conf = _beta(rng, 0.80 + prof.conf_shift, k=40)
            else:                          # duplicate boxes / reflections inflate the count
                count = int(rng.choice([5, 5, 6]))
                density = float(np.clip(count * rng.uniform(0.008, 0.016), 0.03, 0.5))
                motion = abs(rng.normal(0.001, 0.001))
                conf = _beta(rng, 0.52 + prof.conf_shift, k=15)

    label = int(real)
    if rng.random() < label_noise:
        label = 1 - label

    return {
        "camera_name": prof.camera_name,
        "zone_name": prof.zone_name,
        "person_count": int(count),
        "density": round(density, 4),
        "occupancy": bool(count > 1),
        "motion_score": round(float(motion), 5),
        "zone_dwell_seconds": round(dwell, 2),
        "max_confidence": round(float(np.clip(conf, 0.25, 0.99)), 4),
        "event_type": CROWD_EVENT if is_crowd else ZONE_EVENT,
        "label": label,
    }


def generate_zone(prof: ZoneProfile, n: int, seed: int, label_noise: float = 0.05) -> list[dict]:
    rng = np.random.default_rng(seed)
    t0 = datetime(2026, 9, 1, tzinfo=timezone.utc)
    records = []
    for i in range(n):
        r = _sample(rng, prof, label_noise)
        r["timestamp"] = (t0 + timedelta(minutes=int(rng.integers(0, 60 * 24 * 60)))).isoformat()
        records.append(r)
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def generate_all(n_clients: int = 5, seed: int = 7, test_per_zone: int = 250,
                 label_noise: float = 0.05) -> dict:
    """Write one local dataset per zone plus a campus-wide held-out test set."""
    profiles = ZONE_PROFILES[:n_clients]
    summary = {}
    for i, prof in enumerate(profiles):
        recs = generate_zone(prof, prof.n_samples, seed=seed * 100 + i, label_noise=label_noise)
        write_jsonl(recs, CLIENT_DIR / f"{prof.zone_id}.jsonl")
        pos = sum(r["label"] for r in recs) / len(recs)
        crowd = sum(r["event_type"] == CROWD_EVENT for r in recs) / len(recs)
        summary[prof.zone_id] = {"n": len(recs), "positive_rate": round(pos, 3),
                                 "crowd_share": round(crowd, 3), "zone_name": prof.zone_name}

    # The global test set covers ALL five zones equally, i.e. "the whole campus".
    test = []
    for i, prof in enumerate(ZONE_PROFILES):
        test += generate_zone(prof, test_per_zone, seed=seed * 1000 + 50 + i, label_noise=0.0)
    write_jsonl(test, TEST_PATH)
    summary["_test"] = {"n": len(test)}
    return summary


def list_clients() -> list[str]:
    return sorted(p.stem for p in CLIENT_DIR.glob("*.jsonl"))


def load_client(zone_id: str) -> list[dict]:
    return read_jsonl(CLIENT_DIR / f"{zone_id}.jsonl")


def load_test() -> list[dict]:
    return read_jsonl(TEST_PATH)


def ensure_data(n_clients: int = 5, seed: int = 7) -> None:
    if len(list_clients()) < n_clients or not TEST_PATH.exists():
        generate_all(n_clients=n_clients, seed=seed)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate simulated per-zone datasets")
    ap.add_argument("--clients", type=int, default=5, choices=[3, 4, 5])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--label-noise", type=float, default=0.05)
    a = ap.parse_args()
    for old in CLIENT_DIR.glob("*.jsonl"):
        old.unlink()
    print(json.dumps(generate_all(a.clients, a.seed, label_noise=a.label_noise), indent=2))
