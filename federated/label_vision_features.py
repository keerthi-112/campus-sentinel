"""
label_vision_features.py — Turn REAL feature vectors exported by the vision
track into per-zone federated training data.

    python -m vision.run_demo --source clip.mp4 --type file --export-features data/features.jsonl
    python -m federated.label_vision_features data/features.jsonl
    python -m federated.label_vision_features data/features.jsonl --labels my_labels.csv

vision/ writes one record per frame per zone, so this script:
  1. keeps only frames that are actual event candidates (someone dwelling in
     the zone past the threshold, or a crowd-sized count),
  2. down-samples consecutive near-identical frames (one record per --stride s),
  3. labels them — from a CSV you annotated by hand (timestamp_prefix,label),
     or, if none is given, with a transparent weak-labelling rule,
  4. writes federated/data/clients/<zone>.jsonl, one file per zone,
     ready for simulate.py / run_flower.py.

Weak labels are a stand-in until the team hand-labels test clips — say so in
the report if you use them.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from .data import CLIENT_DIR, write_jsonl
from .features import CROWD_EVENT, ZONE_EVENT


def weak_label(r: dict) -> int:
    """Rule used only when no human labels exist."""
    if r["event_type"] == CROWD_EVENT:
        return int(r["person_count"] >= 10 or (r["density"] >= 0.45 and r["motion_score"] >= 0.06))
    return int(r["zone_dwell_seconds"] >= 8.0 and r["max_confidence"] >= 0.65 and r["motion_score"] < 0.06)


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") or "zone"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("features_jsonl")
    ap.add_argument("--labels", help="CSV with columns timestamp_prefix,label")
    ap.add_argument("--dwell", type=float, default=4.0, help="zone dwell threshold used by vision")
    ap.add_argument("--crowd-count", type=int, default=5)
    ap.add_argument("--stride", type=float, default=1.0, help="seconds between kept records per zone")
    a = ap.parse_args()

    manual = {}
    if a.labels:
        with open(a.labels, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                manual[row["timestamp_prefix"]] = int(row["label"])

    per_zone: dict[str, list[dict]] = defaultdict(list)
    last_kept: dict[str, float] = {}
    total = 0
    for line in Path(a.features_jsonl).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        total += 1
        r = json.loads(line)
        if r["zone_dwell_seconds"] >= a.dwell:
            r["event_type"] = ZONE_EVENT
        elif r["person_count"] >= a.crowd_count:
            r["event_type"] = CROWD_EVENT
        else:
            continue
        key = slug(f"{r['camera_name']}_{r['zone_name']}")
        t = datetime.fromisoformat(r["timestamp"].replace("Z", "+00:00")).timestamp()
        if key in last_kept and t - last_kept[key] < a.stride:
            continue
        last_kept[key] = t
        label = next((v for k, v in manual.items() if r["timestamp"].startswith(k)), None)
        r["label"] = weak_label(r) if label is None else label
        r["label_source"] = "weak_rule" if label is None else "manual"
        per_zone[key].append(r)

    for key, recs in per_zone.items():
        write_jsonl(recs, CLIENT_DIR / f"{key}.jsonl")
        pos = sum(x["label"] for x in recs)
        print(f"{key:40s} {len(recs):5d} records  ({pos} positive)")
    print(f"read {total} frame records -> {sum(map(len, per_zone.values()))} candidates "
          f"in {len(per_zone)} zone file(s) under {CLIENT_DIR}")


if __name__ == "__main__":
    main()
