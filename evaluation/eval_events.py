"""
eval_events.py — Restricted-area and crowd event evaluation.

Runs the vision track's own modules (PersonDetector → CentroidTracker →
ZoneMonitor / CrowdMonitor → compute_features) on labelled clips, then the
federated event classifier, and scores the alerts against ground truth.

    python -m evaluation.eval_events                               # scenario clips, YOLO26n
    python -m evaluation.eval_events --model yolo11n.pt
    python -m evaluation.eval_events --clips path/to/your/clips    # your own clips + *_gt.json

Each clip needs <name>.mp4 + <name>_gt.json (make_scenarios.py or mark_events.py).

Timing: the vision modules read time.time() for dwell timers. Here they are
given the clip's own clock (frame index / fps), so results are identical on a
fast or a slow laptop and a 15 s clip takes as long as the CPU needs. The
vision code itself is not modified.

Metrics
  restricted_area_entry  precision, recall, F1, false alerts per minute,
                         detection→alert time (zone entry → alert), alert lag
                         after the dwell threshold, duplicates
  crowd_formation        per-clip accuracy, precision, recall, F1 (+ event P/R)
  both                   the same after the federated classifier's filter
  performance            YOLO ms/frame, pipeline ms/frame, FPS
"""

from __future__ import annotations

import argparse
import json
import sys
import time as _time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import vision.crowd as _vcrowd  # noqa: E402
import vision.zones as _vzones  # noqa: E402
from federated.classifier import EventClassifier  # noqa: E402
from vision.crowd import CrowdConfig, CrowdMonitor  # noqa: E402
from vision.detect import PersonDetector  # noqa: E402
from vision.features import compute_features  # noqa: E402
from vision.tracker import CentroidTracker  # noqa: E402
from vision.zones import ZoneConfig, ZoneMonitor  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
RESULTS = EVAL_DIR / "results"
GRACE = 1.5  # an alert up to 1.5 s after the person left still belongs to that entry
ZONE, CROWD = "restricted_area_entry", "crowd_formation"


class VideoClock:
    """Stand-in for the `time` module inside vision.zones / vision.crowd."""

    def __init__(self) -> None:
        self.t = 1_763_000_000.0

    def time(self) -> float:
        return self.t


def process_clip(video: Path, gt: dict, detector: PersonDetector, clf: EventClassifier,
                 export: list | None) -> dict:
    clock = VideoClock()
    _vzones.time, _vcrowd.time = clock, clock  # give the vision modules video time
    t0 = clock.t
    zone = ZoneConfig(zone_name=gt["zone_name"], polygon=[tuple(p) for p in gt["restricted_region"]],
                      dwell_threshold=gt["dwell_threshold"], tier=gt.get("tier", 3))
    zm = ZoneMonitor([zone])
    cm = CrowdMonitor(CrowdConfig(threshold_count=gt["crowd_count"], dwell_threshold=gt.get("crowd_persist", 3.0)))
    tracker = CentroidTracker()
    cap = cv2.VideoCapture(str(video))
    fps = gt.get("fps") or cap.get(cv2.CAP_PROP_FPS) or 15.0
    alerts, det_ms, pipe_ms = [], [], []
    prev, idx, next_sample = {}, 0, 0.0
    cam = gt["camera_name"]

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        clock.t = t0 + t
        h, w = frame.shape[:2]
        a = _time.perf_counter()
        dets = detector.detect(frame)
        b = _time.perf_counter()
        tracks = tracker.update(dets)
        zc = zm.update(tracks, cam)
        density = round(sum(d.area for d in dets) / float(w * h), 4)
        cc = cm.update(len(dets), density, cam, zone_name=gt["zone_name"])
        for cand in list(zc) + list(cc):
            is_zone = hasattr(cand, "person_id")
            dwell = cand.dwell_seconds  # zone: time in zone; crowd: how long the crowd has persisted
            mconf = cand.max_confidence if is_zone else max((d.confidence for d in dets), default=0.0)
            fv = compute_features(dets, tracks, zone, cam, (h, w), dwell_seconds=dwell,
                                  max_confidence=mconf, prev_centers=prev)
            et = ZONE if is_zone else CROWD
            d = clf.decide(fv, et)
            alerts.append({"type": et, "t": round(t, 2), "confidence": round(float(mconf), 3),
                           "dwell": round(float(dwell), 2), "escalate": bool(d["escalate"]),
                           "score": d["score"], "features": fv})
        # Labelled samples for federated learning, every 0.5 s while a candidate
        # condition holds (someone in the zone past the dwell threshold, or a
        # persisting crowd), i.e. what the classifier sees in production.
        if export is not None and t >= next_sample:
            next_sample = t + 0.5
            zdwell, zconf = zm.get_zone_dwell_info(zone, tracks)
            crowd_start = getattr(cm, "_crowd_start_time", None)
            cdwell = (clock.t - crowd_start) if crowd_start else 0.0
            for et, active, dw, mc in (
                (ZONE, zdwell >= gt["dwell_threshold"], zdwell, zconf),
                (CROWD, len(dets) >= gt["crowd_count"] and cdwell >= gt.get("crowd_persist", 3.0), cdwell,
                 max((x.confidence for x in dets), default=0.0)),
            ):
                if not active:
                    continue
                fv = compute_features(dets, tracks, zone, cam, (h, w), dwell_seconds=dw,
                                      max_confidence=mc, prev_centers=prev)
                label = int(any(e["type"] == et and e["start"] <= t <= e["end"] + GRACE
                                for e in positive_events(gt)))
                export.append({**fv, "event_type": et, "label": label, "label_source": "ground_truth",
                               "clip": video.name})
        prev = {pid: p.center for pid, p in tracks.items()}
        c = _time.perf_counter()
        det_ms.append((b - a) * 1000)
        pipe_ms.append((c - a) * 1000)
        idx += 1
    cap.release()
    return {"alerts": alerts, "frames": idx, "duration": idx / fps,
            "det_ms": det_ms, "pipe_ms": pipe_ms}


def positive_events(gt: dict) -> list[dict]:
    """Restricted entries count as real only if they lasted >= the dwell threshold."""
    out = []
    for e in gt["events"]:
        if e["type"] == ZONE and e["end"] - e["start"] < gt["dwell_threshold"]:
            continue
        out.append(e)
    return out


def match(alerts: list[dict], events: list[dict], etype: str, dwell: float) -> dict:
    evs = [e for e in events if e["type"] == etype]
    matched = [False] * len(evs)
    tp = fp = dup = 0
    latency, lag = [], []
    for a in sorted((x for x in alerts if x["type"] == etype), key=lambda x: x["t"]):
        hit = None
        for i, e in enumerate(evs):
            if e["start"] <= a["t"] <= e["end"] + GRACE:
                hit = i
                if not matched[i]:
                    break
        if hit is None:
            fp += 1
        elif matched[hit]:
            dup += 1
        else:
            matched[hit] = True
            tp += 1
            latency.append(a["t"] - evs[hit]["start"])
            lag.append(a["t"] - evs[hit]["start"] - dwell)
    return {"tp": tp, "fp": fp, "fn": matched.count(False), "dup": dup, "latency": latency, "lag": lag}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else (1.0 if not fn else 0.0)
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def summarise(per_clip: list[dict], filtered: bool) -> dict:
    key = "alerts_fl" if filtered else "alerts"
    minutes = sum(c["duration"] for c in per_clip) / 60
    out = {}
    for etype in (ZONE, CROWD):
        tp = fp = fn = dup = 0
        lat, lag = [], []
        for c in per_clip:
            m = match(c[key], c["positives"], etype, c["dwell"] if etype == ZONE else c["crowd_persist"])
            tp, fp, fn, dup = tp + m["tp"], fp + m["fp"], fn + m["fn"], dup + m["dup"]
            lat += m["latency"]
            lag += m["lag"]
        p, r, f = prf(tp, fp, fn)
        out[etype] = {"tp": tp, "fp": fp, "fn": fn, "duplicates": dup, "precision": p, "recall": r, "f1": f,
                      "false_alerts_per_min": round(fp / minutes, 3) if minutes else 0.0,
                      "detection_to_alert_s_mean": round(float(np.mean(lat)), 2) if lat else None,
                      "detection_to_alert_s_p95": round(float(np.percentile(lat, 95)), 2) if lat else None,
                      "lag_after_threshold_s_mean": round(float(np.mean(lag)), 2) if lag else None}
    # per-clip classification (crowd accuracy / F1 per test video)
    for etype, name in ((CROWD, "crowd_clip_level"), (ZONE, "restricted_clip_level")):
        tp = fp = fn = tn = 0
        for c in per_clip:
            truth = any(e["type"] == etype for e in c["positives"])
            pred = any(a["type"] == etype for a in c[key])
            tp += truth and pred
            fp += (not truth) and pred
            fn += truth and not pred
            tn += (not truth) and not pred
        p, r, f = prf(tp, fp, fn)
        out[name] = {"clips": len(per_clip), "accuracy": round((tp + tn) / max(len(per_clip), 1), 4),
                     "precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Event-level evaluation of the vision pipeline")
    ap.add_argument("--clips", default=str(EVAL_DIR / "clips"))
    ap.add_argument("--model", default="yolo26n.pt")
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default=None, help="name for the results file (default: model + clip folder)")
    ap.add_argument("--export-fl", default=None,
                    help="write ground-truth-labelled feature samples (JSONL) for federated learning")
    a = ap.parse_args()

    clips = sorted(p for p in Path(a.clips).glob("*.mp4") if (p.with_name(p.stem + "_gt.json")).exists())
    if not clips:
        raise SystemExit(f"No <clip>.mp4 + <clip>_gt.json pairs in {a.clips}. "
                         "Run `python -m evaluation.make_scenarios` or label your clips with mark_events.py.")
    detector = PersonDetector(model_name=a.model, confidence_threshold=a.conf, device=a.device)
    clf = EventClassifier()
    export = [] if a.export_fl else None
    per_clip = []
    print(f"Evaluating {len(clips)} clips with {a.model} (conf {a.conf}); FL model "
          f"{clf.info().get('model_version') or 'none'}")
    for v in clips:
        gt = json.loads(v.with_name(v.stem + "_gt.json").read_text(encoding="utf-8"))
        r = process_clip(v, gt, detector, clf, export)
        r.update(clip=v.stem, positives=positive_events(gt), dwell=gt["dwell_threshold"],
                 crowd_persist=gt.get("crowd_persist", 3.0), synthetic=gt.get("synthetic", False))
        r["alerts_fl"] = [x for x in r["alerts"] if x["escalate"]]
        per_clip.append(r)
        n_true = len(r["positives"])
        kinds = lambda xs: ",".join(sorted({x["type"].split("_")[0] for x in xs})) or "-"
        print(f"  {v.stem:26s} truth={kinds(r['positives']):18s} alerts={len(r['alerts'])} "
              f"({kinds(r['alerts'])})  after FL={len(r['alerts_fl'])}  [{n_true} true]")

    det = np.concatenate([c["det_ms"] for c in per_clip])
    pipe = np.concatenate([c["pipe_ms"] for c in per_clip])
    result = {
        "model": a.model, "conf": a.conf, "clips_dir": str(a.clips), "n_clips": len(per_clip),
        "synthetic": all(c["synthetic"] for c in per_clip),
        "total_minutes": round(sum(c["duration"] for c in per_clip) / 60, 2),
        "fl_model_version": clf.info().get("model_version"),
        "vision_only": summarise(per_clip, filtered=False),
        "with_fl_classifier": summarise(per_clip, filtered=True),
        "performance": {"yolo_ms_mean": round(float(det.mean()), 1), "yolo_ms_p95": round(float(np.percentile(det, 95)), 1),
                        "pipeline_ms_mean": round(float(pipe.mean()), 1),
                        "fps": round(1000 / float(pipe.mean()), 1), "frames": int(len(pipe))},
        "per_clip": [{"clip": c["clip"], "truth": [e["type"] for e in c["positives"]],
                      "alerts": [{k: x[k] for k in ("type", "t", "confidence", "dwell", "escalate", "score")}
                                 for x in c["alerts"]]} for c in per_clip],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f"{Path(a.model).stem}_{Path(a.clips).name}"
    out = RESULTS / f"events_{tag}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    v, f = result["vision_only"], result["with_fl_classifier"]
    print("\nRestricted-area entry (event level)       vision only | + FL classifier")
    for k in ("precision", "recall", "f1", "false_alerts_per_min", "detection_to_alert_s_mean", "duplicates"):
        print(f"  {k:30s} {str(v[ZONE][k]):>12s} | {str(f[ZONE][k])}")
    print("Crowd formation (per clip)")
    for k in ("accuracy", "precision", "recall", "f1"):
        print(f"  {k:30s} {str(v['crowd_clip_level'][k]):>12s} | {str(f['crowd_clip_level'][k])}")
    p = result["performance"]
    print(f"Performance: YOLO {p['yolo_ms_mean']} ms/frame, pipeline {p['pipeline_ms_mean']} ms/frame ({p['fps']} FPS)")
    print(f"-> {out.relative_to(REPO)}")
    if export is not None:
        Path(a.export_fl).parent.mkdir(parents=True, exist_ok=True)
        with open(a.export_fl, "w", encoding="utf-8") as fh:
            for rec in export:
                fh.write(json.dumps(rec) + "\n")
        pos = sum(r["label"] for r in export)
        print(f"-> {a.export_fl}: {len(export)} labelled feature samples ({pos} positive)")


if __name__ == "__main__":
    main()
