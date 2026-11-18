"""
eval_detection.py — Person-detection metrics and the YOLO26n vs YOLO11n comparison
(template §14 "Person detection" and §21 "compare YOLO11n and keep the better").

    python -m evaluation.eval_detection                       # COCO128 persons, both models
    python -m evaluation.eval_detection --data my_frames.yaml # your own labelled frames

Reports, per model, for the PERSON class only:
  precision, recall, mAP@0.5, mAP@0.5:0.95   (Ultralytics validator)
  inference ms/frame and FPS on this machine  (real camera-sized frames)

Caveat for the report: COCO128 images come from COCO train2017, which these
pretrained models have seen, so the absolute numbers are optimistic. It is a
fair *comparison* between the two models. For absolute numbers on your own
cameras, label ~50 frames from your clips (YOLO format, class 0 = person, e.g.
with CVAT or Label Studio) and pass their dataset YAML with --data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
EVAL = REPO / "evaluation"
RES = EVAL / "results"


def speed(model, frames: list[np.ndarray], device: str) -> dict:
    for f in frames[:3]:  # warm-up
        model.predict(f, classes=[0], conf=0.4, device=device, verbose=False)
    ts = []
    for f in frames:
        t = time.perf_counter()
        model.predict(f, classes=[0], conf=0.4, device=device, verbose=False)
        ts.append((time.perf_counter() - t) * 1000)
    ts = np.array(ts)
    return {"ms_mean": round(float(ts.mean()), 1), "ms_p95": round(float(np.percentile(ts, 95)), 1),
            "fps": round(1000 / float(ts.mean()), 1), "frames": len(ts),
            "frame_size": f"{frames[0].shape[1]}x{frames[0].shape[0]}"}


def sample_frames(n: int = 40) -> list[np.ndarray]:
    """Camera-sized frames: from the evaluation clips if present, else the sample photos."""
    frames = []
    for clip in sorted((EVAL / "clips").glob("*.mp4")):
        cap = cv2.VideoCapture(str(clip))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for k in range(3):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (k + 1) / 4))
            ok, f = cap.read()
            if ok:
                frames.append(f)
        cap.release()
        if len(frames) >= n:
            break
    if not frames:
        from ultralytics.utils import ASSETS
        frames = [cv2.resize(cv2.imread(str(ASSETS / p)), (960, 540)) for p in ("bus.jpg", "zidane.jpg")] * 20
    return frames[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description="Person detection precision/recall/mAP + speed")
    ap.add_argument("--models", nargs="+", default=["yolo26n.pt", "yolo11n.pt"])
    ap.add_argument("--data", default="coco128.yaml", help="Ultralytics dataset YAML (default: COCO128, auto-download)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    from ultralytics import YOLO, settings

    settings.update({"datasets_dir": str(EVAL / "datasets")})
    frames = sample_frames()
    rows = []
    for m in a.models:
        model = YOLO(m)
        print(f"\n== {m}: validating on {a.data} (person class) ...")
        v = model.val(data=a.data, classes=[0], imgsz=a.imgsz, device=a.device, batch=8,
                      plots=False, verbose=False, project=str(RES / "_val"), name=Path(m).stem, exist_ok=True)
        sp = speed(model, frames, a.device)
        rows.append({
            "model": m, "data": a.data,
            "precision": round(float(v.box.mp), 4), "recall": round(float(v.box.mr), 4),
            "mAP50": round(float(v.box.map50), 4), "mAP50_95": round(float(v.box.map), 4),
            **{f"speed_{k}": val for k, val in sp.items()},
        })
        print(f"   P={rows[-1]['precision']}  R={rows[-1]['recall']}  mAP50={rows[-1]['mAP50']}  "
              f"mAP50-95={rows[-1]['mAP50_95']}  {sp['ms_mean']} ms/frame ({sp['fps']} FPS)")

    best = max(rows, key=lambda r: (r["mAP50_95"], -r["speed_ms_mean"]))
    out = {"device": a.device, "imgsz": a.imgsz, "results": rows, "selected": best["model"],
           "rule": "highest person mAP50-95; ties broken by speed (template §21)"}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "detection.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSelected model: {best['model']}  -> evaluation/results/detection.json")


if __name__ == "__main__":
    sys.exit(main())
