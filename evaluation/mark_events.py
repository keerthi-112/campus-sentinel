"""
mark_events.py — Label your own clips for the event evaluation.

    python -m evaluation.mark_events --video my_clips/lab_door.mp4 --camera "CAM 03" --zone-name "Lab 2 restricted area"

1. Click the corners of the restricted zone on the first frame, then press ENTER
   (or pass --polygon "x,y x,y x,y ..." to skip this step).
2. The clip plays. Press:
     R      a person ENTERS the zone (press R again when they LEAVE)
     C      a crowd FORMS (press C again when it breaks up)
     SPACE  pause / play        A / D   one frame back / forward (while paused)
     U      undo the last label  Q      save and quit
3. Writes <video>_gt.json next to the clip, in the same format as
   make_scenarios.py, so `python -m evaluation.eval_events --clips my_clips` works.

Label every zone entry, also short ones: entries shorter than the dwell
threshold are scored as "should NOT alert" (true negatives / false alarms).
Tip: record 3–4 clips of 20–60 s: someone entering and staying, someone walking
past, someone stepping in briefly, and a group of 5+ gathering.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def pick_polygon(frame) -> list[list[int]]:
    pts: list[list[int]] = []
    win = "Click zone corners, ENTER = done, N = restart"

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append([int(x), int(y)])

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)
    while True:
        view = frame.copy()
        if pts:
            cv2.polylines(view, [np.array(pts, np.int32)], len(pts) > 2, (0, 0, 255), 2)
            for p in pts:
                cv2.circle(view, tuple(p), 5, (255, 255, 255), -1)
        cv2.imshow(win, view)
        k = cv2.waitKey(30) & 0xFF
        if k in (13, 10) and len(pts) >= 3:
            break
        if k in (ord("n"), ord("N")):
            pts.clear()
    cv2.destroyWindow(win)
    return pts


def main() -> None:
    ap = argparse.ArgumentParser(description="Label restricted-zone entries and crowds in a clip")
    ap.add_argument("--video", required=True)
    ap.add_argument("--camera", default="CAM 03")
    ap.add_argument("--zone-name", default="Restricted area")
    ap.add_argument("--polygon", default=None, help='"x,y x,y x,y ..." in frame pixels')
    ap.add_argument("--dwell", type=float, default=4.0)
    ap.add_argument("--crowd-count", type=int, default=5)
    ap.add_argument("--crowd-persist", type=float, default=3.0)
    a = ap.parse_args()

    video = Path(a.video)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, first = cap.read()
    if not ok:
        raise SystemExit(f"cannot read {video}")
    poly = ([[int(v) for v in p.split(",")] for p in a.polygon.split()] if a.polygon else pick_polygon(first))

    events: list[dict] = []
    open_ev: dict[str, float] = {}
    idx, paused = 0, False
    frame = first
    win = "R entry  C crowd  SPACE pause  A/D step  U undo  Q save"
    while True:
        t = idx / fps
        view = frame.copy()
        cv2.polylines(view, [np.array(poly, np.int32)], True, (0, 0, 255), 2)
        status = f"t={t:6.2f}s  frame {idx}/{n_frames}  " + ("PAUSED  " if paused else "") + \
                 " ".join(f"[{k} open since {v:.1f}s]" for k, v in open_ev.items())
        cv2.rectangle(view, (0, 0), (view.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(view, status, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(view, f"labels: {len(events)}", (8, view.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow(win, view)
        k = cv2.waitKey(1 if paused else max(1, int(1000 / fps))) & 0xFF
        if k in (ord("q"), ord("Q")):
            break
        if k == ord(" "):
            paused = not paused
        for key, etype in ((ord("r"), "restricted_area_entry"), (ord("c"), "crowd_formation")):
            if k in (key, key - 32):
                if etype in open_ev:
                    events.append({"type": etype, "start": round(open_ev.pop(etype), 2), "end": round(t, 2)})
                    print(f"  {etype}: {events[-1]['start']}s -> {events[-1]['end']}s")
                else:
                    open_ev[etype] = t
        if k in (ord("u"), ord("U")) and events:
            print(f"  undo {events.pop()}")
        step = 0
        if not paused:
            step = 1
        elif k in (ord("d"), ord("D")):
            step = 1
        elif k in (ord("a"), ord("A")):
            step = -1
        if step:
            idx = max(0, idx + step)
            if step < 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, nxt = cap.read()
            if not ok:
                paused, idx = True, idx - 1
            else:
                frame = nxt
    duration = idx / fps
    for etype, start in open_ev.items():  # close anything still open at the end
        events.append({"type": etype, "start": round(start, 2), "end": round(duration, 2)})
    cap.release()
    cv2.destroyAllWindows()

    gt = {"video": video.name, "fps": fps, "duration": round(n_frames / fps, 2), "synthetic": False,
          "camera_name": a.camera, "zone_name": a.zone_name, "restricted_region": poly,
          "dwell_threshold": a.dwell, "crowd_count": a.crowd_count, "crowd_persist": a.crowd_persist,
          "events": sorted(events, key=lambda e: e["start"])}
    out = video.with_name(video.stem + "_gt.json")
    out.write_text(json.dumps(gt, indent=2), encoding="utf-8")
    print(f"saved {len(events)} labels -> {out}")


if __name__ == "__main__":
    main()
