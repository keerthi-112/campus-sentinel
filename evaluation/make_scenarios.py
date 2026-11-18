"""
make_scenarios.py — Generate a labelled set of CCTV-style test clips.

Real people (cut out of the Ultralytics sample photos, so YOLO must genuinely
detect them) are moved over a static camera background along scripted paths.
Because the paths are scripted, the ground truth is exact: we know the frame
each person enters and leaves the restricted zone and when a crowd forms.

    python -m evaluation.make_scenarios            # -> evaluation/clips/*.mp4 + *_gt.json

These clips make the event evaluation reproducible on any laptop. They are a
complement to, not a replacement for, real recorded clips: record 3–4 of your
own and label them with evaluation/mark_events.py, then evaluate both sets.

Ground-truth policy (same as the system's rules):
  * restricted_area_entry — a person's box centre is inside the zone polygon.
    It is a REAL event only if they stay >= dwell_threshold seconds.
  * crowd_formation — >= crowd_count people visible for >= crowd persistence.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
CLIP_DIR = EVAL_DIR / "clips"
SPRITE_DIR = EVAL_DIR / ".sprites"

W, H, FPS = 960, 540, 15
ZONE = [[560, 170], [930, 170], [930, 525], [560, 525]]
DWELL = 4.0
CROWD_COUNT = 5
CROWD_PERSIST = 3.0
CAMERA = "CAM 03 - Lab 2"
ZONE_NAME = "Lab 2 restricted area"


# ----------------------------------------------------------------------
# sprites + background
# ----------------------------------------------------------------------
def load_sprites() -> list[np.ndarray]:
    """Person cut-outs from the Ultralytics sample images (cached)."""
    SPRITE_DIR.mkdir(exist_ok=True)
    cached = sorted(SPRITE_DIR.glob("p*.png"))
    if cached:
        return [cv2.imread(str(p)) for p in cached]
    from ultralytics import YOLO
    from ultralytics.utils import ASSETS

    model = YOLO("yolo11n.pt")
    sprites = []
    for name in ("bus.jpg", "zidane.jpg"):
        img = cv2.imread(str(ASSETS / name))
        for b in model.predict(img, classes=[0], conf=0.5, verbose=False)[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, b)
            crop = img[y1:y2, x1:x2]
            touches_edge = x1 <= 2 or x2 >= img.shape[1] - 2
            # keep full-body, upright people only (h/w >= 1.8), not half-body or cut-off ones
            if crop.shape[0] > 150 and crop.shape[0] / crop.shape[1] >= 1.8 and not touches_edge:
                sprites.append(crop)
    sprites += [cv2.flip(s, 1) for s in sprites]  # mirrored copies for variety
    for i, s in enumerate(sprites):
        cv2.imwrite(str(SPRITE_DIR / f"p{i}.png"), s)
    return sprites


def background(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bg = np.zeros((H, W, 3), np.uint8)
    for y in range(H):  # wall then floor
        v = 185 - y // 12 if y < 210 else 150 - (y - 210) // 9
        bg[y] = (v, v + 3, v + 6)
    cv2.rectangle(bg, (600, 40), (860, 200), (95, 100, 110), -1)    # lab door
    cv2.rectangle(bg, (60, 60), (300, 180), (120, 130, 140), -1)    # notice board
    noise = rng.normal(0, 3, bg.shape)
    return np.clip(bg + noise, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
# actors: piecewise-linear paths of the box CENTRE (x, y) over time
# ----------------------------------------------------------------------
def at(path: list[tuple[float, float, float]], t: float):
    """path = [(t, x, y), ...]; returns (x, y) or None when off-stage."""
    if t < path[0][0] or t > path[-1][0]:
        return None
    for (t0, x0, y0), (t1, x1, y1) in zip(path, path[1:]):
        if t0 <= t <= t1:
            a = 0 if t1 == t0 else (t - t0) / (t1 - t0)
            return x0 + a * (x1 - x0), y0 + a * (y1 - y0)
    return path[-1][1], path[-1][2]


def inside(x, y) -> bool:
    return cv2.pointPolygonTest(np.array(ZONE, np.int32), (float(x), float(y)), False) >= 0


def render(actors: list[dict], duration: float, path: Path, seed: int) -> int:
    sprites = load_sprites()
    bg = background(seed)
    rng = np.random.default_rng(seed)
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    n = int(duration * FPS)
    for i in range(n):
        t = i / FPS
        frame = bg.copy()
        # draw back-to-front so lower (closer) people overlap correctly
        placed = []
        for a in actors:
            p = at(a["path"], t)
            if p is not None:
                placed.append((p[1], p[0], a))
        for cy, cx, a in sorted(placed, key=lambda p: p[0]):
            spr = sprites[a["sprite"] % len(sprites)]
            h = int(a.get("height", 210))
            w = max(20, int(spr.shape[1] * h / spr.shape[0]))
            s = cv2.resize(spr, (w, h))
            x1, y1 = int(cx - w / 2), int(cy - h / 2)
            xa, ya, xb, yb = max(x1, 0), max(y1, 0), min(x1 + w, W), min(y1 + h, H)
            if xb > xa and yb > ya:
                frame[ya:yb, xa:xb] = s[ya - y1:yb - y1, xa - x1:xb - x1]
        frame = np.clip(frame + rng.normal(0, 2, frame.shape), 0, 255).astype(np.uint8)  # sensor noise
        out.write(frame)
    out.release()
    return n


def ground_truth(actors: list[dict], duration: float) -> list[dict]:
    """Exact events from the scripted paths (sampled at frame rate)."""
    events = []
    times = np.arange(0, duration, 1 / FPS)
    for a in actors:  # restricted entries, per person
        start = None
        for t in list(times) + [duration + 1]:
            p = at(a["path"], t) if t <= duration else None
            ins = p is not None and inside(*p)
            if ins and start is None:
                start = t
            elif not ins and start is not None:
                if t - start >= DWELL:
                    events.append({"type": "restricted_area_entry", "start": round(start, 2),
                                   "end": round(min(t, duration), 2), "person": a["name"]})
                start = None
    start = None  # crowd: >= CROWD_COUNT people visible
    for t in list(times) + [duration + 1]:
        cnt = sum(1 for a in actors if t <= duration and at(a["path"], t) is not None)
        if cnt >= CROWD_COUNT and start is None:
            start = t
        elif cnt < CROWD_COUNT and start is not None:
            if t - start >= CROWD_PERSIST:
                events.append({"type": "crowd_formation", "start": round(start, 2),
                               "end": round(min(t, duration), 2)})
            start = None
    return events


# ----------------------------------------------------------------------
# scenarios
# ----------------------------------------------------------------------
def person(name, sprite, path, height=210):
    return {"name": name, "sprite": sprite, "path": path, "height": height}


def scenarios() -> dict:
    s = {}
    # positives: enter the zone and stay
    s["enter_stay_1"] = (14, [person("A", 0, [(0.5, -40, 330), (3, 700, 330), (10, 720, 340), (13, 1000, 340)])])
    s["enter_stay_2"] = (14, [person("A", 2, [(0, 1000, 420), (2, 820, 410), (9, 800, 400), (11.5, 1000, 400)])])
    s["enter_stay_3"] = (16, [person("A", 4, [(1, -40, 300), (5, 650, 360), (13, 700, 360), (15.5, 300, 300)], 230)])
    s["stay_while_other_passes"] = (15, [
        person("A", 1, [(0.5, -40, 330), (3, 760, 350), (12, 780, 350), (14.5, 1000, 350)]),
        person("B", 3, [(4, -40, 300), (11, 480, 300), (13, 480, 300)])])
    # negatives: walk past / brief entry
    s["walk_past_1"] = (12, [person("A", 0, [(0.5, -40, 330), (9, 520, 330), (11.5, -40, 330)])])
    s["walk_past_2"] = (12, [person("A", 3, [(0, 400, -100), (5, 450, 340), (11, 420, 640)])])
    s["cross_through_zone"] = (10, [person("A", 1, [(0.5, -40, 360), (8, 1000, 360)])])
    s["brief_entry_1"] = (11, [person("A", 2, [(0.5, -40, 330), (3, 640, 330), (5, 650, 330), (7.5, -40, 330)])])
    s["brief_entry_2"] = (11, [person("A", 4, [(1, 1000, 400), (2.5, 870, 400), (3.5, 870, 400), (4.5, 1000, 400)], 230)])
    s["empty_scene"] = (8, [])
    # crowds (left of the zone, so they don't also trigger restricted entry)
    def group(n, t_in, t_out, cx=270, cy=360, spread=230):
        actors = []
        for k in range(n):
            x = cx - spread / 2 + spread * (k % 4) / 3 + (k // 4) * spread / 6
            y = cy - 50 + 95 * (k // 4) + (k % 2) * 12
            sx = -60 if k % 2 == 0 else 560
            actors.append(person(f"G{k}", k, [(t_in + 0.3 * k, sx, y), (t_in + 2 + 0.3 * k, x, y),
                                               (t_out, x, y), (t_out + 2, sx, y)], 170))
        return actors
    s["crowd_gather_1"] = (16, group(6, 0.5, 12))
    s["crowd_gather_2"] = (16, group(7, 0.5, 13, spread=260))
    s["crowd_gather_3"] = (15, group(8, 0.5, 12, spread=300))
    s["small_group_1"] = (14, group(3, 0.5, 11))
    s["small_group_2"] = (14, group(4, 0.5, 11))
    return s


def jitter(actors: list[dict], rng: np.random.Generator, n_sprites: int) -> list[dict]:
    """A different-looking take of the same scenario: shifted timing and positions,
    other people. Ground truth is recomputed from the new paths, so it stays exact."""
    out = []
    dt = rng.uniform(-0.3, 0.8)
    shift = rng.integers(0, n_sprites)
    for a in actors:
        dx, dy = rng.uniform(-35, 35), rng.uniform(-25, 25)
        stretch = rng.uniform(0.85, 1.25)  # how long people stay
        t0 = a["path"][0][0]
        path = [(max(0.0, t0 + dt + (t - t0) * stretch), x + dx, y + dy) for t, x, y in a["path"]]
        out.append({**a, "path": path, "sprite": (a["sprite"] + shift) % n_sprites,
                    "height": a["height"] * rng.uniform(0.9, 1.1)})
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Generate labelled scenario clips")
    ap.add_argument("--out", default=str(CLIP_DIR))
    ap.add_argument("--variants", type=int, default=0,
                    help="instead of the base set, write N jittered takes of every scenario (training data)")
    ap.add_argument("--seed", type=int, default=100)
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_sprites = len(load_sprites())
    rng = np.random.default_rng(a.seed)
    jobs = []
    for name, (duration, actors) in scenarios().items():
        if a.variants:
            for v in range(1, a.variants + 1):
                jobs.append((f"{name}_v{v}", duration + 2, jitter(actors, rng, n_sprites)))
        else:
            jobs.append((name, duration, actors))
    print(f"Writing {len(jobs)} scenario clips to {out_dir}")
    for i, (name, duration, actors) in enumerate(jobs):
        path = out_dir / f"{name}.mp4"
        render(actors, duration, path, seed=i + (1000 if a.variants else 0))
        gt = {
            "video": path.name, "fps": FPS, "duration": duration, "synthetic": True,
            "camera_name": CAMERA, "zone_name": ZONE_NAME, "restricted_region": ZONE,
            "dwell_threshold": DWELL, "crowd_count": CROWD_COUNT, "crowd_persist": CROWD_PERSIST,
            "events": ground_truth(actors, duration),
        }
        (out_dir / f"{name}_gt.json").write_text(json.dumps(gt, indent=2), encoding="utf-8")
        kinds = ", ".join(e["type"].split("_")[0] for e in gt["events"]) or "none"
        print(f"  {name:26s} {duration:4.0f}s  events: {kinds}")


if __name__ == "__main__":
    main()
