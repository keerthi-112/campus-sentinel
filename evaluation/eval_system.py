"""
eval_system.py — End-to-end latency and resource usage of the RUNNING system.

Start the system (scripts/run_all.ps1), start a camera on a clip or webcam,
then in a second terminal:

    python -m evaluation.eval_system --seconds 120
    python -m evaluation.eval_system --seconds 120 --start-camera 2     # also starts camera id 2

It samples GET /api/system/metrics every 2 s and writes
evaluation/results/system.json with:
  CPU % and memory of the backend process and the machine (mean / max)
  frame processing time and FPS per running camera
  alert decision latency (vision candidate → FL classifier → stored)
  report latency (incident → report ready), split into queue wait and generation
  end-to-end estimate: zone entry → alert (dwell threshold + frame time + decision) → report

Walk into the zone a few times during the run so incidents and reports are measured.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import httpx

RES = Path(__file__).resolve().parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password", default="admin123")
    ap.add_argument("--seconds", type=int, default=120)
    ap.add_argument("--start-camera", type=int, default=None)
    ap.add_argument("--dwell", type=float, default=None, help="dwell threshold for the end-to-end estimate")
    a = ap.parse_args()

    c = httpx.Client(base_url=a.url, timeout=10)
    tok = c.post("/api/auth/login", json={"username": a.user, "password": a.password}).json()["token"]
    c.headers["Authorization"] = f"Bearer {tok}"
    if a.start_camera is not None:
        c.post(f"/api/cameras/{a.start_camera}/start")
    cams = {x["id"]: x for x in c.get("/api/cameras").json()}

    samples = []
    end = time.time() + a.seconds
    print(f"Sampling {a.url}/api/system/metrics for {a.seconds}s ...")
    while time.time() < end:
        m = c.get("/api/system/metrics").json()
        samples.append(m)
        cams_txt = ", ".join(f"{x['camera_name']} {x['fps']} fps / {x['frame_ms']} ms" for x in m["cameras"]) or "no camera running"
        print(f"  cpu {m['process']['cpu_percent']:5.1f}%  rss {m['process']['rss_mb']:6.0f} MB  {cams_txt}", end="\r")
        time.sleep(2)
    print()

    last = samples[-1]
    proc_cpu = [s["process"]["cpu_percent"] for s in samples[1:]] or [0]
    sys_cpu = [s["system"]["cpu_percent"] for s in samples[1:]] or [0]
    rss = [s["process"]["rss_mb"] for s in samples]
    frame = [cam["frame_ms"] for s in samples for cam in s["cameras"] if cam.get("frame_ms")]
    fps = [cam["fps"] for s in samples for cam in s["cameras"] if cam.get("fps")]
    lat = last["latency"]
    dwell = a.dwell or (min((cm["dwell_threshold"] for cm in cams.values()), default=4.0))
    frame_s = (statistics.mean(frame) / 1000) if frame else 0.0
    decision_s = (lat["decision_ms"]["mean"] / 1000) if lat.get("decision_ms") else 0.0
    report_s = lat["report_total_s"]["mean"] if lat.get("report_total_s") else None

    out = {
        "seconds": a.seconds, "samples": len(samples),
        "machine": {"cpu_count": last["system"]["cpu_count"], "ram_total_gb": last["system"]["ram_total_gb"]},
        "backend_cpu_percent": {"mean": round(statistics.mean(proc_cpu), 1), "max": max(proc_cpu)},
        "machine_cpu_percent": {"mean": round(statistics.mean(sys_cpu), 1), "max": max(sys_cpu)},
        "backend_rss_mb": {"mean": round(statistics.mean(rss)), "max": max(rss)},
        "frame_ms": {"mean": round(statistics.mean(frame), 1), "max": max(frame)} if frame else None,
        "fps": {"mean": round(statistics.mean(fps), 1)} if fps else None,
        "latency": lat,
        "end_to_end_estimate_s": {
            "zone_entry_to_alert": round(dwell + frame_s + decision_s, 2),
            "alert_to_report": report_s,
            "zone_entry_to_report": round(dwell + frame_s + decision_s + report_s, 2) if report_s else None,
            "note": "zone entry → alert includes the deliberate dwell threshold "
                    f"({dwell}s) that suppresses single-frame false alerts",
        },
    }
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "system.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    e = out["end_to_end_estimate_s"]
    print(f"backend CPU {out['backend_cpu_percent']['mean']}% (max {out['backend_cpu_percent']['max']}), "
          f"RAM {out['backend_rss_mb']['mean']} MB")
    if frame:
        print(f"frame {out['frame_ms']['mean']} ms, {out['fps']['mean']} FPS")
    print(f"zone entry → alert {e['zone_entry_to_alert']} s; alert → report {e['alert_to_report']} s")
    print("-> evaluation/results/system.json")


if __name__ == "__main__":
    main()
