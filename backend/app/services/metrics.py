"""
metrics.py — Live system measurements (end-to-end latency and
basic resource usage). In memory only; reset on restart.

  decision_ms        candidate from vision → classifier → incident stored (+ report queued)
  report_queue_s     incident stored → LLM starts on it (waiting behind other reports)
  report_generation_s LLM start → report ready (retrieval + generation)
  report_total_s     incident stored → report ready (what the operator waits)
  frame_ms / fps     per camera, processing time of one frame (YOLO + tracking + zones)
  cpu / memory       this backend process and the whole machine
"""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timezone

import psutil

_lock = threading.Lock()
_series: dict[str, deque] = {k: deque(maxlen=500) for k in
                             ("decision_ms", "report_queue_s", "report_generation_s", "report_total_s")}
_proc = psutil.Process()
_proc.cpu_percent(None)  # prime the counters
psutil.cpu_percent(None)
_started = time.time()


def record(name: str, value: float) -> None:
    with _lock:
        _series[name].append(float(value))


def _parse(ts: str | None) -> float | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def record_report(job: dict, incident_created_at) -> None:
    """Called once when a report job finishes."""
    if incident_created_at is not None and incident_created_at.tzinfo is None:
        # SQLite returns naive datetimes; they were stored in UTC.
        incident_created_at = incident_created_at.replace(tzinfo=timezone.utc)
    created = incident_created_at.timestamp() if incident_created_at else _parse(job.get("created_at"))
    started, finished = _parse(job.get("started_at")), _parse(job.get("finished_at"))
    if finished and created:
        record("report_total_s", max(0.0, finished - created))
    if finished and started:
        record("report_generation_s", max(0.0, finished - started))
    if started and created:
        record("report_queue_s", max(0.0, started - created))


def _stats(xs) -> dict | None:
    xs = sorted(xs)
    if not xs:
        return None
    pick = lambda q: xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))]
    return {"mean": round(sum(xs) / len(xs), 3), "p50": round(pick(0.5), 3), "p95": round(pick(0.95), 3),
            "max": round(xs[-1], 3), "n": len(xs)}


def snapshot(cameras: list[dict]) -> dict:
    with _lock:
        series = {k: _stats(v) for k, v in _series.items()}
    mem = psutil.virtual_memory()
    children = _proc.children(recursive=True)
    child_rss = sum((c.memory_info().rss for c in children if c.is_running()), 0)
    return {
        "uptime_s": round(time.time() - _started),
        "process": {"cpu_percent": round(_proc.cpu_percent(None), 1),
                    "rss_mb": round(_proc.memory_info().rss / 1048576, 1),
                    "child_processes_rss_mb": round(child_rss / 1048576, 1),
                    "threads": _proc.num_threads()},
        "system": {"cpu_percent": round(psutil.cpu_percent(None), 1), "cpu_count": psutil.cpu_count(),
                   "ram_used_percent": round(mem.percent, 1), "ram_total_gb": round(mem.total / 1e9, 1)},
        "cameras": cameras,
        "latency": series,
    }
