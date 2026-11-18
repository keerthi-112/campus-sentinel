"""
rag_bridge.py — Connects the backend to the RAG & LLM track without changing it.

* Imports rag/ (flat modules) by putting rag/ on sys.path.
* Queues report jobs through rag/jobs.py and runs rag/api.py's own background
  worker, so job states (pending → running → ready/failed), dedup by
  incident_ref, and error messages behave exactly as the RAG track documented.
* A small sync thread copies finished reports into the backend's `reports`
  table and updates each incident's report_status, so the dashboard only has
  to read one database.

If rag/ cannot be imported (e.g. chromadb/ollama not installed), the rest of
the system keeps running and the dashboard shows RAG as unavailable.
"""

from __future__ import annotations

import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from ..config import RAG_DIR

log = logging.getLogger("sentinel.rag")


class RagBridge:
    def __init__(self) -> None:
        self.available = False
        self.error: str | None = None
        self.router = None
        self._api = self._jobs = self._generate = self._config = None
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-report")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._load()

    def _load(self) -> None:
        try:
            if str(RAG_DIR) not in sys.path:
                sys.path.insert(0, str(RAG_DIR))
            import api as rag_api  # rag/api.py
            import config as rag_config  # rag/config.py
            import generate as rag_generate  # rag/generate.py
            import jobs as rag_jobs  # rag/jobs.py

            self._api, self._jobs, self._generate, self._config = rag_api, rag_jobs, rag_generate, rag_config
            self.router = rag_api.router
            self.available = True
        except Exception as exc:  # pragma: no cover - depends on environment
            self.error = f"{type(exc).__name__}: {exc}"
            log.warning("RAG track unavailable: %s", self.error)

    # ------------------------------------------------------------------
    def queue_report(self, incident: dict) -> dict | None:
        """Create (or reuse) a report job for a confirmed incident."""
        if not self.available:
            return None
        payload = {k: incident[k] for k in (
            "event_type", "camera_name", "zone_name", "timestamp", "confidence",
            "duration_seconds", "severity", "status", "incident_ref") if k in incident}
        existing = self._jobs.find_live_by_incident_ref(payload.get("incident_ref"))
        if existing:
            return existing
        job = self._jobs.create(payload)
        self._pool.submit(self._api._run_generation, job["job_id"])
        return job

    def get_job(self, job_id: str) -> dict | None:
        return self._jobs.get(job_id) if self.available else None

    def chat(self, question: str) -> dict:
        if not self.available:
            raise RuntimeError(f"RAG track unavailable: {self.error}")
        return self._generate.generate_chat_answer(question)

    def status(self) -> dict:
        info = {"available": self.available, "error": self.error}
        if not self.available:
            return info
        info["llm_model"] = getattr(self._config, "LLM_MODEL", None)
        info["embed_model"] = getattr(self._config, "EMBED_MODEL", None)
        try:
            import retrieve as rag_retrieve

            info["kb_sections"] = rag_retrieve.get_collection().count()
        except Exception as exc:
            info["kb_sections"] = None
            info["kb_error"] = str(exc)
        try:
            import ollama

            models = ollama.Client(timeout=2).list()
            names = [getattr(m, "model", None) or m.get("model") or m.get("name")
                     for m in (models.models if hasattr(models, "models") else models.get("models", []))]
            info["ollama_reachable"] = True
            info["ollama_models"] = names
        except Exception as exc:
            info["ollama_reachable"] = False
            info["ollama_error"] = str(exc)[:200]
        return info

    # ------------------------------------------------------------------
    # Sync finished jobs into the backend DB
    # ------------------------------------------------------------------
    def sync_once(self) -> int:
        if not self.available:
            return 0
        from ..db import SessionLocal
        from ..models import Incident, Report

        changed = 0
        with SessionLocal() as db:
            pending = db.query(Incident).filter(
                Incident.report_job_id.isnot(None),
                Incident.report_status.in_(["pending", "running"])).all()
            for inc in pending:
                job = self._jobs.get(inc.report_job_id)
                if not job or job["status"] == inc.report_status:
                    continue
                inc.report_status = job["status"]
                inc.report_error = job.get("error")
                if job["status"] == "ready":  # only successful reports count towards latency
                    from . import metrics

                    metrics.record_report(job, inc.created_at)
                if job["status"] == "ready" and job.get("report"):
                    r = job["report"]
                    row = db.query(Report).filter(Report.incident_ref == inc.incident_ref).first() or Report(
                        incident_ref=inc.incident_ref)
                    row.report_text = r.get("report_text", "")
                    row.suggested_action = r.get("suggested_action", "")
                    row.source_section = r.get("source_section", "")
                    row.created_time = r.get("created_time", "")
                    row.job_id = inc.report_job_id
                    db.add(row)
                changed += 1
            db.commit()
        return changed

    def start_sync(self, interval: float = 2.0) -> None:
        if self._thread or not self.available:
            return

        def loop():
            while not self._stop.wait(interval):
                try:
                    self.sync_once()
                except Exception:  # pragma: no cover
                    log.exception("report sync failed")

        self._thread = threading.Thread(target=loop, name="rag-sync", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._pool.shutdown(wait=False, cancel_futures=True)


rag = RagBridge()
