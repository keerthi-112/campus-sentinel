"""
incidents.py — The incident decision stage.

    candidate (vision) + feature vector
        → federated event classifier: escalate or suppress?
        → store the incident (both outcomes, for evaluation)
        → escalated ones get a RAG report job

This is the "Incident Decision" box in the architecture diagram.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from federated.classifier import EventClassifier

from ..models import Incident
from . import metrics
from .rag_bridge import rag

log = logging.getLogger("sentinel.incidents")

classifier = EventClassifier()
_lock = threading.Lock()


def _fallback_features(inc: dict) -> dict:
    """Best-effort feature vector when the sender only has the incident record."""
    crowd = inc["event_type"] == "crowd_formation"
    return {
        "camera_name": inc["camera_name"], "zone_name": inc["zone_name"], "timestamp": inc["timestamp"],
        "person_count": 6 if crowd else 1, "density": 0.3 if crowd else 0.05,
        "occupancy": crowd, "motion_score": 0.03,
        "zone_dwell_seconds": float(inc.get("duration_seconds", 0.0)),
        "max_confidence": float(inc.get("confidence", 0.0)),
    }


def _make_ref(inc: dict) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    cam = "".join(c for c in inc["camera_name"].lower() if c.isalnum())
    return f"{inc['event_type']}_{cam}_{ts}"


def handle_candidate(db: Session, incident: dict, features: dict | None = None,
                     camera_id: int | None = None, source: str = "vision") -> tuple[Incident, bool]:
    """Run the classifier, persist the incident, queue a report if escalated.

    Returns (incident_row, created). Re-sending the same incident_ref is idempotent.
    """
    t0 = time.perf_counter()
    incident = dict(incident)
    incident["incident_ref"] = incident.get("incident_ref") or _make_ref(incident)

    with _lock:
        existing = db.query(Incident).filter(Incident.incident_ref == incident["incident_ref"]).first()
        if existing:
            return existing, False

        partial = features is None
        fv = features or _fallback_features(incident)
        decision = classifier.decide(fv, incident["event_type"])
        reason = decision["reason"] + (" (partial features: incident record only)" if partial else "")

        row = Incident(
            incident_ref=incident["incident_ref"],
            event_type=incident["event_type"],
            camera_id=camera_id,
            camera_name=incident["camera_name"],
            zone_name=incident["zone_name"],
            timestamp=incident["timestamp"],
            confidence=float(incident["confidence"]),
            duration_seconds=float(incident["duration_seconds"]),
            severity=incident.get("severity", "medium"),
            status="open" if decision["escalate"] else "suppressed",
            classifier_decision="escalated" if decision["escalate"] else "suppressed",
            classifier_score=decision["score"],
            model_version=decision["model_version"],
            decision_reason=reason,
            features=fv,
            source=source,
        )
        db.add(row)
        db.commit()
        db.refresh(row)

    if decision["escalate"]:
        queue_report(db, row)
    metrics.record("decision_ms", (time.perf_counter() - t0) * 1000)
    log.info("incident %s %s (score=%s)", row.incident_ref, row.classifier_decision, row.classifier_score)
    return row, True


def queue_report(db: Session, row: Incident, force: bool = False) -> None:
    if not rag.available:
        row.report_status = "failed"
        row.report_error = f"RAG track unavailable: {rag.error}"
        db.commit()
        return
    payload = {
        "incident_ref": row.incident_ref if not force else f"{row.incident_ref}#retry{datetime.now().timestamp():.0f}",
        "event_type": row.event_type, "camera_name": row.camera_name, "zone_name": row.zone_name,
        "timestamp": row.timestamp, "confidence": row.confidence,
        "duration_seconds": row.duration_seconds, "severity": row.severity, "status": "open",
    }
    try:
        job = rag.queue_report(payload)
    except Exception as exc:
        row.report_status, row.report_error = "failed", str(exc)
        db.commit()
        return
    if job:
        row.report_job_id = job["job_id"]
        row.report_status = job["status"]
        row.report_error = job.get("error")
        db.commit()
