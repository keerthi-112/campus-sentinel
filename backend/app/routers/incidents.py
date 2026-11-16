"""Incidents: list / detail / status workflow, plus the ingest endpoint for an external vision process."""

from __future__ import annotations

from collections import Counter

from fastapi import APIRouter, Depends, Header, HTTPException, Response, status
from sqlalchemy.orm import Session

from ..auth import current_user
from ..config import INGEST_KEY
from ..db import get_db
from ..models import Camera, Incident, Report
from ..schemas import IncidentIn, IncidentOut, IncidentPatch, ReportOut
from ..services.incidents import handle_candidate, queue_report

router = APIRouter(prefix="/api/incidents", tags=["incidents"])


def to_out(db: Session, inc: Incident) -> IncidentOut:
    out = IncidentOut.model_validate(inc)
    rep = db.query(Report).filter(Report.incident_ref == inc.incident_ref).first()
    out.report = ReportOut.model_validate(rep) if rep else None
    return out


@router.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
def ingest(body: IncidentIn, response: Response, db: Session = Depends(get_db),
           x_ingest_key: str | None = Header(default=None)):
    """For `python -m vision.run_demo --rag-url http://127.0.0.1:8000/api/incidents/ingest`.

    Runs the same decision stage as the built-in camera workers: federated
    classifier → incident → RAG report. Returns `job_id` + `status` so the
    vision track's post_incident_to_rag() log line works unchanged.
    """
    if INGEST_KEY and x_ingest_key != INGEST_KEY:
        raise HTTPException(401, "Invalid ingest key")
    data = body.model_dump(exclude={"features"})
    cam = db.query(Camera).filter(Camera.camera_name == body.camera_name).first()
    row, created = handle_candidate(db, data, features=body.features,
                                    camera_id=cam.id if cam else None, source="external")
    if not created:
        response.status_code = status.HTTP_200_OK
    return {"incident_id": row.id, "incident_ref": row.incident_ref,
            "decision": row.classifier_decision, "score": row.classifier_score,
            "job_id": row.report_job_id, "status": row.report_status or row.status}


@router.get("", response_model=list[IncidentOut], dependencies=[Depends(current_user)])
def list_incidents(db: Session = Depends(get_db), status_filter: str | None = None,
                   event_type: str | None = None, camera_id: int | None = None,
                   include_suppressed: bool = True, limit: int = 100):
    q = db.query(Incident)
    if status_filter:
        q = q.filter(Incident.status == status_filter)
    if event_type:
        q = q.filter(Incident.event_type == event_type)
    if camera_id:
        q = q.filter(Incident.camera_id == camera_id)
    if not include_suppressed:
        q = q.filter(Incident.status != "suppressed")
    rows = q.order_by(Incident.id.desc()).limit(max(1, min(limit, 500))).all()
    return [to_out(db, r) for r in rows]


@router.get("/stats", dependencies=[Depends(current_user)])
def stats(db: Session = Depends(get_db)):
    rows = db.query(Incident.status, Incident.event_type, Incident.severity,
                    Incident.classifier_decision).all()
    return {
        "total": len(rows),
        "by_status": dict(Counter(r[0] for r in rows)),
        "by_event_type": dict(Counter(r[1] for r in rows)),
        "by_severity": dict(Counter(r[2] for r in rows)),
        "by_decision": dict(Counter(r[3] for r in rows)),
        "active": sum(1 for r in rows if r[0] in ("open", "acknowledged")),
    }


@router.get("/{incident_id}", response_model=IncidentOut, dependencies=[Depends(current_user)])
def get_incident(incident_id: int, db: Session = Depends(get_db)):
    inc = db.get(Incident, incident_id)
    if not inc:
        raise HTTPException(404, "Incident not found")
    return to_out(db, inc)


@router.patch("/{incident_id}", response_model=IncidentOut, dependencies=[Depends(current_user)])
def update_status(incident_id: int, body: IncidentPatch, db: Session = Depends(get_db)):
    inc = db.get(Incident, incident_id)
    if not inc:
        raise HTTPException(404, "Incident not found")
    inc.status = body.status
    db.commit()
    return to_out(db, inc)


@router.post("/{incident_id}/report", response_model=IncidentOut, dependencies=[Depends(current_user)])
def regenerate_report(incident_id: int, db: Session = Depends(get_db)):
    """(Re)generate the report — also lets an operator escalate a suppressed candidate."""
    inc = db.get(Incident, incident_id)
    if not inc:
        raise HTTPException(404, "Incident not found")
    if inc.report_status in ("pending", "running"):
        raise HTTPException(409, "A report is already being generated")
    if inc.status == "suppressed":
        inc.status = "open"
        inc.decision_reason = (inc.decision_reason or "") + " | escalated manually by operator"
    queue_report(db, inc, force=inc.report_status == "ready")
    return to_out(db, inc)
