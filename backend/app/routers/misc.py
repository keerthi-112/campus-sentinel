"""Reports, federated learning status/training, operator chat, and system status."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..auth import admin_user, current_user
from ..db import get_db
from ..models import Camera, FederatedRound, Incident, Report
from ..schemas import ChatIn, ReportOut, TrainIn
from ..services import metrics
from ..services.incidents import classifier
from ..services.rag_bridge import rag
from ..services.training import latest_rounds, latest_summary, sync_rounds, training
from ..services.vision_worker import VISION_AVAILABLE, VISION_ERROR, vision_manager

# ----------------------------------------------------------------------
reports = APIRouter(prefix="/api/reports", tags=["reports"], dependencies=[Depends(current_user)])


@reports.get("", response_model=list[ReportOut])
def list_reports(db: Session = Depends(get_db), limit: int = 50):
    return db.query(Report).order_by(Report.id.desc()).limit(max(1, min(limit, 200))).all()


@reports.get("/{incident_ref}", response_model=ReportOut)
def report_for(incident_ref: str, db: Session = Depends(get_db)):
    r = db.query(Report).filter(Report.incident_ref == incident_ref).first()
    if not r:
        raise HTTPException(404, "No report for this incident yet")
    return r


# ----------------------------------------------------------------------
federated = APIRouter(prefix="/api/federated", tags=["federated"], dependencies=[Depends(current_user)])


@federated.get("/model")
def model_info():
    return classifier.info()


@federated.get("/rounds")
def rounds():
    """Latest run, per strategy, straight from the FL track's log."""
    data = latest_rounds()
    if not data:
        return {"run_id": None, "strategies": {}, "config": {}}
    return data


@federated.get("/summary")
def summary():
    return latest_summary() or {}


@federated.get("/history")
def history(db: Session = Depends(get_db)):
    """All runs that were mirrored into the database."""
    sync_rounds()
    rows = db.query(FederatedRound).order_by(FederatedRound.id).all()
    runs: dict[str, dict] = {}
    for r in rows:
        run = runs.setdefault(r.run_id, {"run_id": r.run_id, "engine": r.engine,
                                         "created_at": r.created_at, "strategies": {}})
        run["strategies"].setdefault(r.strategy, []).append(
            {"round_number": r.round_number, "accuracy": r.metrics.get("accuracy"),
             "f1": r.metrics.get("f1"), "clients": r.participating_clients})
    return list(runs.values())[::-1]


@federated.get("/status")
def train_status():
    return training.status()


@federated.post("/train", dependencies=[Depends(admin_user)])
def train(body: TrainIn):
    try:
        return training.start(body.model_dump())
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


# ----------------------------------------------------------------------
chat = APIRouter(prefix="/api/chat", tags=["chat"], dependencies=[Depends(current_user)])


@chat.post("")
def ask(body: ChatIn):
    """Operator question → retrieval + LLM → answer with cited section. No incident needed."""
    try:
        return rag.chat(body.question)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


# ----------------------------------------------------------------------
system = APIRouter(prefix="/api/system", tags=["system"], dependencies=[Depends(current_user)])


@system.get("/status")
def system_status(db: Session = Depends(get_db)):
    cams = db.query(Camera).all()
    return {
        "vision": {"available": VISION_AVAILABLE, "error": VISION_ERROR,
                   "cameras": len(cams),
                   "running": sum(1 for c in cams if vision_manager.is_running(c.id))},
        "classifier": classifier.info(),
        "rag": rag.status(),
        "training": {"running": training.running},
        "incidents": {"total": db.query(Incident).count(),
                      "active": db.query(Incident).filter(Incident.status.in_(["open", "acknowledged"])).count()},
    }


@system.get("/metrics")
def system_metrics(db: Session = Depends(get_db)):
    """Live latency + resource numbers for the dashboard and evaluation/eval_system.py."""
    cams = []
    for c in db.query(Camera).all():
        rt = vision_manager.runtime(c.id)
        if rt and rt["state"] == "running":
            cams.append({"id": c.id, "camera_name": c.camera_name, "fps": rt["fps"],
                         "frame_ms": rt.get("frame_ms"), "frames": rt["frames"]})
    return metrics.snapshot(cams)
