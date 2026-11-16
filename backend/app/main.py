"""
Campus Sentinel — Application backend (FastAPI + SQLite).

Wires the four tracks together:

  vision/ camera workers ─► federated event classifier ─► incidents (SQLite)
                                                    └──► rag/ report jobs ─► reports (SQLite)
  dashboard ◄── REST + MJPEG ── this API ── chat ──► rag/ (retrieval + LLM)

Run from the backend/ folder:

    uvicorn app.main:app --reload --port 8000

API docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import config  # noqa: F401  (sets up sys.path for vision/, federated/, rag/)
from .auth import hash_password
from .config import ADMIN_PASSWORD, ADMIN_USER, CORS_ORIGINS
from .db import SessionLocal, init_db
from .models import Camera, User
from .routers import auth, cameras, incidents, misc
from .services.incidents import handle_candidate
from .services.rag_bridge import rag
from .services.training import sync_rounds
from .services.vision_worker import CameraSpec, vision_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("sentinel")


def _on_candidate(incident: dict, features: dict, camera_id: int) -> bool:
    """Called from camera worker threads for every vision incident candidate."""
    with SessionLocal() as db:
        _, created = handle_candidate(db, incident, features=features, camera_id=camera_id, source="vision")
        return created


def _seed() -> None:
    with SessionLocal() as db:
        if not db.query(User).first():
            db.add(User(username=ADMIN_USER, password_hash=hash_password(ADMIN_PASSWORD), role="admin"))
            log.info("created default admin user '%s'", ADMIN_USER)
        if not db.query(Camera).first():
            db.add(Camera(camera_name="CAM 01 - Main Gate", zone_name="Restricted Zone Alpha",
                          source_type="webcam", source="0", restricted_region=[], tier=2,
                          dwell_threshold=4.0))
        db.commit()


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    _seed()
    vision_manager.on_candidate = _on_candidate
    rag.start_sync()
    try:
        sync_rounds()
    except Exception:
        pass
    # Resume cameras that were monitoring before a restart.
    with SessionLocal() as db:
        for cam in db.query(Camera).filter(Camera.active_state.is_(True)).all():
            try:
                vision_manager.start(CameraSpec.from_row(cam))
            except Exception as exc:
                log.warning("could not resume %s: %s", cam.camera_name, exc)
    yield
    vision_manager.stop_all()
    rag.stop()


app = FastAPI(title="Campus Sentinel API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

app.include_router(auth.router)
app.include_router(cameras.router)
app.include_router(incidents.router)
app.include_router(misc.reports)
app.include_router(misc.federated)
app.include_router(misc.chat)
app.include_router(misc.system)

# The RAG track's own endpoints, exactly as documented in rag/README.md,
# under /api/rag (POST /api/rag/reports, GET /api/rag/reports/{job_id}, POST /api/rag/chat).
if rag.router is not None:
    app.include_router(rag.router, prefix="/api/rag")


@app.get("/health")
def health():
    return {"status": "ok", "rag": rag.available, "vision": vision_manager.available}
