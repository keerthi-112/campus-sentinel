"""FastAPI router exposing the RAG track's entry points.

The Application track mounts this in backend/app/main.py with one line:

    from rag.api import router as rag_router
    app.include_router(rag_router)

Report generation is asynchronous. A report takes tens of seconds to write, and
an operator must not wait on prose to learn that someone is standing in a
restricted zone — so POST /reports returns a job id immediately and the report
is written in the background. Poll GET /reports/{job_id} for the result.

Chat stays synchronous: the operator asked the question and is waiting for the
answer, so there is nothing to hand back early.
"""
import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException, Response, status
from pydantic import BaseModel, Field

import jobs
from config import LLM_TIMEOUT_SECONDS
from generate import generate_chat_answer, generate_report

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rag"])

# At import, i.e. once per worker process: make sure the store exists, and fail
# anything a previous process left mid-flight so nobody polls a dead job.
jobs.init_db()
_swept = jobs.sweep_orphaned(older_than_seconds=LLM_TIMEOUT_SECONDS)
if _swept:
    logger.warning("failed %d report job(s) stranded by a previous run", _swept)


class Incident(BaseModel):
    """Mirrors docs/schemas/incident.json."""

    event_type: str
    camera_name: str
    zone_name: str
    timestamp: str
    confidence: float = Field(ge=0.0, le=1.0)
    duration_seconds: float = Field(ge=0.0)
    severity: str
    status: str
    incident_ref: str | None = None


class ChatQuestion(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


def _run_generation(job_id: str) -> None:
    """Background worker. Runs in FastAPI's threadpool, so the blocking LLM call
    never touches the event loop.

    Every exit path has to leave a terminal status behind — a job stuck on
    'running' is a client polling forever.
    """
    incident = jobs.get_incident(job_id)
    if incident is None:
        logger.error("report job %s vanished before it could run", job_id)
        return

    jobs.mark_running(job_id)
    try:
        jobs.mark_ready(job_id, generate_report(incident))
    except RuntimeError as exc:
        # Expected, diagnosable failures: model missing, Ollama down, timeout,
        # malformed model output. The message already says how to fix it.
        logger.warning("report job %s failed: %s", job_id, exc)
        jobs.mark_failed(job_id, str(exc))
    except Exception as exc:
        logger.exception("report job %s failed unexpectedly", job_id)
        jobs.mark_failed(job_id, f"Unexpected error while generating the report: {exc}")


@router.post("/reports", status_code=status.HTTP_202_ACCEPTED)
def request_report(incident: Incident, background_tasks: BackgroundTasks, response: Response) -> dict:
    """Queue a report for a confirmed incident. Returns immediately.

    Resubmitting an incident that already has a live job returns that job rather
    than paying for a second generation — retries and double-clicks are free.
    """
    payload = incident.model_dump(exclude_none=True)

    existing = jobs.find_live_by_incident_ref(payload.get("incident_ref"))
    if existing:
        response.status_code = status.HTTP_200_OK
        return existing

    job = jobs.create(payload)
    background_tasks.add_task(_run_generation, job["job_id"])
    return job


@router.get("/reports/{job_id}")
def get_report(job_id: str) -> dict:
    """Poll a queued report. `status` is pending, running, ready, or failed."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No report job with id {job_id}")
    return job


@router.get("/reports")
def list_reports(limit: int = 20) -> list[dict]:
    """Recent report jobs, newest first — backs the dashboard's report list."""
    return jobs.list_recent(limit=max(1, min(limit, 100)))


@router.post("/chat")
def chat(payload: ChatQuestion) -> dict:
    """Answer an operator question from the knowledge base. Needs no incident."""
    try:
        return generate_chat_answer(payload.question)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
