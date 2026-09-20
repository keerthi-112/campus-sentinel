"""FastAPI router exposing the RAG track's two entry points.

The Application track mounts this in backend/app/main.py with one line:

    from rag.api import router as rag_router
    app.include_router(rag_router)

Keeping the router here means retrieval/prompt changes stay inside rag/ and
don't require a backend edit.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from generate import generate_chat_answer, generate_report

router = APIRouter(tags=["rag"])


class Incident(BaseModel):
    """Mirrors docs/schemas/incident.json."""

    event_type: str
    camera_name: str
    zone_name: str
    timestamp: str
    confidence: float
    duration_seconds: float
    severity: str
    status: str
    incident_ref: str | None = None


class ChatQuestion(BaseModel):
    question: str


@router.post("/reports")
def create_report(incident: Incident) -> dict:
    """Generate a sourced incident report. Called once an incident is confirmed."""
    try:
        return generate_report(incident.model_dump(exclude_none=True))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/chat")
def chat(payload: ChatQuestion) -> dict:
    """Answer an operator question from the knowledge base. Needs no incident."""
    try:
        return generate_chat_answer(payload.question)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
