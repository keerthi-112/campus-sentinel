"""Durable job store for asynchronous report generation.

Report generation takes tens of seconds, so the API accepts an incident, hands
back a job id, and writes the report when it's ready. This module is the record
of what's in flight.

SQLite rather than an in-memory dict for two reasons that both happen in
practice: uvicorn run with multiple workers would otherwise create a job in one
process and 404 on it from another, and a restart mid-generation would strand
the client polling a job nobody is working on.
"""
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from config import JOBS_DB

PENDING = "pending"
RUNNING = "running"
READY = "ready"
FAILED = "failed"

UNFINISHED = (PENDING, RUNNING)

SCHEMA = """
CREATE TABLE IF NOT EXISTS report_jobs (
    job_id        TEXT PRIMARY KEY,
    incident_ref  TEXT,
    status        TEXT NOT NULL,
    incident_json TEXT NOT NULL,
    report_json   TEXT,
    error         TEXT,
    created_at    TEXT NOT NULL,
    started_at    TEXT,
    finished_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_report_jobs_incident_ref ON report_jobs (incident_ref);
CREATE INDEX IF NOT EXISTS idx_report_jobs_created_at ON report_jobs (created_at DESC);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _connect():
    """One connection per operation — safe across the threadpool FastAPI runs
    background work in, where a shared connection would not be."""
    connection = sqlite3.connect(JOBS_DB, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        # WAL lets a poll read while a background thread writes.
        connection.execute("PRAGMA journal_mode=WAL")
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db() -> None:
    JOBS_DB.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as connection:
        connection.executescript(SCHEMA)


def _row_to_dict(row: sqlite3.Row) -> dict:
    return {
        "job_id": row["job_id"],
        "incident_ref": row["incident_ref"],
        "status": row["status"],
        "report": json.loads(row["report_json"]) if row["report_json"] else None,
        "error": row["error"],
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
    }


def create(incident: dict) -> dict:
    job_id = str(uuid.uuid4())
    with _connect() as connection:
        connection.execute(
            "INSERT INTO report_jobs (job_id, incident_ref, status, incident_json, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (job_id, incident.get("incident_ref"), PENDING, json.dumps(incident), _now()),
        )
    return get(job_id)


def get(job_id: str) -> dict | None:
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM report_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
    return _row_to_dict(row) if row else None


def get_incident(job_id: str) -> dict | None:
    with _connect() as connection:
        row = connection.execute(
            "SELECT incident_json FROM report_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
    return json.loads(row["incident_json"]) if row else None


def find_live_by_incident_ref(incident_ref: str) -> dict | None:
    """An existing job for this incident that is running or already finished well.

    Makes a resubmitted incident idempotent — a retried request or double-click
    returns the job already in flight instead of paying for a second generation.
    A previously failed job is ignored so a retry can genuinely retry.
    """
    if not incident_ref:
        return None
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM report_jobs WHERE incident_ref = ? AND status != ?"
            " ORDER BY created_at DESC LIMIT 1",
            (incident_ref, FAILED),
        ).fetchone()
    return _row_to_dict(row) if row else None


def list_recent(limit: int = 20) -> list[dict]:
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM report_jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def mark_running(job_id: str) -> None:
    with _connect() as connection:
        connection.execute(
            "UPDATE report_jobs SET status = ?, started_at = ? WHERE job_id = ?",
            (RUNNING, _now(), job_id),
        )


def mark_ready(job_id: str, report: dict) -> None:
    with _connect() as connection:
        connection.execute(
            "UPDATE report_jobs SET status = ?, report_json = ?, error = NULL, finished_at = ?"
            " WHERE job_id = ?",
            (READY, json.dumps(report), _now(), job_id),
        )


def mark_failed(job_id: str, error: str) -> None:
    with _connect() as connection:
        connection.execute(
            "UPDATE report_jobs SET status = ?, error = ?, finished_at = ? WHERE job_id = ?",
            (FAILED, error, _now(), job_id),
        )


def sweep_orphaned(older_than_seconds: int) -> int:
    """Fail jobs that can no longer be in flight, and report how many.

    Nothing resumes a job across a restart, so a stranded one would otherwise
    sit on 'running' with a client polling it forever.

    Age is the test rather than "any unfinished job", because several uvicorn
    workers share this database: a worker starting up must not fail a job
    another worker is legitimately still working on. Past the LLM timeout no
    job can legitimately still be running, whoever owns it.
    """
    cutoff = datetime.now(timezone.utc).timestamp() - older_than_seconds
    cutoff_iso = datetime.fromtimestamp(cutoff, timezone.utc).isoformat()

    with _connect() as connection:
        cursor = connection.execute(
            "UPDATE report_jobs SET status = ?, error = ?, finished_at = ?"
            " WHERE status IN (?, ?) AND created_at < ?",
            (
                FAILED,
                "Interrupted — the server stopped while this report was being generated. Resubmit the incident.",
                _now(),
                PENDING,
                RUNNING,
                cutoff_iso,
            ),
        )
        return cursor.rowcount
