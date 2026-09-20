"""Scenario tests for the async report API.

Covers the paths that actually go wrong in operation — a dead LLM, a stranded
job, a resubmitted incident — not just the happy one. The LLM is stubbed so the
suite runs in seconds and tests the plumbing rather than model quality
(eval_reports.py measures that).

    python test_api.py
"""
import tempfile
import time
from pathlib import Path

# Point the job store at a throwaway database BEFORE anything imports it, so a
# test run never leaves stub reports in the store the dashboard reads.
import config

config.JOBS_DB = Path(tempfile.mkdtemp(prefix="sentinel-test-")) / "report_jobs.db"

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402
import jobs  # noqa: E402

INCIDENT = {
    "incident_ref": "test-001",
    "event_type": "restricted_area_entry",
    "camera_name": "CAM 03",
    "zone_name": "Lab 2 restricted area",
    "timestamp": "2026-09-20T18:30:04Z",
    "confidence": 0.94,
    "duration_seconds": 4.2,
    "severity": "medium",
    "status": "open",
}

STUB_REPORT = {
    "incident_ref": "test-001",
    "report_text": "stub",
    "suggested_action": "stub",
    "source_section": "Restricted Area Rules, Section 2 - Unauthorized Entry",
    "created_time": "2026-09-20T18:30:07Z",
}

app = FastAPI()
app.include_router(api.router)
client = TestClient(app)

results = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, condition, detail))
    print(f"  {'PASS' if condition else 'FAIL'}  {name}{('  — ' + detail) if detail and not condition else ''}")


def wait_for_terminal(job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/reports/{job_id}").json()
        if job["status"] in (jobs.READY, jobs.FAILED):
            return job
        time.sleep(0.05)
    return client.get(f"/reports/{job_id}").json()


print("\n1. Happy path — incident accepted, report generated in background")
_original = api.generate_report
api.generate_report = lambda incident: STUB_REPORT
try:
    response = client.post("/reports", json=INCIDENT)
    check("returns 202 Accepted immediately", response.status_code == 202, str(response.status_code))
    job = response.json()
    check("hands back a job id", bool(job.get("job_id")))
    check("starts unfinished", job["status"] in jobs.UNFINISHED, job["status"])
    check("report not ready yet", job["report"] is None)

    finished = wait_for_terminal(job["job_id"])
    check("reaches ready", finished["status"] == jobs.READY, f"{finished['status']}: {finished['error']}")
    check("carries the report", finished["report"] == STUB_REPORT)
    check("records finished_at", bool(finished["finished_at"]))
finally:
    api.generate_report = _original

print("\n2. Duplicate submission — same incident_ref is idempotent")
api.generate_report = lambda incident: STUB_REPORT
try:
    again = client.post("/reports", json=INCIDENT)
    check("returns 200, not 202", again.status_code == 200, str(again.status_code))
    check("returns the existing job", again.json()["job_id"] == job["job_id"])
finally:
    api.generate_report = _original

print("\n3. LLM unavailable — job fails with an actionable message")
def _boom(incident):
    raise RuntimeError("Could not reach Ollama (connection refused). Is it running? Start it and retry.")

api.generate_report = _boom
try:
    failing = client.post("/reports", json={**INCIDENT, "incident_ref": "test-002"})
    check("still accepts the incident", failing.status_code == 202, str(failing.status_code))
    finished = wait_for_terminal(failing.json()["job_id"])
    check("ends failed, not stuck", finished["status"] == jobs.FAILED, finished["status"])
    check("error says how to fix it", "Is it running?" in (finished["error"] or ""), str(finished["error"]))
    check("no report attached", finished["report"] is None)
finally:
    api.generate_report = _original

print("\n4. Unexpected crash — still reaches a terminal state")
def _crash(incident):
    raise ValueError("something nobody predicted")

api.generate_report = _crash
try:
    crashing = client.post("/reports", json={**INCIDENT, "incident_ref": "test-003"})
    finished = wait_for_terminal(crashing.json()["job_id"])
    check("ends failed rather than hanging", finished["status"] == jobs.FAILED, finished["status"])
    check("error is recorded", "something nobody predicted" in (finished["error"] or ""))
finally:
    api.generate_report = _original

print("\n5. Failed jobs are retryable — a new submission is not deduped onto them")
api.generate_report = lambda incident: STUB_REPORT
try:
    retry = client.post("/reports", json={**INCIDENT, "incident_ref": "test-002"})
    check("creates a fresh job", retry.status_code == 202, str(retry.status_code))
    finished = wait_for_terminal(retry.json()["job_id"])
    check("succeeds on retry", finished["status"] == jobs.READY, finished["status"])
finally:
    api.generate_report = _original

print("\n6. Unknown job id")
missing = client.get("/reports/does-not-exist")
check("returns 404", missing.status_code == 404, str(missing.status_code))

print("\n7. Malformed incident is rejected before any work is queued")
check("missing fields -> 422", client.post("/reports", json={"event_type": "x"}).status_code == 422)
check("confidence > 1 -> 422", client.post("/reports", json={**INCIDENT, "confidence": 1.5}).status_code == 422)
check("negative duration -> 422", client.post("/reports", json={**INCIDENT, "duration_seconds": -1}).status_code == 422)

print("\n8. Stranded jobs are swept, and only once they're old enough")
stranded = jobs.create({**INCIDENT, "incident_ref": "test-stranded"})
jobs.mark_running(stranded["job_id"])
check("a fresh running job survives a sweep", jobs.sweep_orphaned(older_than_seconds=300) == 0)
check("still running", jobs.get(stranded["job_id"])["status"] == jobs.RUNNING)
swept = jobs.sweep_orphaned(older_than_seconds=0)
check("an aged job is swept", swept >= 1, f"swept {swept}")
after = jobs.get(stranded["job_id"])
check("swept job is failed", after["status"] == jobs.FAILED, after["status"])
check("explains why", "Interrupted" in (after["error"] or ""))

print("\n9. Listing")
listing = client.get("/reports?limit=5")
check("returns a list", listing.status_code == 200 and isinstance(listing.json(), list))
check("respects limit", len(listing.json()) <= 5)
check("newest first", listing.json() == sorted(listing.json(), key=lambda j: j["created_at"], reverse=True))

print("\n10. Chat validation")
check("empty question -> 422", client.post("/chat", json={"question": ""}).status_code == 422)

_original_chat = api.generate_chat_answer
api.generate_chat_answer = lambda q: (_ for _ in ()).throw(RuntimeError("Could not reach Ollama"))
try:
    check("LLM down -> 503", client.post("/chat", json={"question": "hi"}).status_code == 503)
finally:
    api.generate_chat_answer = _original_chat

passed = sum(1 for _, ok, _ in results if ok)
print(f"\n{passed}/{len(results)} checks passed")
if passed != len(results):
    print("\nFailures:")
    for name, ok, detail in results:
        if not ok:
            print(f"  - {name}  {detail}")
    raise SystemExit(1)
