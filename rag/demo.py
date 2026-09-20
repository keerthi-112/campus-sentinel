"""Narrated end-to-end demo of the RAG track.

Starts the service in-process and drives it over real HTTP, so what you see is
what the React dashboard will get.

    python demo.py

Needs Ollama running with both models pulled. Nothing from the other three
tracks is required — the incident below stands in for what the vision track
will eventually send.
"""
import threading
import time

import httpx
import uvicorn
from fastapi import FastAPI

import config
from api import router
from ingest import load_documents
from retrieve import retrieve

PORT = 8123
BASE = f"http://127.0.0.1:{PORT}"

RUN = time.strftime("%H%M%S")

INCIDENT = {
    # Unique per run, so the demo always shows a real generation rather than
    # the idempotent hit on a report an earlier run already produced.
    "incident_ref": f"demo-{RUN}-a",
    "event_type": "restricted_area_entry",
    "camera_name": "CAM 03",
    "zone_name": "Lab 2 restricted area",
    "timestamp": "2026-09-20T22:47:11Z",
    "confidence": 0.94,
    "duration_seconds": 4.2,
    "severity": "medium",
    "status": "open",
}


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


client = httpx.Client(base_url=BASE, timeout=600)


def start_server() -> None:
    app = FastAPI()
    app.include_router(router)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()

    for _ in range(100):
        try:
            client.get("/reports", timeout=1)
            return
        except httpx.HTTPError:
            time.sleep(0.1)
    raise SystemExit("demo server did not start")


def main() -> None:
    rule("1. THE KNOWLEDGE BASE — what the system can cite")
    records = load_documents()
    by_doc: dict[str, list[str]] = {}
    for record in records:
        by_doc.setdefault(record["title"], []).append(record["section"].split(", ", 1)[-1])
    for title, sections in by_doc.items():
        print(f"\n  {title}")
        for section in sections:
            print(f"      - {section}")
    print(f"\n  {len(records)} sections across {len(by_doc)} documents, embedded with {config.EMBED_MODEL}")

    rule("2. SEMANTIC RETRIEVAL — matching on meaning, not keywords")
    question = "someone is hanging around a lab they shouldn't be in late at night"
    print(f"\n  Query: {question!r}")
    print("  (note: none of these words appear verbatim in the documents)\n")
    for i, hit in enumerate(retrieve(question, k=3), 1):
        print(f"    {i}. [distance {hit['distance']:.3f}]  {hit['section']}")

    rule("3. AN INCIDENT ARRIVES — the alert must not wait on the report")
    start_server()
    print(f"\n  Incident: {INCIDENT['event_type']} at {INCIDENT['camera_name']}")
    print(f"            zone={INCIDENT['zone_name']}  confidence={INCIDENT['confidence']}")
    print(f"            dwell={INCIDENT['duration_seconds']}s  time={INCIDENT['timestamp']}")

    t0 = time.time()
    response = client.post("/reports", json=INCIDENT)
    accepted_ms = (time.time() - t0) * 1000
    job = response.json()

    print(f"\n  POST /reports  ->  HTTP {response.status_code} in {accepted_ms:.0f} ms")
    print(f"  job_id: {job['job_id']}")
    print(f"  status: {job['status']}   (the dashboard can already show the alert)")

    # A second incident, submitted while the first is still generating: shows
    # steady-state latency and that jobs queue rather than collide.
    second = {**INCIDENT, "incident_ref": f"demo-{RUN}-b", "camera_name": "CAM 07", "zone_name": "Server room"}
    t_second = time.time()
    second_job = client.post("/reports", json=second).json()
    print(f"\n  A second incident arrives at CAM 07 while the first is still generating:")
    print(f"  POST /reports  ->  accepted in {(time.time() - t_second) * 1000:.0f} ms")
    print(f"  status: {second_job['status']}   — queued alongside the first")

    print("\n  Polling the first report while it is written in the background:")
    seen = set()
    while True:
        job = client.get(f"/reports/{job['job_id']}").json()
        elapsed = time.time() - t0
        if job["status"] not in seen:
            seen.add(job["status"])
            print(f"    t+{elapsed:5.1f}s   status={job['status']}")
        if job["status"] in ("ready", "failed"):
            break
        time.sleep(1)

    total = time.time() - t0
    rule("4. THE GENERATED REPORT")
    if job["status"] == "failed":
        print(f"\n  FAILED: {job['error']}")
    else:
        report = job["report"]
        print(f"\n  {report['report_text']}\n")
        print(f"  RECOMMENDED ACTION: {report['suggested_action']}")
        print(f"  SOURCE:             {report['source_section']}")
        print(f"\n  Written by {config.LLM_MODEL} in {total:.1f}s — during which the alert")
        print(f"  had already been visible for {total - accepted_ms/1000:.0f}s.")

    rule("5. OPERATOR CHAT — same knowledge base, no incident needed")
    chat_question = "how quickly must an operator acknowledge a tier 3 alert?"
    print(f"\n  Q: {chat_question}")
    t1 = time.time()
    answer = client.post("/chat", json={"question": chat_question}).json()
    print(f"\n  A: {answer['answer']}")
    print(f"\n  SOURCE: {answer['source_section']}   ({time.time() - t1:.1f}s)")

    rule("SUMMARY")
    print(f"""
  Everything above ran locally. No API keys, no internet, no data leaving
  this machine — {config.EMBED_MODEL} and {config.LLM_MODEL} both run on Ollama here.

  Alert latency:  {accepted_ms:.0f} ms        (what the operator actually waits for)
  Report latency: {total:.1f} s          (written in the background)

  Still to come from the other tracks: real detections from vision/ replacing
  the hand-written incident above, and backend/ + frontend/ rendering these
  same endpoints as a dashboard.
""")


if __name__ == "__main__":
    main()
