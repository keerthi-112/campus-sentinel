"""
Backend API tests — run from backend/:

    python -m tests.test_backend

Uses a throw-away database and stubs the LLM call, so it needs neither Ollama
nor a camera and runs in seconds. The federated classifier and the RAG job
queue are the real ones.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

TMP = tempfile.mkdtemp(prefix="sentinel-test-")
os.environ["SENTINEL_DATA_DIR"] = TMP
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.services.rag_bridge import rag  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SAMPLE = json.loads((REPO / "docs/schemas/incident.json").read_text())
SAMPLE.pop("_description", None)

if rag.available:  # stub the slow LLM step, keep the real job queue
    rag._api.generate_report = lambda inc: {
        "incident_ref": inc.get("incident_ref"), "report_text": "stub report",
        "suggested_action": "stub action", "source_section": "Restricted Area Rules, Section 2 - Unauthorized Entry",
        "created_time": "2026-01-01T00:00:00Z"}
    rag._generate.generate_chat_answer = lambda q: {"answer": f"stub: {q}", "source_section": "X, Y"}


class BackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.__enter__()
        r = cls.client.post("/api/auth/login", json={"username": "admin", "password": "admin123"})
        cls.h = {"Authorization": f"Bearer {r.json()['token']}"}

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    # --- auth -----------------------------------------------------------
    def test_login_rejects_bad_password(self):
        r = self.client.post("/api/auth/login", json={"username": "admin", "password": "nope"})
        self.assertEqual(r.status_code, 401)

    def test_endpoints_require_token(self):
        self.assertEqual(self.client.get("/api/incidents").status_code, 401)
        self.assertEqual(self.client.get("/api/cameras").status_code, 401)

    def test_me_and_users(self):
        self.assertEqual(self.client.get("/api/auth/me", headers=self.h).json()["role"], "admin")
        r = self.client.post("/api/auth/users", headers=self.h,
                             json={"username": "operator1", "password": "secret12", "role": "operator"})
        self.assertIn(r.status_code, (201, 409))

    # --- cameras --------------------------------------------------------
    def test_camera_crud_and_validation(self):
        body = {"camera_name": "Test Cam", "zone_name": "Z", "source_type": "file", "source": "x.mp4",
                "restricted_region": [[0, 0], [10, 0], [10, 10]], "tier": 3}
        r = self.client.post("/api/cameras", headers=self.h, json=body)
        self.assertEqual(r.status_code, 201, r.text)
        cid = r.json()["id"]
        self.assertEqual(self.client.post("/api/cameras", headers=self.h, json=body).status_code, 409)
        bad = dict(body, camera_name="Bad", restricted_region=[[0, 0], [1, 1]])
        self.assertEqual(self.client.post("/api/cameras", headers=self.h, json=bad).status_code, 422)
        r = self.client.put(f"/api/cameras/{cid}", headers=self.h, json=dict(body, dwell_threshold=6))
        self.assertEqual(r.json()["dwell_threshold"], 6)
        self.assertEqual(self.client.get(f"/api/cameras/{cid}/snapshot", headers=self.h).status_code, 422)
        self.assertEqual(self.client.delete(f"/api/cameras/{cid}", headers=self.h).status_code, 204)

    # --- incidents ------------------------------------------------------
    def _wait_report(self, iid, timeout=10):
        end = time.time() + timeout
        while time.time() < end:
            inc = self.client.get(f"/api/incidents/{iid}", headers=self.h).json()
            if inc["report_status"] in ("ready", "failed"):
                return inc
            time.sleep(0.3)
        return inc

    def test_ingest_escalates_and_reports(self):
        inc = dict(SAMPLE, incident_ref="t-esc-1", confidence=0.93, duration_seconds=14.0,
                   features={"person_count": 1, "density": 0.035, "occupancy": False, "motion_score": 0.0015,
                             "zone_dwell_seconds": 4.1, "max_confidence": 0.93})  # standing in the zone
        r = self.client.post("/api/incidents/ingest", json=inc)
        self.assertEqual(r.status_code, 202, r.text)
        out = r.json()
        self.assertEqual(out["decision"], "escalated")
        self.assertIn("job_id", out)
        if rag.available:
            full = self._wait_report(out["incident_id"])
            self.assertEqual(full["report_status"], "ready")
            self.assertEqual(full["report"]["suggested_action"], "stub action")
        # idempotent
        self.assertEqual(self.client.post("/api/incidents/ingest", json=inc).status_code, 200)

    def test_classifier_suppresses_obvious_false_alarm(self):
        inc = dict(SAMPLE, incident_ref="t-sup-1", confidence=0.42, duration_seconds=4.1,
                   features={"person_count": 1, "density": 0.03, "occupancy": False, "motion_score": 0.0008,
                             "zone_dwell_seconds": 4.1, "max_confidence": 0.42})  # low-confidence misdetection
        out = self.client.post("/api/incidents/ingest", json=inc).json()
        self.assertEqual(out["decision"], "suppressed")
        iid = out["incident_id"]
        self.assertEqual(self.client.get(f"/api/incidents/{iid}", headers=self.h).json()["status"], "suppressed")
        # operator can override
        r = self.client.post(f"/api/incidents/{iid}/report", headers=self.h)
        self.assertEqual(r.json()["status"], "open")

    def test_status_workflow_and_stats(self):
        out = self.client.post("/api/incidents/ingest", json=dict(SAMPLE, incident_ref="t-wf-1")).json()
        iid = out["incident_id"]
        for s in ("acknowledged", "resolved"):
            r = self.client.patch(f"/api/incidents/{iid}", headers=self.h, json={"status": s})
            self.assertEqual(r.json()["status"], s)
        self.assertEqual(self.client.patch(f"/api/incidents/{iid}", headers=self.h,
                                           json={"status": "bogus"}).status_code, 422)
        st = self.client.get("/api/incidents/stats", headers=self.h).json()
        self.assertGreaterEqual(st["total"], 1)

    def test_rejects_malformed_incident(self):
        self.assertEqual(self.client.post("/api/incidents/ingest", json={"event_type": "x"}).status_code, 422)
        bad = dict(SAMPLE, confidence=1.5)
        self.assertEqual(self.client.post("/api/incidents/ingest", json=bad).status_code, 422)

    # --- federated / chat / system ---------------------------------------
    def test_federated_endpoints(self):
        self.assertIn("loaded", self.client.get("/api/federated/model", headers=self.h).json())
        self.assertIn("strategies", self.client.get("/api/federated/rounds", headers=self.h).json())
        self.assertIn("state", self.client.get("/api/federated/status", headers=self.h).json())
        r = self.client.post("/api/federated/train", headers=self.h, json={"rounds": 0})
        self.assertEqual(r.status_code, 422)

    def test_chat(self):
        r = self.client.post("/api/chat", headers=self.h, json={"question": "what now?"})
        self.assertEqual(r.status_code, 200 if rag.available else 503)
        if rag.available:
            self.assertIn("source_section", r.json())

    def test_system_metrics(self):
        m = self.client.get("/api/system/metrics", headers=self.h).json()
        for key in ("process", "system", "cameras", "latency"):
            self.assertIn(key, m)
        self.assertIsNotNone(m["latency"]["decision_ms"])  # set by the ingest tests

    def test_system_status(self):
        s = self.client.get("/api/system/status", headers=self.h).json()
        for key in ("vision", "classifier", "rag", "incidents"):
            self.assertIn(key, s)


if __name__ == "__main__":
    unittest.main(verbosity=2)
