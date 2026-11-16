"""
training.py — Start federated training from the dashboard and mirror its
results into the database.

Training runs as a separate process (`python -m federated.simulate` or
`python -m federated.run_flower`) so it never blocks live video monitoring.
When it finishes, the new artifacts/global_model.json is picked up by the
classifier on its next decision, i.e. a new global model takes effect only
after a round sequence finishes cleanly.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

from ..config import DATA_DIR, FEDERATED_DIR, REPO_ROOT

ARTIFACTS = FEDERATED_DIR / "artifacts"
ROUNDS_PATH = ARTIFACTS / "rounds.json"
SUMMARY_PATH = ARTIFACTS / "summary.json"
STATUS_PATH = ARTIFACTS / "training_status.json"
LOG_PATH = DATA_DIR / "training.log"


def _read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


class TrainingManager:
    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.last_request: dict | None = None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, req: dict) -> dict:
        with self._lock:
            if self.running:
                raise RuntimeError("a training run is already in progress")
            if req["engine"] == "flower":
                strategy = "fedavg" if req["strategy"] == "both" else req["strategy"]
                cmd = [sys.executable, "-m", "federated.run_flower", "--strategy", strategy,
                       "--rounds", str(req["rounds"]), "--clients", str(req["clients"]),
                       "--local-epochs", str(req["local_epochs"]), "--mu", str(req["mu"])]
            else:
                cmd = [sys.executable, "-m", "federated.simulate", "--strategy", req["strategy"],
                       "--rounds", str(req["rounds"]), "--clients", str(req["clients"]),
                       "--local-epochs", str(req["local_epochs"]), "--mu", str(req["mu"])]
            # reset the status file so the dashboard never shows the previous run as current
            tmp = STATUS_PATH.with_suffix(".json.tmp")
            STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps({"state": "starting", "current_round": 0,
                                       "total_rounds": req["rounds"], "engine": req["engine"]}), encoding="utf-8")
            tmp.replace(STATUS_PATH)
            log = LOG_PATH.open("w", encoding="utf-8")
            self.proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
            self.last_request = req
            threading.Thread(target=self._watch, args=(self.proc,), daemon=True).start()
            return {"started": True, "pid": self.proc.pid, "command": " ".join(cmd[1:])}

    def _watch(self, proc: subprocess.Popen) -> None:
        proc.wait()
        try:
            sync_rounds()
        except Exception:
            pass

    def status(self) -> dict:
        st = _read(STATUS_PATH) or {"state": "idle"}
        st["process_running"] = self.running
        if not self.running and st.get("state") == "running":
            st["state"] = "interrupted"
        if self.proc is not None and self.proc.poll() not in (None, 0):
            st["state"] = "failed"
            try:
                st["log_tail"] = LOG_PATH.read_text(encoding="utf-8")[-1500:]
            except Exception:
                pass
        st["last_request"] = self.last_request
        return st


def latest_rounds() -> dict | None:
    return _read(ROUNDS_PATH)


def latest_summary() -> dict | None:
    return _read(SUMMARY_PATH)


def sync_rounds() -> int:
    """Upsert the latest run's per-round log into federated_rounds."""
    from ..db import SessionLocal
    from ..models import FederatedRound

    data = latest_rounds()
    if not data:
        return 0
    n = 0
    with SessionLocal() as db:
        for strategy, hist in data.get("strategies", {}).items():
            for r in hist:
                exists = db.query(FederatedRound).filter_by(
                    run_id=data["run_id"], strategy=strategy, round_number=r["round_number"]).first()
                if exists:
                    continue
                db.add(FederatedRound(
                    run_id=data["run_id"], engine=data.get("engine", "simulation"),
                    round_number=r["round_number"], strategy=strategy,
                    participating_clients=r.get("participating_clients", []),
                    metrics=r.get("metrics", {}), comm_bytes=int(r.get("comm_bytes", 0)),
                    cumulative_comm_bytes=int(r.get("cumulative_comm_bytes", 0))))
                n += 1
        db.commit()
    return n


training = TrainingManager()
