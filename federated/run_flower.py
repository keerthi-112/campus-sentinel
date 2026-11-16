"""
run_flower.py — Launch a real Flower deployment on one machine:
one server process + one client process per zone, talking over gRPC.

    python -m federated.run_flower                     # FedAvg, 3 zones, 15 rounds
    python -m federated.run_flower --strategy fedprox --clients 5 --rounds 20

Each client is a separate OS process that can read only its own zone file —
the same shape as a real multi-site deployment, just on localhost.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from .data import ensure_data, list_clients

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Flower server + zone clients locally")
    ap.add_argument("--strategy", default="fedavg", choices=["fedavg", "fedprox"])
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--clients", type=int, default=3, choices=[3, 4, 5])
    ap.add_argument("--local-epochs", type=int, default=3)
    ap.add_argument("--mu", type=float, default=0.05)
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()

    ensure_data(n_clients=a.clients)
    zones = list_clients()[: a.clients]
    py = sys.executable
    addr = f"127.0.0.1:{a.port}"

    server = subprocess.Popen(
        [py, "-m", "federated.flower_server", "--address", f"0.0.0.0:{a.port}",
         "--strategy", a.strategy, "--rounds", str(a.rounds), "--clients", str(a.clients),
         "--local-epochs", str(a.local_epochs), "--mu", str(a.mu)],
        cwd=REPO_ROOT)
    time.sleep(4)  # give gRPC a moment to bind

    clients = [subprocess.Popen([py, "-m", "federated.flower_client", "--zone", z, "--server", addr],
                                cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
               for z in zones]
    print(f"[run_flower] started server + clients: {', '.join(zones)}")

    code = server.wait()
    for c in clients:
        try:
            c.wait(timeout=15)
        except subprocess.TimeoutExpired:
            c.kill()
    sys.exit(code)


if __name__ == "__main__":
    main()
