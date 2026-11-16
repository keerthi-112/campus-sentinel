"""
flower_server.py — Flower coordinating server for the event classifier.

Aggregates zone updates with FedAvg (default) or FedProx, evaluates every new
global model on the campus-wide held-out test set, logs round metrics, and
writes the final global classifier where the live system picks it up.

    python -m federated.flower_server --strategy fedavg --rounds 15 --clients 3

Normally started for you by run_flower.py together with the zone clients.
"""

from __future__ import annotations

import argparse
import time
import uuid
import warnings

import flwr as fl
from flwr.common import ndarrays_to_parameters

from .data import load_test
from .features import records_to_xy
from .model import EventMLP, evaluate, save_model_json
from .simulate import MODEL_PATH, ROUNDS_PATH, SUMMARY_PATH, _now, set_status, write_json_atomic

warnings.filterwarnings("ignore", category=DeprecationWarning)


def main() -> None:
    ap = argparse.ArgumentParser(description="Campus Sentinel Flower server")
    ap.add_argument("--address", default="0.0.0.0:8080")
    ap.add_argument("--strategy", default="fedavg", choices=["fedavg", "fedprox"])
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--clients", type=int, default=3, help="clients required per round")
    ap.add_argument("--local-epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--mu", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    run_id = uuid.uuid4().hex[:8]
    Xt, yt = records_to_xy(load_test())
    model = EventMLP(seed=a.seed)
    model_bytes = model.n_bytes()
    history: list[dict] = []
    state = {"clients": [], "t": time.perf_counter(), "cum": 0}
    set_status(state="running", run_id=run_id, started_at=_now(), current_round=0,
               total_rounds=a.rounds, strategy=a.strategy, engine="flower", error=None)

    def fit_config(server_round: int) -> dict:
        state["t"] = time.perf_counter()
        return {"local_epochs": a.local_epochs, "lr": a.lr, "batch_size": 32, "server_round": server_round}

    def fit_metrics(metrics):
        # Called with [(num_examples, metrics_dict), ...] from every client of the round.
        state["clients"] = sorted(str(m.get("zone", "?")) for _, m in metrics)
        total = sum(n for n, _ in metrics) or 1
        return {"train_loss": sum(n * float(m.get("train_loss", 0)) for n, m in metrics) / total}

    def evaluate_fn(server_round, parameters, config):
        model.set_params(parameters)
        m = evaluate(model, Xt, yt)
        if server_round == 0:
            print(f"[server] initial model  acc={m['accuracy']:.3f}")
            return m["loss"], {"accuracy": m["accuracy"]}
        n_sel = len(state["clients"]) or a.clients
        round_bytes = 2 * model_bytes * n_sel
        state["cum"] += round_bytes
        history.append({
            "round_number": server_round, "strategy": a.strategy,
            "participating_clients": state["clients"], "num_clients": n_sel,
            "metrics": m, "client_metrics": {}, "comm_bytes": round_bytes,
            "cumulative_comm_bytes": state["cum"],
            "duration_ms": round((time.perf_counter() - state["t"]) * 1000, 1), "timestamp": _now(),
        })
        set_status(current_round=server_round, current_strategy=a.strategy, last_accuracy=m["accuracy"])
        write_json_atomic(ROUNDS_PATH, {"run_id": run_id, "engine": "flower", "created_at": _now(),
                                        "config": {"strategy": a.strategy, "rounds": a.rounds, "deployed": a.strategy,
                                                   "live": True},
                                        "strategies": {a.strategy: history}})
        print(f"[server] round {server_round:3d}/{a.rounds}  clients={state['clients']}  "
              f"acc={m['accuracy']:.3f}  f1={m['f1']:.3f}")
        return m["loss"], {"accuracy": m["accuracy"], "f1": m["f1"]}

    common = dict(
        fraction_fit=1.0, fraction_evaluate=0.0,
        min_fit_clients=a.clients, min_available_clients=a.clients, min_evaluate_clients=0,
        initial_parameters=ndarrays_to_parameters(model.get_params()),
        evaluate_fn=evaluate_fn, on_fit_config_fn=fit_config,
        fit_metrics_aggregation_fn=fit_metrics,
    )
    if a.strategy == "fedprox":
        strategy = fl.server.strategy.FedProx(proximal_mu=a.mu, **common)
    else:
        strategy = fl.server.strategy.FedAvg(**common)

    print(f"[server] {a.strategy} | {a.rounds} rounds | waiting for {a.clients} zone clients on {a.address}")
    try:
        fl.server.start_server(server_address=a.address,
                               config=fl.server.ServerConfig(num_rounds=a.rounds),
                               strategy=strategy)
    except Exception as exc:
        set_status(state="failed", error=str(exc), finished_at=_now())
        raise

    if not history:
        set_status(state="failed", error="no rounds completed", finished_at=_now())
        raise SystemExit("[server] no rounds completed")

    final = history[-1]
    version = f"{run_id}-r{final['round_number']}"
    save_model_json(model, MODEL_PATH, extra={
        "model_version": version, "run_id": run_id, "strategy": a.strategy,
        "round_number": final["round_number"], "participating_clients": final["participating_clients"],
        "metrics": final["metrics"], "threshold": 0.5, "engine": "flower", "created_at": _now(),
    })
    config = {"strategy": a.strategy, "rounds": a.rounds, "clients": final["participating_clients"],
              "local_epochs": a.local_epochs, "lr": a.lr, "proximal_mu": a.mu,
              "deployed": a.strategy, "model_params": model.n_params(), "model_bytes": model_bytes,
              "seed": a.seed}
    write_json_atomic(ROUNDS_PATH, {"run_id": run_id, "engine": "flower", "created_at": _now(),
                                    "config": config, "strategies": {a.strategy: history}})
    # Keep any baseline numbers from an earlier simulation run, update the federated row.
    import json
    summary = {}
    if SUMMARY_PATH.exists():
        try:
            summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
    summary.setdefault("federated", {})
    summary["federated"][a.strategy] = {
        "final": final["metrics"],
        "best_accuracy": max(r["metrics"]["accuracy"] for r in history),
        "rounds_to_98pct": next(r["round_number"] for r in history
                                if r["metrics"]["accuracy"] >= 0.98 * final["metrics"]["accuracy"]),
        "total_comm_bytes": final["cumulative_comm_bytes"],
        "engine": "flower",
    }
    summary.update(run_id=run_id, created_at=_now())
    write_json_atomic(SUMMARY_PATH, summary)
    set_status(state="done", finished_at=_now(), model_version=version)
    print(f"[server] saved global model {version} -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
