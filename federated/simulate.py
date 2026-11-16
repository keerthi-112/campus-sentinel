"""
simulate.py — Run the full federated-learning experiment in one process.

What it does
------------
1. Makes sure every simulated zone has its local dataset (data.py).
2. Baselines:
     * local-only   — each zone trains alone on its own data
     * centralized  — all data pooled (upper bound; NOT privacy-preserving,
                      reported only for comparison)
3. Federated training with FedAvg and/or FedProx for N rounds. Per round we
   log: round number, strategy, participating clients, global accuracy / F1 /
   loss / AUC on a campus-wide held-out test set, and communication cost.
4. Saves the deployed global classifier to artifacts/global_model.json (written
   atomically, so the live system only ever loads a finished round), plus
   artifacts/rounds.json and artifacts/summary.json for the dashboard.

This in-process engine implements the same FedAvg / FedProx maths as the
Flower run (run_flower.py) — it exists so experiments are fast, reproducible
and runnable without opening network ports. Use run_flower.py to show the real
client/server deployment over gRPC.

    python -m federated.simulate                       # both strategies, 5 zones
    python -m federated.simulate --strategy fedavg --rounds 30 --clients 3
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import ensure_data, list_clients, load_client, load_test
from .features import records_to_xy
from .model import EventMLP, TrainConfig, evaluate, save_model_json

FED_DIR = Path(__file__).resolve().parent
ARTIFACTS = FED_DIR / "artifacts"
MODEL_PATH = ARTIFACTS / "global_model.json"
ROUNDS_PATH = ARTIFACTS / "rounds.json"
SUMMARY_PATH = ARTIFACTS / "summary.json"
STATUS_PATH = ARTIFACTS / "training_status.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def set_status(**kw) -> None:
    current = {}
    if STATUS_PATH.exists():
        try:
            current = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except Exception:
            current = {}
    current.update(kw, updated_at=_now())
    write_json_atomic(STATUS_PATH, current)


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------
def fedavg_aggregate(results: list[tuple[list[np.ndarray], int]]) -> list[np.ndarray]:
    """Weighted average of client parameters, weight = number of local samples.

    FedProx uses the same server-side aggregation; its difference is the
    proximal term each client adds to its local loss (see model.EventMLP.fit).
    """
    total = sum(n for _, n in results)
    return [
        sum(params[i] * (n / total) for params, n in results).astype(np.float32)
        for i in range(len(results[0][0]))
    ]


# ----------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------
def train_early_stopped(X, y, max_epochs: int, cfg: TrainConfig, seed: int, val_frac: float = 0.15):
    """Train on (X, y) with a held-out validation split; keep the epoch with the lowest
    validation loss. Used for the baselines so they are not handicapped by over-training."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = max(int(len(X) * val_frac), 10)
    val, tr = idx[:n_val], idx[n_val:]
    m = EventMLP(seed=seed)
    best, best_loss, since = m.get_params(), float("inf"), 0
    for ep in range(max_epochs):
        m.fit(X[tr], y[tr], TrainConfig(epochs=1, lr=cfg.lr, batch_size=cfg.batch_size, seed=seed + ep))
        vl = evaluate(m, X[val], y[val])["loss"]
        if vl < best_loss - 1e-4:
            best, best_loss, since = m.get_params(), vl, 0
        else:
            since += 1
            if since >= 8:  # patience
                break
    m.set_params(best)
    return m


def train_local_only(clients: dict, Xt, yt, epochs: int, cfg: TrainConfig, seed: int) -> dict:
    out = {}
    for cid, (X, y) in clients.items():
        m = train_early_stopped(X, y, epochs, cfg, seed)
        out[cid] = {"global_test": evaluate(m, Xt, yt), "own_data": evaluate(m, X, y), "n_train": int(len(X))}
    return out


def train_centralized(clients: dict, Xt, yt, epochs: int, cfg: TrainConfig, seed: int) -> dict:
    X = np.concatenate([c[0] for c in clients.values()])
    y = np.concatenate([c[1] for c in clients.values()])
    m = train_early_stopped(X, y, epochs, cfg, seed)
    return {"global_test": evaluate(m, Xt, yt), "n_train": int(len(X))}


# ----------------------------------------------------------------------
# Federated loop
# ----------------------------------------------------------------------
def run_federated(clients: dict, Xt, yt, *, strategy: str, rounds: int, local_epochs: int,
                  lr: float, batch_size: int, mu: float, fraction_fit: float,
                  heterogeneous: bool, seed: int, progress=None, on_round=None):
    rng = np.random.default_rng(seed)
    global_model = EventMLP(seed=seed)
    model_bytes = global_model.n_bytes()
    history = []
    cumulative = 0
    cids = list(clients)

    for rnd in range(1, rounds + 1):
        t0 = time.perf_counter()
        k = max(1, int(round(fraction_fit * len(cids))))
        selected = sorted(rng.choice(cids, size=k, replace=False).tolist())
        g_params = global_model.get_params()

        results, client_metrics = [], {}
        for j, cid in enumerate(selected):
            X, y = clients[cid]
            local = EventMLP(seed=seed)
            local.set_params(g_params)
            # Systems heterogeneity: slow zones manage fewer local epochs.
            ep = int(rng.integers(1, local_epochs + 1)) if heterogeneous else local_epochs
            loss = local.fit(X, y, TrainConfig(
                epochs=ep, lr=lr, batch_size=batch_size,
                proximal_mu=mu if strategy == "fedprox" else 0.0,
                seed=seed * 1000 + rnd * 10 + j), global_params=g_params)
            results.append((local.get_params(), len(X)))
            client_metrics[cid] = {"n": int(len(X)), "epochs": ep, "train_loss": round(loss, 4)}

        global_model.set_params(fedavg_aggregate(results))
        metrics = evaluate(global_model, Xt, yt)
        # each selected client downloads the global model and uploads its update
        round_bytes = 2 * model_bytes * len(selected)
        cumulative += round_bytes
        rec = {
            "round_number": rnd,
            "strategy": strategy,
            "participating_clients": selected,
            "num_clients": len(selected),
            "metrics": metrics,
            "client_metrics": client_metrics,
            "comm_bytes": round_bytes,
            "cumulative_comm_bytes": cumulative,
            "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
            "timestamp": _now(),
        }
        history.append(rec)
        if progress:
            progress(strategy, rnd, rounds, metrics)
        if on_round:
            on_round(strategy, history)
    return global_model, history


def rounds_to_converge(history: list[dict], frac: float = 0.98) -> int:
    final = history[-1]["metrics"]["accuracy"]
    for rec in history:
        if rec["metrics"]["accuracy"] >= frac * final:
            return rec["round_number"]
    return history[-1]["round_number"]


# ----------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------
def run_experiment(strategy: str = "both", rounds: int = 20, n_clients: int = 5,
                   local_epochs: int = 3, lr: float = 0.01, batch_size: int = 32,
                   mu: float = 0.05, fraction_fit: float = 1.0, heterogeneous: bool = True,
                   deploy: str = "fedavg", seed: int = 42, quiet: bool = False) -> dict:
    run_id = uuid.uuid4().hex[:8]
    set_status(state="running", run_id=run_id, started_at=_now(), current_round=0,
               total_rounds=rounds, strategy=strategy, error=None)
    try:
        ensure_data(n_clients=n_clients)
        cids = list_clients()[:n_clients]
        clients = {cid: records_to_xy(load_client(cid)) for cid in cids}
        Xt, yt = records_to_xy(load_test())
        cfg = TrainConfig(lr=lr, batch_size=batch_size)

        def progress(strat, rnd, total, m):
            set_status(current_round=rnd, total_rounds=total, current_strategy=strat,
                       last_accuracy=m["accuracy"])
            if not quiet:
                print(f"  [{strat:7s}] round {rnd:3d}/{total}  acc={m['accuracy']:.3f}  "
                      f"f1={m['f1']:.3f}  loss={m['loss']:.3f}")

        if not quiet:
            print(f"Clients: {', '.join(f'{c} ({len(clients[c][1])})' for c in cids)}  |  test n={len(yt)}")
        # Baselines use the same total amount of local computation as FL.
        local_only = train_local_only(clients, Xt, yt, rounds * local_epochs, cfg, seed)
        centralized = train_centralized(clients, Xt, yt, rounds * local_epochs, cfg, seed)

        strategies = ["fedavg", "fedprox"] if strategy == "both" else [strategy]
        histories, models = {}, {}
        live_cfg = {"strategy": strategy, "rounds": rounds, "clients": cids, "deployed": deploy,
                    "local_epochs": local_epochs, "proximal_mu": mu, "live": True}

        def on_round(strat, hist):
            # written every round so the dashboard chart grows while training runs
            write_json_atomic(ROUNDS_PATH, {"run_id": run_id, "engine": "simulation", "created_at": _now(),
                                            "config": live_cfg, "strategies": {**histories, strat: hist}})

        for s in strategies:
            models[s], histories[s] = run_federated(
                clients, Xt, yt, strategy=s, rounds=rounds, local_epochs=local_epochs, lr=lr,
                batch_size=batch_size, mu=mu, fraction_fit=fraction_fit,
                heterogeneous=heterogeneous, seed=seed, progress=progress, on_round=on_round)

        deploy = deploy if deploy in models else strategies[0]
        final = histories[deploy][-1]
        model_bytes = models[deploy].n_bytes()

        save_model_json(models[deploy], MODEL_PATH, extra={
            "model_version": f"{run_id}-r{final['round_number']}",
            "run_id": run_id,
            "strategy": deploy,
            "round_number": final["round_number"],
            "participating_clients": final["participating_clients"],
            "metrics": final["metrics"],
            "threshold": 0.5,
            "engine": "simulation",
            "created_at": _now(),
        })

        config = {"strategy": strategy, "rounds": rounds, "clients": cids, "local_epochs": local_epochs,
                  "lr": lr, "batch_size": batch_size, "proximal_mu": mu, "fraction_fit": fraction_fit,
                  "heterogeneous": heterogeneous, "deployed": deploy, "seed": seed,
                  "model_params": models[deploy].n_params(), "model_bytes": model_bytes}
        write_json_atomic(ROUNDS_PATH, {"run_id": run_id, "engine": "simulation",
                                        "created_at": _now(), "config": config,
                                        "strategies": histories})

        local_acc = [v["global_test"]["accuracy"] for v in local_only.values()]
        # How well the deployed global model does on each zone's OWN data vs. that zone alone
        zone_accuracy = {
            cid: {"n": int(len(clients[cid][1])),
                  "federated_own": evaluate(models[deploy], *clients[cid])["accuracy"],
                  "local_only_own": local_only[cid]["own_data"]["accuracy"],
                  "local_only_global": local_only[cid]["global_test"]["accuracy"]}
            for cid in cids}
        summary = {
            "run_id": run_id,
            "created_at": _now(),
            "config": config,
            "centralized": centralized,
            "local_only": local_only,
            "local_only_mean_accuracy": round(float(np.mean(local_acc)), 4),
            "zone_accuracy": zone_accuracy,
            "federated": {
                s: {
                    "final": h[-1]["metrics"],
                    "best_accuracy": max(r["metrics"]["accuracy"] for r in h),
                    "rounds_to_98pct": rounds_to_converge(h),
                    "total_comm_bytes": h[-1]["cumulative_comm_bytes"],
                } for s, h in histories.items()
            },
            "privacy_note": "Only model parameters "
                            f"({model_bytes} bytes per transfer) left each zone; raw "
                            "events and video never did.",
        }
        write_json_atomic(SUMMARY_PATH, summary)
        set_status(state="done", finished_at=_now(), model_version=f"{run_id}-r{final['round_number']}")
        return summary
    except Exception as exc:
        set_status(state="failed", error=str(exc), finished_at=_now())
        raise


def print_summary(s: dict) -> None:
    print("\n" + "=" * 64)
    print("FEDERATED LEARNING RESULTS (campus-wide held-out test set)")
    print("=" * 64)
    print(f"{'Local-only (per zone)':34s}")
    for cid, v in s["local_only"].items():
        print(f"   {cid:12s} n={v['n_train']:5d}   acc={v['global_test']['accuracy']:.3f}  "
              f"f1={v['global_test']['f1']:.3f}")
    print(f"{'Local-only mean':34s} acc={s['local_only_mean_accuracy']:.3f}")
    for strat, v in s["federated"].items():
        f = v["final"]
        print(f"{strat.upper() + ' (federated)':34s} acc={f['accuracy']:.3f}  f1={f['f1']:.3f}  "
              f"auc={f['auc']:.3f}  conv@{v['rounds_to_98pct']}  comm={v['total_comm_bytes']/1024:.1f} KiB")
    c = s["centralized"]["global_test"]
    print(f"{'Centralized (upper bound)':34s} acc={c['accuracy']:.3f}  f1={c['f1']:.3f}  auc={c['auc']:.3f}")
    print(f"\nDeployed: {s['config']['deployed']}  ->  {MODEL_PATH.relative_to(FED_DIR.parent)}")
    print(s["privacy_note"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Campus Sentinel — federated event-classifier experiment")
    ap.add_argument("--strategy", default="both", choices=["fedavg", "fedprox", "both"])
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--clients", type=int, default=5, choices=[3, 4, 5])
    ap.add_argument("--local-epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--mu", type=float, default=0.05, help="FedProx proximal coefficient")
    ap.add_argument("--fraction-fit", type=float, default=1.0, help="share of zones per round")
    ap.add_argument("--homogeneous", action="store_true",
                    help="every zone runs the full local epochs (default: stragglers do fewer)")
    ap.add_argument("--deploy", default="fedavg", choices=["fedavg", "fedprox"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    s = run_experiment(a.strategy, a.rounds, a.clients, a.local_epochs, a.lr, a.batch_size, a.mu,
                       a.fraction_fit, not a.homogeneous, a.deploy, a.seed, a.quiet)
    print_summary(s)


if __name__ == "__main__":
    main()
