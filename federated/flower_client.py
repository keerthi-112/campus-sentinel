"""
flower_client.py — One campus zone as a Flower client.

The client loads ONLY its own zone's event features, trains the event
classifier locally when the server asks, and sends back model parameters.
Raw events never leave this process.

    python -m federated.flower_client --zone lab_2 --server 127.0.0.1:8080
"""

from __future__ import annotations

import argparse
import warnings

import flwr as fl

from .data import load_client
from .features import records_to_xy
from .model import EventMLP, TrainConfig, evaluate

warnings.filterwarnings("ignore", category=DeprecationWarning)


class ZoneClient(fl.client.NumPyClient):
    def __init__(self, zone_id: str, seed: int = 42) -> None:
        self.zone_id = zone_id
        self.X, self.y = records_to_xy(load_client(zone_id))
        self.model = EventMLP(seed=seed)
        self.seed = seed

    def get_parameters(self, config):
        return self.model.get_params()

    def fit(self, parameters, config):
        self.model.set_params(parameters)
        global_params = [p.copy() for p in parameters]
        mu = float(config.get("proximal_mu", 0.0))  # set by Flower's FedProx strategy
        loss = self.model.fit(self.X, self.y, TrainConfig(
            epochs=int(config.get("local_epochs", 3)),
            lr=float(config.get("lr", 0.01)),
            batch_size=int(config.get("batch_size", 32)),
            proximal_mu=mu,
            seed=self.seed + int(config.get("server_round", 0)),
        ), global_params=global_params)
        return self.model.get_params(), len(self.X), {"zone": self.zone_id, "train_loss": float(loss)}

    def evaluate(self, parameters, config):
        self.model.set_params(parameters)
        m = evaluate(self.model, self.X, self.y)
        return float(m["loss"]), len(self.X), {"accuracy": float(m["accuracy"]), "zone": self.zone_id}


def main() -> None:
    ap = argparse.ArgumentParser(description="Campus Sentinel Flower zone client")
    ap.add_argument("--zone", required=True, help="zone id, e.g. gate_1, lab_2, parking")
    ap.add_argument("--server", default="127.0.0.1:8080")
    a = ap.parse_args()
    client = ZoneClient(a.zone)
    print(f"[client {a.zone}] {len(client.y)} local samples — connecting to {a.server}")
    fl.client.start_client(server_address=a.server, client=client.to_client())
    print(f"[client {a.zone}] done")


if __name__ == "__main__":
    main()
