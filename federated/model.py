"""
model.py — The small event classifier that is trained federatedly.

A one-hidden-layer MLP written in plain NumPy:

    7 features -> 16 ReLU units -> 1 sigmoid output  (P(escalate))

Why NumPy and not PyTorch: the model has ~145 parameters, trains in
milliseconds, and keeping it dependency-free means every zone client (and the
backend that serves it) can run it anywhere. Its parameters are a plain list of
NumPy arrays, which is exactly the "NDArrays" format Flower moves around.

Local training supports a FedProx proximal term: the loss gets an extra
    (mu / 2) * ||w - w_global||^2
which keeps a client's weights from drifting too far from the global model when
zones have very different (non-IID) data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .features import FEATURE_NAMES, N_FEATURES

Params = list[np.ndarray]


@dataclass
class TrainConfig:
    epochs: int = 3
    lr: float = 0.01
    batch_size: int = 32
    proximal_mu: float = 0.0  # 0 => plain FedAvg local training
    pos_weight: float = 1.0
    seed: int = 0


class EventMLP:
    def __init__(self, n_in: int = N_FEATURES, hidden: int = 16, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.n_in = n_in
        self.hidden = hidden
        self.W1 = (rng.standard_normal((n_in, hidden)) * np.sqrt(2.0 / n_in)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = (rng.standard_normal((hidden, 1)) * np.sqrt(1.0 / hidden)).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)

    # ------------------------------------------------------------------
    # Parameter exchange (what travels between client and server)
    # ------------------------------------------------------------------
    def get_params(self) -> Params:
        return [self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy()]

    def set_params(self, params: Params) -> None:
        self.W1, self.b1, self.W2, self.b2 = (np.asarray(p, dtype=np.float32).copy() for p in params)

    def n_params(self) -> int:
        return sum(p.size for p in self.get_params())

    def n_bytes(self) -> int:
        """Bytes needed to send the model once (float32)."""
        return sum(p.nbytes for p in self.get_params())

    # ------------------------------------------------------------------
    # Forward / predict
    # ------------------------------------------------------------------
    def _forward(self, X: np.ndarray):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ self.W2 + self.b2
        p = 1.0 / (1.0 + np.exp(-np.clip(z2, -30, 30)))
        return z1, a1, p.reshape(-1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float32))
        return self._forward(X)[2]

    # ------------------------------------------------------------------
    # Local training (Adam, optional FedProx proximal term)
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, cfg: TrainConfig,
            global_params: Params | None = None) -> float:
        """Train in place. Returns the mean training loss of the last epoch."""
        if len(X) == 0:
            return float("nan")
        rng = np.random.default_rng(cfg.seed)
        params = [self.W1, self.b1, self.W2, self.b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1_, b2_, eps = 0.9, 0.999, 1e-8
        step = 0
        anchor = [np.asarray(g, dtype=np.float32) for g in global_params] if global_params else None
        last_loss = 0.0

        for _ in range(cfg.epochs):
            order = rng.permutation(len(X))
            losses = []
            for start in range(0, len(X), cfg.batch_size):
                idx = order[start:start + cfg.batch_size]
                xb, yb = X[idx], y[idx]
                z1, a1, p = self._forward(xb)
                w = np.where(yb > 0.5, cfg.pos_weight, 1.0).astype(np.float32)
                pc = np.clip(p, 1e-7, 1 - 1e-7)
                losses.append(float(np.mean(-w * (yb * np.log(pc) + (1 - yb) * np.log(1 - pc)))))

                # Backprop of weighted BCE through sigmoid
                dz2 = (w * (p - yb) / len(xb)).reshape(-1, 1).astype(np.float32)
                gW2 = a1.T @ dz2
                gb2 = dz2.sum(axis=0)
                da1 = dz2 @ self.W2.T
                dz1 = da1 * (z1 > 0)
                gW1 = xb.T @ dz1
                gb1 = dz1.sum(axis=0)
                grads = [gW1, gb1, gW2, gb2]

                if anchor is not None and cfg.proximal_mu > 0:
                    grads = [g + cfg.proximal_mu * (pp - a) for g, pp, a in zip(grads, params, anchor)]

                step += 1
                for i, (pp, g) in enumerate(zip(params, grads)):
                    m[i] = b1_ * m[i] + (1 - b1_) * g
                    v[i] = b2_ * v[i] + (1 - b2_) * g * g
                    mh = m[i] / (1 - b1_ ** step)
                    vh = v[i] / (1 - b2_ ** step)
                    pp -= (cfg.lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)
            last_loss = float(np.mean(losses))
        return last_loss

    # ------------------------------------------------------------------
    # Serialisation (what the backend loads)
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "architecture": f"MLP {self.n_in}-{self.hidden}-1 (ReLU, sigmoid)",
            "feature_names": FEATURE_NAMES,
            "hidden": self.hidden,
            "params": [p.tolist() for p in self.get_params()],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EventMLP":
        model = cls(n_in=len(d["feature_names"]), hidden=int(d["hidden"]))
        model.set_params([np.array(p, dtype=np.float32) for p in d["params"]])
        return model


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def _auc(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y > 0.5], s[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evaluate(model: EventMLP, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict:
    if len(X) == 0:
        return {"accuracy": float("nan"), "precision": float("nan"), "recall": float("nan"),
                "f1": float("nan"), "loss": float("nan"), "auc": float("nan"), "n": 0}
    p = model.predict_proba(X)
    pred = (p >= threshold).astype(np.float32)
    tp = float(np.sum((pred == 1) & (y == 1)))
    fp = float(np.sum((pred == 1) & (y == 0)))
    fn = float(np.sum((pred == 0) & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    loss = float(np.mean(-(y * np.log(pc) + (1 - y) * np.log(1 - pc))))
    return {
        "accuracy": round(float(np.mean(pred == y)), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "loss": round(loss, 4),
        "auc": round(_auc(y, p), 4),
        "n": int(len(y)),
    }


def save_model_json(model: EventMLP, path, extra: dict | None = None) -> None:
    """Atomically write the model so a reader never sees a half-written file."""
    import os
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**model.to_dict(), **(extra or {})}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    os.replace(tmp, path)
