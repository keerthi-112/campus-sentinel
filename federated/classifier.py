"""
classifier.py — Serve the federated global event classifier at decision time.

This is the "Event Classifier" box in the architecture: given an incident
candidate from vision/ and its event feature vector, decide whether to
ESCALATE (create an incident, write a report) or SUPPRESS it as a false alarm.

The model file is re-read automatically whenever a federated round finishes
and replaces artifacts/global_model.json, so a new global model takes effect
without restarting the live system. Before any training has happened the
classifier falls back to escalating everything (fail-safe: never hide a real
intrusion because no model exists yet).

    from federated.classifier import EventClassifier
    clf = EventClassifier()
    decision = clf.decide(feature_vector, event_type="restricted_area_entry")
    # {'escalate': True, 'score': 0.91, 'model_version': '1a2b3c4d-r20', ...}
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from .features import to_vector
from .model import EventMLP

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "global_model.json"


class EventClassifier:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH, threshold: float | None = None) -> None:
        self.model_path = Path(model_path)
        self._threshold_override = threshold
        self._lock = threading.Lock()
        self._mtime: float | None = None
        self._model: EventMLP | None = None
        self.meta: dict = {}
        self._maybe_reload()

    # ------------------------------------------------------------------
    def _maybe_reload(self) -> None:
        try:
            mtime = self.model_path.stat().st_mtime
        except FileNotFoundError:
            self._model, self.meta, self._mtime = None, {}, None
            return
        if mtime == self._mtime:
            return
        try:
            data = json.loads(self.model_path.read_text(encoding="utf-8"))
            model = EventMLP.from_dict(data)
        except Exception:
            return  # keep the previous model if the file is unreadable
        self._model = model
        self.meta = {k: v for k, v in data.items() if k != "params"}
        self._mtime = mtime

    @property
    def loaded(self) -> bool:
        with self._lock:
            self._maybe_reload()
            return self._model is not None

    @property
    def threshold(self) -> float:
        if self._threshold_override is not None:
            return self._threshold_override
        return float(self.meta.get("threshold", 0.5))

    def info(self) -> dict:
        with self._lock:
            self._maybe_reload()
            return {"loaded": self._model is not None, "path": str(self.model_path),
                    "threshold": self.threshold, **self.meta}

    # ------------------------------------------------------------------
    def score(self, feature_vector: dict, event_type: str | None = None) -> float | None:
        with self._lock:
            self._maybe_reload()
            if self._model is None:
                return None
            return float(self._model.predict_proba(to_vector(feature_vector, event_type))[0])

    def decide(self, feature_vector: dict, event_type: str | None = None) -> dict:
        s = self.score(feature_vector, event_type)
        if s is None:
            return {"escalate": True, "score": None, "threshold": self.threshold,
                    "model_version": None, "reason": "no federated model trained yet — escalating by default"}
        esc = s >= self.threshold
        return {"escalate": esc, "score": round(s, 4), "threshold": self.threshold,
                "model_version": self.meta.get("model_version"),
                "reason": f"classifier score {s:.2f} {'>=' if esc else '<'} threshold {self.threshold:.2f}"}
