"""
test_federated.py — Tests for the federated learning track.

    python -m federated.test_federated

Note: the end-to-end test runs a short 4-round experiment, which replaces
artifacts/. Run `python -m federated.simulate` afterwards to restore the full run.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from federated.classifier import EventClassifier
from federated.data import ZONE_PROFILES, generate_zone
from federated.features import FEATURE_NAMES, N_FEATURES, records_to_xy, to_vector
from federated.model import EventMLP, TrainConfig, evaluate, save_model_json
from federated.simulate import fedavg_aggregate

REPO = Path(__file__).resolve().parent.parent
SCHEMA_FV = json.loads((REPO / "docs/schemas/event_feature_vector.json").read_text())
SCHEMA_FV.pop("_description", None)


class TestFeatures(unittest.TestCase):
    def test_schema_example_vectorises(self):
        v = to_vector(SCHEMA_FV, "restricted_area_entry")
        self.assertEqual(v.shape, (N_FEATURES,))
        self.assertEqual(len(FEATURE_NAMES), N_FEATURES)
        self.assertTrue(np.all(np.isfinite(v)))
        self.assertEqual(v[-1], 0.0)

    def test_crowd_flag(self):
        self.assertEqual(to_vector(SCHEMA_FV, "crowd_formation")[-1], 1.0)

    def test_missing_fields_do_not_crash(self):
        self.assertEqual(to_vector({}).shape, (N_FEATURES,))


class TestModel(unittest.TestCase):
    def setUp(self):
        recs = generate_zone(ZONE_PROFILES[0], 600, seed=1)
        self.X, self.y = records_to_xy(recs)

    def test_training_improves_accuracy(self):
        m = EventMLP(seed=0)
        before = evaluate(m, self.X, self.y)["loss"]
        m.fit(self.X, self.y, TrainConfig(epochs=30))
        after = evaluate(m, self.X, self.y)
        self.assertLess(after["loss"], before)
        self.assertGreater(after["accuracy"], 0.75)

    def test_param_roundtrip(self):
        m = EventMLP(seed=3)
        m2 = EventMLP.from_dict(json.loads(json.dumps(m.to_dict())))
        np.testing.assert_allclose(m.predict_proba(self.X[:5]), m2.predict_proba(self.X[:5]), rtol=1e-6)

    def test_fedprox_stays_closer_to_global(self):
        g = EventMLP(seed=0)
        anchor = g.get_params()
        plain, prox = EventMLP(seed=0), EventMLP(seed=0)
        plain.fit(self.X, self.y, TrainConfig(epochs=10, proximal_mu=0.0), global_params=anchor)
        prox.fit(self.X, self.y, TrainConfig(epochs=10, proximal_mu=1.0), global_params=anchor)
        dist = lambda m: sum(float(np.sum((a - b) ** 2)) for a, b in zip(m.get_params(), anchor))
        self.assertLess(dist(prox), dist(plain))


class TestAggregation(unittest.TestCase):
    def test_fedavg_is_sample_weighted(self):
        a = [np.ones((2, 2), np.float32)]
        b = [np.zeros((2, 2), np.float32)]
        out = fedavg_aggregate([(a, 300), (b, 100)])
        np.testing.assert_allclose(out[0], 0.75)


class TestClassifier(unittest.TestCase):
    def test_fallback_escalates_without_model(self):
        clf = EventClassifier(model_path="/nonexistent/model.json")
        d = clf.decide(SCHEMA_FV, "restricted_area_entry")
        self.assertTrue(d["escalate"])
        self.assertIsNone(d["score"])

    def test_hot_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.json"
            clf = EventClassifier(model_path=path)
            self.assertFalse(clf.loaded)
            save_model_json(EventMLP(seed=1), path, extra={"model_version": "v1", "threshold": 0.5})
            self.assertTrue(clf.loaded)
            self.assertEqual(clf.info()["model_version"], "v1")
            time.sleep(0.05)
            save_model_json(EventMLP(seed=2), path, extra={"model_version": "v2", "threshold": 0.5})
            d = clf.decide(SCHEMA_FV)
            self.assertEqual(d["model_version"], "v2")
            self.assertIsInstance(d["score"], float)


class TestEndToEnd(unittest.TestCase):
    def test_simulation_cli(self):
        out = subprocess.run([sys.executable, "-m", "federated.simulate", "--rounds", "4",
                              "--clients", "3", "--strategy", "both", "--quiet"],
                             cwd=REPO, capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        rounds = json.loads((REPO / "federated/artifacts/rounds.json").read_text())
        self.assertEqual(set(rounds["strategies"]), {"fedavg", "fedprox"})
        rec = rounds["strategies"]["fedavg"][-1]
        for key in ("round_number", "strategy", "participating_clients", "metrics", "comm_bytes"):
            self.assertIn(key, rec)
        self.assertTrue(EventClassifier().loaded)


if __name__ == "__main__":
    unittest.main(verbosity=2)
