"""
fl_real_features.py — Federated learning with features from the REAL vision pipeline.

Answers: "the FL classifier was trained on synthetic zone data — does it work
on what the vision pipeline actually produces, and does federating real
pipeline data in help?"

Steps (all automatic):
  1. Train the synthetic-only federated model (5 simulated zones).
  2. Run the vision pipeline on the TEST clips → ground-truth-labelled feature
     samples (never used for training) + the event metrics with that model.
  3. Run the vision pipeline on separate TRAINING clips (jittered takes) →
     labelled samples saved as an extra zone client: federated/data/real/cam03_lab2.jsonl
  4. Federated training again with 5 synthetic zones + the real-pipeline zone.
  5. Re-run the event evaluation on the test clips with the new global model.

    python -m evaluation.make_scenarios                                   # test clips
    python -m evaluation.make_scenarios --out evaluation/clips_train --variants 3
    python -m evaluation.fl_real_features

The model deployed at the end is the one trained WITH the real-pipeline zone.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EVAL = REPO / "evaluation"
RES = EVAL / "results"
ART = REPO / "federated" / "artifacts"
REAL_DIR = REPO / "federated" / "data" / "real"


def run(*args: str) -> None:
    print("$", " ".join(args))
    subprocess.run([sys.executable, *args], cwd=REPO, check=True)


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--test-clips", default=str(EVAL / "clips"))
    ap.add_argument("--train-clips", default=str(EVAL / "clips_train"))
    ap.add_argument("--model", default="yolo26n.pt")
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    real_test = RES / "fl_real_test_samples.jsonl"

    # 1. synthetic-only federated model
    run("-m", "federated.simulate", "--quiet", "--no-real")
    shutil.copy(ART / "summary.json", RES / "fl_summary_synthetic_only.json")
    shutil.copy(ART / "global_model.json", RES / "global_model_synthetic_only.json")
    # 2. test clips with the synthetic-only model
    run("-m", "evaluation.eval_events", "--clips", a.test_clips, "--model", a.model,
        "--tag", "fl_before", "--export-fl", str(real_test))
    # 3. training clips -> extra zone client
    run("-m", "evaluation.eval_events", "--clips", a.train_clips, "--model", a.model,
        "--tag", "train_clips", "--export-fl", str(REAL_DIR / "cam03_lab2.jsonl"))
    # 4. synthetic zones + real-pipeline zone, scored on the held-out real samples
    run("-m", "federated.simulate", "--quiet", "--real-dir", str(REAL_DIR), "--real-test", str(real_test))
    shutil.copy(ART / "summary.json", RES / "fl_summary_with_real_zone.json")
    # 5. test clips with the new global model
    run("-m", "evaluation.eval_events", "--clips", a.test_clips, "--model", a.model, "--tag", "fl_after")

    # synthetic-only model on the real test samples (scored here so before/after use the same samples)
    sys.path.insert(0, str(REPO))
    from federated.data import read_jsonl
    from federated.features import records_to_xy
    from federated.model import EventMLP, evaluate

    synth = load(RES / "fl_summary_synthetic_only.json")
    with_real = load(RES / "fl_summary_with_real_zone.json")
    before_ev = load(RES / "events_fl_before.json")["with_fl_classifier"]
    after_ev = load(RES / "events_fl_after.json")["with_fl_classifier"]
    vision_ev = load(RES / "events_fl_before.json")["vision_only"]
    Xr, yr = records_to_xy(read_jsonl(real_test))
    before_samples = None
    snap = RES / "global_model_synthetic_only.json"
    if snap.exists():
        before_samples = evaluate(EventMLP.from_dict(load(snap)), Xr, yr)

    out = {
        "real_test_samples": int(len(yr)),
        "real_train_samples": sum(1 for _ in open(REAL_DIR / "cam03_lab2.jsonl", encoding="utf-8")),
        "synthetic_only_model_on_real_samples": before_samples,
        "with_real_zone_on_real_samples": with_real.get("real_pipeline_test"),
        "synthetic_test_accuracy": {
            "synthetic_only_fedavg": synth["federated"]["fedavg"]["final"]["accuracy"],
            "with_real_zone_fedavg": with_real["federated"]["fedavg"]["final"]["accuracy"]},
        "events_on_test_clips": {
            "vision_only": vision_ev,
            "fl_filter_synthetic_only": before_ev,
            "fl_filter_with_real_zone": after_ev},
    }
    (RES / "fl_real.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    z, c = "restricted_area_entry", "crowd_clip_level"
    print("\n=== Federated learning with real vision-pipeline features ===")
    print(f"training samples from pipeline: {out['real_train_samples']}, held-out test samples: {out['real_test_samples']}")
    if before_samples:
        print(f"accuracy on real samples: synthetic-only {before_samples['accuracy']:.3f} -> "
              f"with real zone {with_real['real_pipeline_test']['fedavg']['accuracy']:.3f}")
    for name, ev in (("vision only", vision_ev), ("FL filter, synthetic-only", before_ev),
                     ("FL filter, with real zone", after_ev)):
        print(f"  {name:28s} restricted P/R {ev[z]['precision']}/{ev[z]['recall']}   crowd F1 {ev[c]['f1']}")
    print(f"-> {(RES / 'fl_real.json').relative_to(REPO)}")


if __name__ == "__main__":
    main()
