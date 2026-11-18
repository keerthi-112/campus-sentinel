# federated/

Owns: the simulated zone clients, the small **event classifier**, the Flower
server, FedAvg + FedProx aggregation, and the per-round metrics.

- **Reads:** `docs/schemas/event_feature_vector.json`-shaped records, one file per zone
  (synthetic by default, or real exports from `vision/`).
- **Produces:** `artifacts/global_model.json`, the global classifier that the live
  system uses to decide **escalate vs. suppress** for every incident candidate,
  plus `artifacts/rounds.json` and `artifacts/summary.json`, which the dashboard's
  FL status page reads.

Only the classifier is federated. YOLO stays pretrained and local. The classifier
uses 6 compact numbers per event, so one model transfer is **516 bytes**, and raw
video or raw events never leave a zone.

## Setup

```bash
# from the repo root
pip install -r federated/requirements.txt
```

## Usage (run everything from the repo root)

```bash
python -m federated.data --clients 5          # (re)generate 5 non-IID zone datasets + global test set
python -m federated.simulate                  # FedAvg + FedProx, 20 rounds, baselines, deploys FedAvg
                                              # (+ any real vision-pipeline zone in data/real/, see below)
python -m federated.plots                     # report figures -> artifacts/plots/*.png
python -m federated.run_flower                # REAL Flower: 1 server + 3 client processes over gRPC
python -m federated.run_flower --strategy fedprox --clients 5 --rounds 20
python -m federated.test_federated            # 10 unit / integration tests
```

Running the Flower pieces by hand (e.g. on 3 laptops):

```bash
python -m federated.flower_server --strategy fedavg --rounds 15 --clients 3   # coordinator
python -m federated.flower_client --zone gate_1  --server <server-ip>:8080   # zone A
python -m federated.flower_client --zone lab_2   --server <server-ip>:8080   # zone B
python -m federated.flower_client --zone parking --server <server-ip>:8080   # zone C
```

### A zone with features from the real vision pipeline

`data/real/cam03_lab2.jsonl` holds 282 candidate feature vectors produced by the
**actual vision pipeline** (`vision/` modules on 45 labelled training clips) with
ground-truth labels. `simulate.py` adds every `data/real/*.jsonl` as an extra zone
client automatically (`--no-real` to skip). To rebuild it and measure the effect
on held-out real clips:

```bash
python -m evaluation.make_scenarios --out evaluation/clips_train --variants 3
python -m evaluation.fl_real_features      # see evaluation/README.md
```

From your own labelled clips: `python -m evaluation.eval_events --clips my_clips --export-fl federated/data/real/my_cam.jsonl`.
From unlabelled `vision.run_demo --export-features` output: `python -m federated.label_vision_features features.jsonl`
(weak rule labels; prefer ground truth).

The dashboard can also start a training run: **Federated → Start training**.

## The model

| | |
|---|---|
| Input (6) | `person_count, density, occupancy, motion_score, max_confidence, is_crowd_event` |
| Not an input | `zone_dwell_seconds`: the classifier runs at the moment the vision rule fires, when dwell (or crowd persistence) has just reached its threshold, so it is constant there and carries no information. The rule already enforces it. |
| Model | MLP 6 → 16 (ReLU) → 1 (sigmoid) = 129 parameters, NumPy only |
| Output | P(real incident). ≥ 0.5 → escalate (create incident + RAG report), else suppress |
| Scaling | fixed, agreed constants (`features.py`) matched to the ranges the vision pipeline really produces (density ≈ 0.02–0.3, motion ≈ 0–0.03 per frame), so zones don't share statistics |
| Local training | Adam, BCE. FedProx adds `mu/2 · ‖w − w_global‖²` |
| Aggregation | sample-weighted average (FedAvg). FedProx uses the same server step |

## Simulated zones (non-IID by design)

| Zone | Camera | Local samples | Real-incident rate | Crowd share | Note |
|---|---|---|---|---|---|
| gate_1 | CAM 01 Main Gate | 1200 | 0.31 | 0.56 | busy, crowd-heavy |
| lab_2 | CAM 03 Lab 2 | 180 | 0.52 | 0.02 | small, almost no crowds |
| parking | CAM 05 Parking B | 420 | 0.38 | 0.17 | dark → lower YOLO confidence |
| library | CAM 07 Atrium | 320 | 0.34 | 0.35 | |
| hostel | CAM 09 Hostel C | 260 | 0.45 | 0.17 | |
| real_cam03_lab2 | CAM 03 (real pipeline) | 282 | 1.00 | 0.46 | from labelled clips, see above |

Each record is a candidate **as the vision rule emits it**, with value ranges
calibrated against the real pipeline's output. False alarms are modelled
explicitly:
- **zone:** walking through slowly (high motion), misdetections such as
  shadows or posters (low confidence, no motion), and authorised staff (the
  hard, ambiguous case);
- **crowd:** people streaming past (high motion, low density) and duplicate
  boxes or reflections (low confidence).

Real events are people standing or loitering in a zone, and gatherings or
surges. Labels have 5% noise. The test set has 250 events from every simulated
zone, which stands for "the whole campus".

## Results

Deployed run: 5 simulated zones + the real-pipeline zone, 20 rounds, 3 local
epochs, stragglers on. Campus-wide synthetic test set:

| Setting | Accuracy | F1 | AUC |
|---|---|---|---|
| Local-only, mean of the 5 simulated zones | 0.782 | — | — |
| Local-only, lab_2 (smallest zone) | 0.742 | 0.722 | — |
| **FedAvg (deployed)** | **0.851** | **0.830** | **0.926** |
| FedProx (mu = 0.05) | 0.846 | 0.822 | 0.920 |
| Centralized (pooled, upper bound, not private) | 0.875 | 0.845 | 0.939 |

Synthetic zones only (`--no-real`): local-only 0.782, FedAvg 0.860,
FedProx 0.854, centralized 0.870. Communication: 121 KiB for the whole run.

**On held-out samples from the real vision pipeline** (96 samples from test clips
never used in training): FedAvg **0.885**, versus 0.729 for the synthetic-only
model. In the event evaluation, the classifier with the real zone federated in
keeps every true restricted-area entry (recall 1.00, was 0.75) and every crowd
(F1 1.00), with no false alerts. Full table: `evaluation/results/EVALUATION.md`.

How to read this for the report:
- Every simulated zone gains from federation, and the small lab zone gains most.
- FedAvg gets within about 2 points of the centralized upper bound without
  pooling any data.
- Federating in a zone with real pipeline data fixes the gap between the
  simulated feature distributions and what the cameras really produce. This is
  the non-IID benefit the project is about.
- Baselines use a 15% validation split with early stopping, so they are not
  handicapped by over-training.
- FedProx is close behind FedAvg here. Its proximal term pays off under
  stronger client drift: try `--mu 0.01`, `--local-epochs 8` or
  `--fraction-fit 0.6`. FedAvg is deployed and FedProx is the comparison, as the
  plan's risk section says.

## Files

| File | Purpose |
|---|---|
| `features.py` | feature order + fixed scaling (schema → vector) |
| `model.py` | NumPy MLP, local training with optional FedProx term, metrics, atomic save |
| `data.py` | zone profiles, synthetic non-IID data, per-zone JSONL files |
| `simulate.py` | in-process FL engine + full experiment (baselines, FedAvg, FedProx, logs) |
| `flower_server.py` / `flower_client.py` / `run_flower.py` | real Flower deployment over gRPC |
| `classifier.py` | `EventClassifier`: hot-reloading inference used by the backend |
| `label_vision_features.py` | unlabelled vision exports → weak-labelled per-zone files |
| `data/real/` | zone clients built from the real vision pipeline (ground-truth labels) |
| `plots.py` | convergence, local vs federated, and communication figures |
| `test_federated.py` | tests |

## Output artifacts (`artifacts/`)

- `global_model.json`: weights + `model_version`, strategy, round, clients, metrics. Written
  atomically, so the live system only ever loads a finished round.
- `rounds.json`: per-round log, rewritten after every round so the dashboard chart grows live: `round_number`, `strategy`, `participating_clients`,
  `metrics`, `comm_bytes`, `cumulative_comm_bytes`.
- `summary.json`: comparison table (local-only / FedAvg / FedProx / centralized) and per-zone accuracy.
- `training_status.json`: live progress, polled by the dashboard.
