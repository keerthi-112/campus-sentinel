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
trains on 7 compact numbers per event, so one model transfer is **580 bytes**, and
raw video or raw events never leave a zone.

## Setup

```bash
# from the repo root
pip install -r federated/requirements.txt
```

## Usage (run everything from the repo root)

```bash
python -m federated.data --clients 5          # (re)generate 5 non-IID zone datasets + global test set
python -m federated.simulate                  # FedAvg + FedProx, 20 rounds, baselines, deploys FedAvg
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

Training on real vision exports instead of synthetic data:

```bash
python -m vision.run_demo --source clip.mp4 --type file --export-features data/features.jsonl --no-window
python -m federated.label_vision_features data/features.jsonl [--labels labels.csv]
python -m federated.simulate --clients 3
```

The dashboard can also start a training run: **Federated → Start training**.

## The model

| | |
|---|---|
| Input (7) | `person_count, density, occupancy, motion_score, zone_dwell_seconds, max_confidence, is_crowd_event` |
| Model | MLP 7 → 16 (ReLU) → 1 (sigmoid) = 145 parameters, NumPy only |
| Output | P(real incident). ≥ 0.5 → escalate (create incident + RAG report), else suppress |
| Scaling | fixed, agreed constants (`features.py`), so zones don't share statistics |
| Local training | Adam, BCE. FedProx adds `mu/2 · ‖w − w_global‖²` |
| Aggregation | sample-weighted average (FedAvg). FedProx uses the same server step |

## Simulated zones (non-IID by design)

| Zone | Camera | Local samples | Real-incident rate | Crowd share | Note |
|---|---|---|---|---|---|
| gate_1 | CAM 01 Main Gate | 1200 | 0.27 | 0.58 | busy, crowd-heavy |
| lab_2 | CAM 03 Lab 2 | 180 | 0.54 | 0.04 | small, almost no crowds |
| parking | CAM 05 Parking B | 420 | 0.36 | 0.15 | dark → lower YOLO confidence |
| library | CAM 07 Atrium | 320 | 0.29 | 0.36 | |
| hostel | CAM 09 Hostel C | 260 | 0.45 | 0.21 | |

False alarms are modelled explicitly: pass-throughs (dwell just over the timer,
high motion), detection flicker (low confidence), authorised staff (the hard
case), and normal congestion that looks like a crowd. 5% label noise. The test
set has 250 events from every zone, which stands for "the whole campus".

## Results (default run: 5 zones, 20 rounds, 3 local epochs, stragglers on)

| Setting | Accuracy | F1 | AUC |
|---|---|---|---|
| Local-only, mean of 5 zones | 0.805 | — | — |
| Local-only, lab_2 (smallest zone) | 0.685 | 0.686 | — |
| **FedAvg (deployed)** | **0.876** | **0.835** | **0.955** |
| FedProx (mu = 0.05) | 0.849 | 0.783 | 0.947 |
| Centralized (pooled, upper bound, not private) | 0.884 | 0.849 | 0.956 |

Total communication for a 20-round, 5-zone run: about 113 KiB.

How to read this for the report: every zone gains from federation, and the small
lab zone gains most (0.69 → 0.88), because it almost never sees crowds locally.
FedAvg gets within 1 point of the centralized upper bound without pooling any data.
Baselines (local-only, centralized) use a 15% validation split with early stopping,
so they are not handicapped by over-training. On this data,
FedProx converges more slowly. The proximal term is a brake that pays off under
stronger client drift. Try `--mu 0.01`, `--local-epochs 8` or `--fraction-fit 0.6`
to explore this. That's why FedAvg is deployed and FedProx is the comparison,
as the dossier's risk plan says.

## Files

| File | Purpose |
|---|---|
| `features.py` | feature order + fixed scaling (schema → vector) |
| `model.py` | NumPy MLP, local training with optional FedProx term, metrics, atomic save |
| `data.py` | zone profiles, synthetic non-IID data, per-zone JSONL files |
| `simulate.py` | in-process FL engine + full experiment (baselines, FedAvg, FedProx, logs) |
| `flower_server.py` / `flower_client.py` / `run_flower.py` | real Flower deployment over gRPC |
| `classifier.py` | `EventClassifier`: hot-reloading inference used by the backend |
| `label_vision_features.py` | real vision exports → labelled per-zone training files |
| `plots.py` | convergence, local vs federated, and communication figures |
| `test_federated.py` | tests |

## Output artifacts (`artifacts/`)

- `global_model.json`: weights + `model_version`, strategy, round, clients, metrics. Written
  atomically, so the live system only ever loads a finished round.
- `rounds.json`: per-round log, rewritten after every round so the dashboard chart grows live: `round_number`, `strategy`, `participating_clients`,
  `metrics`, `comm_bytes`, `cumulative_comm_bytes`.
- `summary.json`: comparison table (local-only / FedAvg / FedProx / centralized) and per-zone accuracy.
- `training_status.json`: live progress, polled by the dashboard.
