# Campus Sentinel: evaluation results (template §14)

_Generated 2026-11-17 19:16 by `python -m evaluation.make_report`._

## 1. Person detection: precision, recall, mAP, frame rate, response time

Dataset: `coco128.yaml` (person class only), device `cpu`, image size 640. Speed measured on 960x540 camera frames.

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | ms / frame | FPS |
|---|---|---|---|---|---|---|
| yolo26n.pt **(selected)** | 0.808 | 0.709 | 0.795 | 0.546 | 60.8 | 16.5 |
| yolo11n.pt | 0.755 | 0.717 | 0.786 | 0.541 | 49.7 | 20.1 |

Selection rule: highest person mAP50-95; ties broken by speed (template §21). COCO128 images come from COCO train2017, so absolute numbers are optimistic; the comparison between models is fair. For camera-specific numbers, pass your own labelled frames with `--data`.

## 2. Restricted-area events: precision, recall, false alert rate, detection-to-alert time

15 clips, 3.3 min (synthetic scenario clips with exact ground truth), model `yolo26n.pt`, FL model `1594638c-r20`.

| | Precision | Recall | F1 | False alerts / min | Detection → alert (mean, p95) | Lag after dwell threshold |
|---|---|---|---|---|---|---|
| Vision rules only | 1.000 | 1.000 | 1.000 | 0.0 | 3.97 s, 4.0 s | within one frame |
| + FL classifier | 1.000 | 1.000 | 1.000 | 0.0 | 3.97 s, 4.0 s | within one frame |

Detection → alert is measured from the moment the person enters the zone, so it includes the deliberate dwell threshold (4 s) that stops single-frame false alerts.

## 3. Crowd event: accuracy and F1 on the test videos

| | Accuracy | Precision | Recall | F1 | Clips |
|---|---|---|---|---|---|
| Vision rules only | 1.000 | 1.000 | 1.000 | 1.000 | 15 |
| + FL classifier | 1.000 | 1.000 | 1.000 | 1.000 | 15 |

Pipeline speed on these clips: YOLO 52.7 ms/frame, full pipeline 52.8 ms/frame (18.9 FPS) on this machine's CPU.

Same clips with `yolo11n.pt`: restricted P/R 1.0/1.0, crowd F1 1.0, 18.2 FPS.

## 4. Federated learning: local versus federated accuracy, convergence, communication cost

Deployed run: 6 zone clients (5 simulated zones + 1 real vision-pipeline zone), 20 rounds. Scored on the campus-wide synthetic test set.

| Setting | Accuracy | F1 | AUC | Rounds to 98% | Communication |
|---|---|---|---|---|---|
| Local only (mean of the simulated zones) | 78.2% | — | — | — | 0 |
| **Federated FEDAVG** | **85.1%** | 0.830 | 0.926 | 5 | 120.9 KiB |
| **Federated FEDPROX** | **84.6%** | 0.822 | 0.920 | 5 | 120.9 KiB |
| Centralized (upper bound, not private) | 87.5% | 0.845 | 0.939 | — | all raw data |

Only model parameters (516 bytes per transfer) left each zone; raw events and video never did.

Synthetic zones only (5 clients): local-only mean 78.2%, FedAvg 86.0%, FedProx 85.4%, centralized 87.0%.

The smallest zone (lab_2, 180 events) reaches 74.2% alone on the campus-wide test set; through federation it gets the global model's accuracy.

**With features from the real vision pipeline** (`evaluation/fl_real_features.py`): 282 labelled samples from training clips became an extra zone client; 96 samples from separate test clips were held out.

- Accuracy on held-out real-pipeline samples: synthetic-only model 72.9% → with the real zone federated in 88.5%.
- Test clips, vision rules only: restricted recall 1.0, precision 1.0; crowd F1 1.0
- Test clips, FL filter, synthetic zones only: restricted recall 0.75, precision 1.0; crowd F1 1.0
- Test clips, FL filter, with real-pipeline zone: restricted recall 1.0, precision 1.0; crowd F1 1.0

## 5. RAG: retrieval quality, source correctness, answer faithfulness

Retrieval quality from the RAG track's own `rag/eval_retrieval.py`: recall@3 = 100% on 16 labelled queries.

_Needs Ollama: run `python -m evaluation.eval_rag --models llama3.2:3b llama3.1:8b` on a laptop with both models pulled. That also gives the timing table that justifies the model choice (§6, §21)._

## 6. LLM report: required fields, consistency with the SOP, unsupported recommendation rate

_Produced by the same `evaluation.eval_rag` run._

## 7. System: end-to-end latency and basic resource usage

Measured over 60 s on a 2-core, 8.4 GB machine.

| Measure | Value |
|---|---|
| Backend CPU | 87.1% mean, 98.7% max |
| Backend memory | 883 MB mean, 892.1 MB max |
| Frame processing | 64.8 ms (14.6 FPS) |
| Alert decision (candidate → FL → stored) | 8.749 ms mean, p95 11.947 ms |
| Zone entry → alert | 4.07 s (includes the dwell threshold) |
