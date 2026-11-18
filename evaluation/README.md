# evaluation/

Scripts that measure every part of Campus Sentinel and collect the results in
`results/EVALUATION.md`.
Nothing here changes the `vision/`, `rag/` or `federated/` code. The scripts
import and run them as they are.

```bash
# from the repo root, venv active
python -m evaluation.make_scenarios                      # 15 labelled test clips (exact ground truth)
python -m evaluation.eval_detection                      # person detection, YOLO26n vs YOLO11n
python -m evaluation.eval_events                         # restricted-area + crowd events
python -m evaluation.make_scenarios --out evaluation/clips_train --variants 3
python -m evaluation.fl_real_features                    # FL with real vision-pipeline features
python -m evaluation.eval_rag --models llama3.2:3b llama3.1:8b   # RAG + LLM report (needs Ollama)
python -m evaluation.eval_system --seconds 120           # system latency + resources (while the app runs)
python -m evaluation.make_report                         # -> results/EVALUATION.md
```

| Part | Script | Measures | Output |
|---|---|---|---|
| Person detection | `eval_detection.py` | precision, recall, mAP@0.5, mAP@0.5:0.95 (person class), ms/frame, FPS for YOLO26n and YOLO11n; selects the better one | `results/detection.json` |
| Restricted-area events | `eval_events.py` | precision, recall, F1, false alerts / min, detection → alert time, lag after the dwell threshold; with and without the FL classifier | `results/events_<model>_<clips>.json` |
| Crowd event | `eval_events.py` | per-clip accuracy, precision, recall, F1 | same file |
| Federated learning | `federated/simulate.py`, `fl_real_features.py` | local vs federated vs centralized, convergence, communication; accuracy on real-pipeline features | `federated/artifacts/summary.json`, `results/fl_real.json` |
| RAG | `eval_rag.py` | source correctness, expected section retrieved, answer support, automatic faithfulness, plus a CSV for manual rating | `results/rag.json`, `results/rag_manual_rating.csv` |
| LLM report | `eval_rag.py` | required fields, SOP consistency (right document + low-confidence → manual review), unsupported-recommendation rate, seconds per report per model | `results/rag.json` |
| System | `eval_system.py` + Reports tab | backend CPU / RAM, frame time, FPS, alert decision latency, alert → report latency, zone entry → report | `results/system.json` |

## Test clips

`make_scenarios.py` moves real people (cut out of the Ultralytics sample
photos, so YOLO genuinely has to detect them) over a static camera background
along scripted paths. That gives **exact** ground truth for 15 situations:
entering and staying, walking past, crossing through quickly, stepping in
briefly, a second person passing, gatherings of 6–8 people, and small groups of
3–4. They are reproducible, but easier than real footage.

**Also evaluate on real clips.** Record 3–4 clips of 20–60 s with a phone or
webcam, then label each one:

```bash
python -m evaluation.mark_events --video my_clips/door.mp4 --camera "CAM 03" --zone-name "Lab 2 restricted area"
python -m evaluation.eval_events --clips my_clips
```

Report both sets: synthetic clips as a controlled test, real clips as the
realistic one.

## How the event evaluation works

The vision modules use `time.time()` for dwell timers. `eval_events.py` gives
them the clip's own clock (frame index / fps), so results are identical on any
laptop. An alert matches a ground-truth event if it is of the same type and
falls between the event's start and 1.5 s after its end. Extra alerts for an
already-matched event count as duplicates; all other alerts are false positives.
Zone entries shorter than the dwell threshold are negatives: an alert on them is
a false alarm.

## Notes on the results

- **YOLO26n** loads and runs with the vision track's `PersonDetector`
  unchanged (`--model yolo26n.pt`). The backend now uses it by default
  (`SENTINEL_YOLO_MODEL`).
- **COCO128** images are from COCO train2017, which the models were trained
  on. Use them to compare models, not as absolute accuracy. Label ~50 of your
  own frames for absolute numbers (`--data`).
- **RAG faithfulness** has an automatic proxy (word-level support in the
  retrieved excerpts) and a manual-rating sheet. Quote both.
- **LLM model choice:** `eval_rag.py --models llama3.2:3b llama3.1:8b` gives
  seconds per report for each. If 8B is too slow on the demo laptop, that table
  shows why 3B is the default on typical laptops.
- `eval_rag.py --stub` is a self-test without Ollama. Its numbers are not results.
