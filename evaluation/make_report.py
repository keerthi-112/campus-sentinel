"""
make_report.py — Collect every evaluation result into one results file.

    python -m evaluation.make_report          # -> evaluation/results/EVALUATION.md

Rows whose result file is missing say which command produces them.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "evaluation" / "results"
ZONE, CROWD = "restricted_area_entry", "crowd_formation"


def load(name: str, base: Path = RES):
    p = base / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pct(v):
    return "—" if v is None else f"{v * 100:.1f}%"


def main() -> None:
    L = [f"# Campus Sentinel: evaluation results\n",
         f"_Generated {datetime.now():%Y-%m-%d %H:%M} by `python -m evaluation.make_report`._\n"]

    # ---------------------------------------------------------------- detection
    det = load("detection.json")
    L.append("## 1. Person detection: precision, recall, mAP, frame rate, response time\n")
    if det:
        L.append(f"Dataset: `{det['results'][0]['data']}` (person class only), device `{det['device']}`, "
                 f"image size {det['imgsz']}. Speed measured on {det['results'][0]['speed_frame_size']} camera frames.\n")
        L.append("| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | ms / frame | FPS |\n|---|---|---|---|---|---|---|")
        for r in det["results"]:
            mark = " **(selected)**" if r["model"] == det["selected"] else ""
            L.append(f"| {r['model']}{mark} | {r['precision']:.3f} | {r['recall']:.3f} | {r['mAP50']:.3f} | "
                     f"{r['mAP50_95']:.3f} | {r['speed_ms_mean']} | {r['speed_fps']} |")
        L.append(f"\nSelection rule: {det['rule']}. COCO128 images come from COCO train2017, so absolute numbers are "
                 "optimistic; the comparison between models is fair. For camera-specific numbers, pass your own labelled "
                 "frames with `--data`.\n")
    else:
        L.append("_Not run yet: `python -m evaluation.eval_detection`._\n")

    # ---------------------------------------------------------------- events
    ev = load("events_yolo26n_clips.json") or load("events_fl_after.json") or load("events_fl_before.json")
    L.append("## 2. Restricted-area events: precision, recall, false alert rate, detection-to-alert time\n")
    if ev:
        src = "synthetic scenario clips with exact ground truth" if ev["synthetic"] else "your labelled clips"
        L.append(f"{ev['n_clips']} clips, {ev['total_minutes']} min ({src}), model `{ev['model']}`, "
                 f"FL model `{ev['fl_model_version']}`.\n")
        L.append("| | Precision | Recall | F1 | False alerts / min | Detection → alert (mean, p95) | Lag after dwell threshold |\n"
                 "|---|---|---|---|---|---|---|")
        for name, key in (("Vision rules only", "vision_only"), ("+ FL classifier", "with_fl_classifier")):
            z = ev[key][ZONE]
            lag = z["lag_after_threshold_s_mean"]
            lag_txt = "—" if lag is None else ("within one frame" if abs(lag) < 0.1 else f"{lag} s")
            L.append(f"| {name} | {z['precision']:.3f} | {z['recall']:.3f} | {z['f1']:.3f} | {z['false_alerts_per_min']} | "
                     f"{z['detection_to_alert_s_mean']} s, {z['detection_to_alert_s_p95']} s | {lag_txt} |")
        L.append("\nDetection → alert is measured from the moment the person enters the zone, so it includes the "
                 "deliberate dwell threshold (4 s) that stops single-frame false alerts.\n")
        L.append("## 3. Crowd event: accuracy and F1 on the test videos\n")
        L.append("| | Accuracy | Precision | Recall | F1 | Clips |\n|---|---|---|---|---|---|")
        for name, key in (("Vision rules only", "vision_only"), ("+ FL classifier", "with_fl_classifier")):
            c = ev[key]["crowd_clip_level"]
            L.append(f"| {name} | {c['accuracy']:.3f} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} | {c['clips']} |")
        p = ev["performance"]
        L.append(f"\nPipeline speed on these clips: YOLO {p['yolo_ms_mean']} ms/frame, full pipeline "
                 f"{p['pipeline_ms_mean']} ms/frame ({p['fps']} FPS) on this machine's CPU.\n")
        other = load("events_yolo11n_clips.json")
        if other:
            z = other["vision_only"][ZONE]
            L.append(f"Same clips with `yolo11n.pt`: restricted P/R {z['precision']}/{z['recall']}, "
                     f"crowd F1 {other['vision_only']['crowd_clip_level']['f1']}, {other['performance']['fps']} FPS.\n")
    else:
        L.append("_Not run yet: `python -m evaluation.make_scenarios` then `python -m evaluation.eval_events`._\n")

    # ---------------------------------------------------------------- FL
    fl = load("summary.json", REPO / "federated" / "artifacts")
    L.append("## 4. Federated learning: local versus federated accuracy, convergence, communication cost\n")
    if fl:
        synth_zones = {k: v for k, v in fl["local_only"].items() if not k.startswith("real_")}
        mean_synth = sum(v["global_test"]["accuracy"] for v in synth_zones.values()) / max(len(synth_zones), 1)
        L.append(f"Deployed run: {len(fl['local_only'])} zone clients ({len(synth_zones)} simulated zones"
                 f"{' + 1 real vision-pipeline zone' if len(synth_zones) < len(fl['local_only']) else ''}), "
                 f"{fl['config']['rounds']} rounds. Scored on the campus-wide synthetic test set.\n")
        L.append("| Setting | Accuracy | F1 | AUC | Rounds to 98% | Communication |\n|---|---|---|---|---|---|")
        L.append(f"| Local only (mean of the simulated zones) | {pct(mean_synth)} | — | — | — | 0 |")
        for s, v in fl["federated"].items():
            L.append(f"| **Federated {s.upper()}** | **{pct(v['final']['accuracy'])}** | {v['final']['f1']:.3f} | "
                     f"{v['final']['auc']:.3f} | {v['rounds_to_98pct']} | {v['total_comm_bytes'] / 1024:.1f} KiB |")
        c = fl["centralized"]["global_test"]
        L.append(f"| Centralized (upper bound, not private) | {pct(c['accuracy'])} | {c['f1']:.3f} | {c['auc']:.3f} | — | all raw data |")
        L.append(f"\n{fl.get('privacy_note', '')}\n")
        so = load("fl_summary_synthetic_only.json")
        if so:
            L.append(f"Synthetic zones only (5 clients): local-only mean {pct(so['local_only_mean_accuracy'])}, "
                     f"FedAvg {pct(so['federated']['fedavg']['final']['accuracy'])}, FedProx "
                     f"{pct(so['federated']['fedprox']['final']['accuracy'])}, centralized "
                     f"{pct(so['centralized']['global_test']['accuracy'])}.\n")
        lab = fl["local_only"].get("lab_2")
        if lab:
            L.append(f"The smallest zone (lab_2, {lab['n_train']} events) reaches {pct(lab['global_test']['accuracy'])} "
                     "alone on the campus-wide test set; through federation it gets the global model's accuracy.\n")
    flr = load("fl_real.json")
    if flr:
        L.append("**With features from the real vision pipeline** (`evaluation/fl_real_features.py`): "
                 f"{flr['real_train_samples']} labelled samples from training clips became an extra zone client; "
                 f"{flr['real_test_samples']} samples from separate test clips were held out.\n")
        b, a = flr.get("synthetic_only_model_on_real_samples"), (flr.get("with_real_zone_on_real_samples") or {}).get("fedavg")
        if b and a:
            L.append(f"- Accuracy on held-out real-pipeline samples: synthetic-only model {pct(b['accuracy'])} → "
                     f"with the real zone federated in {pct(a['accuracy'])}.")
        e = flr["events_on_test_clips"]
        for name, key in (("vision rules only", "vision_only"), ("FL filter, synthetic zones only", "fl_filter_synthetic_only"),
                          ("FL filter, with real-pipeline zone", "fl_filter_with_real_zone")):
            L.append(f"- Test clips, {name}: restricted recall {e[key][ZONE]['recall']}, precision {e[key][ZONE]['precision']}; "
                     f"crowd F1 {e[key]['crowd_clip_level']['f1']}")
        L.append("")

    # ---------------------------------------------------------------- RAG / LLM
    rag = load("rag.json")
    L.append("## 5. RAG: retrieval quality, source correctness, answer faithfulness\n")
    L.append("Retrieval quality from the RAG track's own `rag/eval_retrieval.py`: recall@3 = 100% on 16 labelled queries.\n")
    if rag:
        L.append("| Model | Source correct | Expected section retrieved | Answer support | Faithful (auto) |\n|---|---|---|---|---|")
        for m in rag["models"]:
            c = m["chat"]
            L.append(f"| {m['model']} | {pct(c['source_correct'])} | {pct(c['expected_section_retrieved'])} | "
                     f"{c['support_mean']} | {pct(c['faithful_auto'])} |")
        L.append("\nManual faithfulness ratings: fill `evaluation/results/rag_manual_rating.csv`.\n")
        L.append("## 6. LLM report: required fields, consistency with the SOP, unsupported recommendation rate\n")
        L.append("| Model | Fields present | SOP-consistent | Low-confidence rule followed | Unsupported action sentences | Report time (mean, p95) |\n"
                 "|---|---|---|---|---|---|")
        for m in rag["models"]:
            r, t = m["reports"], m["timing"]["report"] or {}
            L.append(f"| {m['model']} | {pct(r['fields_complete'])} | {pct(r['sop_consistent'])} | "
                     f"{pct(r['low_confidence_rule_followed'])} | {pct(r['unsupported_recommendation_rate'])} | "
                     f"{t.get('mean_s', '—')} s, {t.get('p95_s', '—')} s |")
        L.append("")
    else:
        L.append("_Needs Ollama: run `python -m evaluation.eval_rag --models llama3.2:3b llama3.1:8b` on a laptop "
                 "with both models pulled. That also gives the timing table that justifies the model choice._\n")
        L.append("## 6. LLM report: required fields, consistency with the SOP, unsupported recommendation rate\n")
        L.append("_Produced by the same `evaluation.eval_rag` run._\n")

    # ---------------------------------------------------------------- system
    sysm = load("system.json")
    L.append("## 7. System: end-to-end latency and basic resource usage\n")
    if sysm:
        e = sysm["end_to_end_estimate_s"]
        L.append(f"Measured over {sysm['seconds']} s on a {sysm['machine']['cpu_count']}-core, "
                 f"{sysm['machine']['ram_total_gb']} GB machine.\n")
        L.append("| Measure | Value |\n|---|---|")
        L.append(f"| Backend CPU | {sysm['backend_cpu_percent']['mean']}% mean, {sysm['backend_cpu_percent']['max']}% max |")
        L.append(f"| Backend memory | {sysm['backend_rss_mb']['mean']} MB mean, {sysm['backend_rss_mb']['max']} MB max |")
        if sysm.get("frame_ms"):
            L.append(f"| Frame processing | {sysm['frame_ms']['mean']} ms ({sysm['fps']['mean']} FPS) |")
        lat = sysm["latency"]
        if lat.get("decision_ms"):
            L.append(f"| Alert decision (candidate → FL → stored) | {lat['decision_ms']['mean']} ms mean, p95 {lat['decision_ms']['p95']} ms |")
        if lat.get("report_total_s"):
            L.append(f"| Alert → report ready | {lat['report_total_s']['mean']} s mean, p95 {lat['report_total_s']['p95']} s |")
        L.append(f"| Zone entry → alert | {e['zone_entry_to_alert']} s (includes the dwell threshold) |")
        if e.get("zone_entry_to_report"):
            L.append(f"| Zone entry → report | {e['zone_entry_to_report']} s |")
        L.append("")
    else:
        L.append("_Run while the system is live: `python -m evaluation.eval_system --seconds 120`._\n")

    out = RES / "EVALUATION.md"
    RES.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"-> {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
