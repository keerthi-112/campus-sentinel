"""
eval_rag.py — RAG and LLM-report evaluation,
plus the LLM timing comparison behind the model choice.

Uses the RAG track's own retrieve / generate code unchanged. Needs Ollama with
nomic-embed-text and the LLM(s) pulled, and the knowledge base ingested.

    python -m evaluation.eval_rag                                  # model from rag/config.py
    python -m evaluation.eval_rag --models llama3.2:3b llama3.1:8b # compare models (timing + quality)
    python -m evaluation.eval_rag --stub                           # self-test without Ollama (numbers meaningless)

Chat answers (16 questions from rag/eval_retrieval.py, each with the section that SHOULD be used)
  source_correct     the cited section is the expected one
  support            share of the answer's content words found in the retrieved excerpts
  faithful_auto      support >= 0.6 (no substantial content outside the excerpts)
Reports (20 incidents covering both event types, all zone tiers, low confidence, after hours)
  fields_complete    all required fields present
  sop_consistent     cited document fits the event type AND special rules are followed
                     (confidence < 0.6 → manual review/verify; crowd → crowd/emergency guidance)
  unsupported_rate   share of suggested-action sentences with < 50% support in the excerpts
Timing             seconds per report / answer, per model

Also writes results/rag_manual_rating.csv. Rate ~10 answers and reports by hand
(faithful: yes / partly / no) and quote both the automatic and manual numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RAG = REPO / "rag"
RES = REPO / "evaluation" / "results"
sys.path.insert(0, str(RAG))

STOP = set("""a an the and or of to in on at for from by with without is are was were be been being this that these
those it its as into than then there their they them which who whom whose what when where why how all any each
both more most other some such no nor not only own same so too very can will just should would could may might
must shall do does did done have has had having i you he she we our your his her them operator incident policy
section excerpt excerpts provided based""".split())

ZONE, CROWD = "restricted_area_entry", "crowd_formation"
ALLOWED_DOCS = {
    ZONE: {"Restricted Area Rules", "Security Procedures", "Escalation Rules", "Visitor Rules", "Emergency Response Steps"},
    CROWD: {"Crowd Management Guidance", "Emergency Response Steps", "Escalation Rules", "Security Procedures"},
}


def incidents() -> list[dict]:
    zones = [("Lab 2 restricted area", "CAM 03", 3), ("Server Room", "CAM 11", 3), ("Parking Lot B", "CAM 05", 1),
             ("Hostel Block C", "CAM 09", 2), ("Main Gate", "CAM 01", 2)]
    out, k = [], 0
    for et in (ZONE, CROWD):
        for (zone, cam, tier) in zones:
            for conf, hour in ((0.91, "14:20"), (0.48, "22:45")):
                k += 1
                sev = {3: "high", 2: "medium", 1: "low"}[tier] if et == ZONE else ("high" if tier >= 2 else "medium")
                out.append({"incident_ref": f"eval-{k:02d}", "event_type": et, "camera_name": cam, "zone_name": zone,
                            "timestamp": f"2026-11-16T{hour}:00Z", "confidence": conf,
                            "duration_seconds": 6.5 if et == ZONE else 12.0, "severity": sev, "status": "open"})
    return out


def words(text: str) -> list[str]:
    toks = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower())
    stem = lambda w: re.sub(r"(ing|ed|es|s)$", "", w) if len(w) > 5 else w
    return [stem(t) for t in toks if len(t) >= 4 and t not in STOP]


def support(text: str, context: str) -> float:
    ws = words(text)
    if not ws:
        return 1.0
    ctx = set(words(context))
    return round(sum(w in ctx for w in ws) / len(ws), 3)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?;])\s+|\n+", text or "") if len(s.strip()) > 3]


def doc_of(section: str, hits: list[dict]) -> str | None:
    for h in hits:
        if h["section"] == section:
            return h.get("title")
    return None


def stub_backend() -> None:
    """Self-test mode: deterministic fake embeddings + an extractive 'LLM'. Exercises
    every code path without Ollama; the numbers it produces are NOT results."""
    import hashlib

    import chromadb
    import numpy as np

    import generate
    import ingest
    import retrieve

    def emb(text: str) -> list[float]:
        v = np.zeros(256)
        for w in words(text):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
        return (v / (np.linalg.norm(v) or 1)).tolist()

    client = chromadb.EphemeralClient()
    col = client.get_or_create_collection("stub", metadata={"hnsw:space": "cosine"})
    records = ingest.load_documents() if hasattr(ingest, "load_documents") else None
    if records is None:  # fall back: parse markdown directly
        records = []
        for md in sorted((RAG / "documents").glob("*.md")):
            title, cur, buf = None, None, []
            for line in md.read_text(encoding="utf-8").splitlines() + ["## END"]:
                if line.startswith("# "):
                    title = line[2:].strip()
                elif line.startswith("## "):
                    if cur:
                        records.append({"title": title, "source": md.name, "section": cur, "text": "\n".join(buf).strip()})
                    cur, buf = line[3:].strip(), []
                else:
                    buf.append(line)
    col.add(ids=[f"s{i}" for i in range(len(records))], documents=[r["text"] for r in records],
            embeddings=[emb(r["text"] + " " + r["section"]) for r in records],
            metadatas=[{"title": r["title"], "source": r["source"], "section": r["section"]} for r in records])

    def fake_retrieve(query, k=retrieve.RETRIEVAL_TOP_K):
        res = col.query(query_embeddings=[emb(query)], n_results=k)
        return [{"text": t, "distance": d, **m} for t, m, d in
                zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]

    def fake_ask(system_prompt, user_prompt):
        first = re.search(r"\[(.+?)\]\n(.+?)(?:\n\n\[|\Z)", user_prompt.split("excerpts:", 1)[-1], re.S)
        label, text = (first.group(1), first.group(2)) if first else ("", "")
        sents = sentences(text)
        if "Question:" in user_prompt:
            return {"answer": " ".join(sents[:2]), "source_section": label}
        return {"report_text": "Incident recorded. " + (sents[0] if sents else ""),
                "suggested_action": " ".join(sents[1:3]) or "Notify the operator.", "source_section": label}

    retrieve.retrieve = fake_retrieve
    generate.retrieve = fake_retrieve
    generate._ask = fake_ask


def run_model(model: str, stub: bool) -> dict:
    import generate
    import retrieve
    from eval_retrieval import TEST_SET

    generate.LLM_MODEL = model
    k = retrieve.RETRIEVAL_TOP_K if hasattr(retrieve, "RETRIEVAL_TOP_K") else 3
    chat_rows, report_rows, chat_t, report_t, errors = [], [], [], [], []

    for q, expected in TEST_SET:
        hits = retrieve.retrieve(q, k=k)
        ctx = "\n".join(h["text"] for h in hits)
        t = time.perf_counter()
        try:
            ans = generate.generate_chat_answer(q)
        except Exception as exc:  # timeouts, malformed JSON
            errors.append(f"chat: {q[:40]}: {exc}")
            continue
        chat_t.append(time.perf_counter() - t)
        sup = support(ans["answer"], ctx)
        chat_rows.append({"question": q, "expected": expected, "cited": ans["source_section"],
                          "source_correct": expected in ans["source_section"],
                          "expected_retrieved": any(expected in h["section"] for h in hits),
                          "support": sup, "faithful_auto": sup >= 0.6, "answer": ans["answer"]})

    for inc in incidents():
        t = time.perf_counter()
        try:
            rep = generate.generate_report(inc)
        except Exception as exc:
            errors.append(f"report: {inc['incident_ref']}: {exc}")
            continue
        report_t.append(time.perf_counter() - t)
        query = (f"{inc['event_type']} in {inc['zone_name']}, {inc['severity']} severity, detection confidence "
                 f"{inc['confidence']}, duration {inc['duration_seconds']}s, at {inc['timestamp']}")
        hits = retrieve.retrieve(query, k=k)
        ctx = "\n".join(h["text"] for h in hits)
        doc = doc_of(rep["source_section"], hits)
        rules = []
        if inc["confidence"] < 0.6:
            rules.append(bool(re.search(r"manual|review|verif|confirm", rep["suggested_action"] + rep["report_text"], re.I)))
        sop_ok = (doc in ALLOWED_DOCS[inc["event_type"]]) and all(rules)
        acts = sentences(rep["suggested_action"])
        unsupported = [s for s in acts if support(s, ctx) < 0.5]
        report_rows.append({
            "incident_ref": inc["incident_ref"], "event_type": inc["event_type"], "zone": inc["zone_name"],
            "confidence": inc["confidence"], "cited": rep["source_section"], "cited_document": doc,
            "fields_complete": all(rep.get(f) for f in ("incident_ref", "report_text", "suggested_action",
                                                        "source_section", "created_time")),
            "sop_consistent": sop_ok, "low_conf_rule_ok": None if not rules else rules[0],
            "action_sentences": len(acts), "unsupported_sentences": len(unsupported),
            "report_support": support(rep["report_text"], ctx + " " + json.dumps(inc)),
            "suggested_action": rep["suggested_action"], "report_text": rep["report_text"]})

    def stat(xs):
        return {"mean_s": round(statistics.mean(xs), 2), "p50_s": round(statistics.median(xs), 2),
                "p95_s": round(sorted(xs)[max(0, int(len(xs) * 0.95) - 1)], 2), "n": len(xs)} if xs else None

    n_c, n_r = max(len(chat_rows), 1), max(len(report_rows), 1)
    total_sent = sum(r["action_sentences"] for r in report_rows) or 1
    low = [r for r in report_rows if r["low_conf_rule_ok"] is not None]
    return {
        "model": model, "stub": stub, "errors": errors,
        "chat": {"n": len(chat_rows),
                 "source_correct": round(sum(r["source_correct"] for r in chat_rows) / n_c, 3),
                 "expected_section_retrieved": round(sum(r["expected_retrieved"] for r in chat_rows) / n_c, 3),
                 "support_mean": round(statistics.mean([r["support"] for r in chat_rows]), 3) if chat_rows else None,
                 "faithful_auto": round(sum(r["faithful_auto"] for r in chat_rows) / n_c, 3)},
        "reports": {"n": len(report_rows),
                    "fields_complete": round(sum(r["fields_complete"] for r in report_rows) / n_r, 3),
                    "sop_consistent": round(sum(r["sop_consistent"] for r in report_rows) / n_r, 3),
                    "low_confidence_rule_followed": round(sum(r["low_conf_rule_ok"] for r in low) / max(len(low), 1), 3),
                    "unsupported_recommendation_rate": round(sum(r["unsupported_sentences"] for r in report_rows) / total_sent, 3),
                    "reports_with_unsupported_action": round(sum(r["unsupported_sentences"] > 0 for r in report_rows) / n_r, 3)},
        "timing": {"chat": stat(chat_t), "report": stat(report_t)},
        "chat_rows": chat_rows, "report_rows": report_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG + LLM report evaluation")
    ap.add_argument("--models", nargs="+", default=None, help="Ollama models to compare (default: rag/config.py LLM_MODEL)")
    ap.add_argument("--stub", action="store_true", help="self-test without Ollama; numbers are meaningless")
    a = ap.parse_args()

    import config as rag_config
    if a.stub:
        stub_backend()
    models = a.models or [rag_config.LLM_MODEL]
    RES.mkdir(parents=True, exist_ok=True)
    results = []
    for m in models:
        print(f"\n== {m}{' (STUB)' if a.stub else ''}: 16 chat questions + 20 incident reports ...")
        r = run_model(m, a.stub)
        results.append(r)
        c, p, t = r["chat"], r["reports"], r["timing"]
        print(f"  chat:    source correct {c['source_correct']:.0%}  (expected section retrieved {c['expected_section_retrieved']:.0%})  "
              f"support {c['support_mean']}  faithful(auto) {c['faithful_auto']:.0%}")
        print(f"  reports: fields {p['fields_complete']:.0%}  SOP-consistent {p['sop_consistent']:.0%}  "
              f"low-conf rule {p['low_confidence_rule_followed']:.0%}  unsupported actions {p['unsupported_recommendation_rate']:.0%}")
        if t["report"]:
            print(f"  timing:  report {t['report']['mean_s']} s (p95 {t['report']['p95_s']}), chat {t['chat']['mean_s']} s")
        if r["errors"]:
            print(f"  {len(r['errors'])} errors, first: {r['errors'][0]}")

    name = "rag_stub.json" if a.stub else "rag.json"
    (RES / name).write_text(json.dumps({"models": results}, indent=2), encoding="utf-8")
    if not a.stub:
        with open(RES / "rag_manual_rating.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["model", "kind", "id", "question_or_incident", "cited", "auto_support", "text", "manual_faithful (yes/partly/no)"])
            for r in results:
                for row in r["chat_rows"]:
                    w.writerow([r["model"], "chat", "", row["question"], row["cited"], row["support"], row["answer"], ""])
                for row in r["report_rows"]:
                    w.writerow([r["model"], "report", row["incident_ref"], f"{row['event_type']} @ {row['zone']} conf {row['confidence']}",
                                row["cited"], row["report_support"], row["report_text"] + " | ACTION: " + row["suggested_action"], ""])
    print(f"\n-> evaluation/results/{name}" + ("" if a.stub else " and rag_manual_rating.csv"))


if __name__ == "__main__":
    main()
