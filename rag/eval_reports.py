"""Measure generated-report quality against the evaluation table's LLM row:
required fields present, and whether the cited source is real and was actually
retrieved (an invented citation is the clearest automatable faithfulness failure).

Needs the LLM, unlike eval_retrieval.py.

    python eval_reports.py
"""
from config import RETRIEVAL_TOP_K
from generate import generate_report
from retrieve import get_collection, retrieve

REQUIRED_FIELDS = ["incident_ref", "report_text", "suggested_action", "source_section", "created_time"]

TEST_INCIDENTS = [
    {
        "incident_ref": "eval-001",
        "event_type": "restricted_area_entry",
        "camera_name": "CAM 03",
        "zone_name": "Lab 2 restricted area",
        "timestamp": "2026-09-20T02:14:00Z",
        "confidence": 0.94,
        "duration_seconds": 12.0,
        "severity": "high",
        "status": "open",
    },
    {
        "incident_ref": "eval-002",
        "event_type": "restricted_area_entry",
        "camera_name": "CAM 07",
        "zone_name": "Storage room B",
        "timestamp": "2026-09-20T11:05:00Z",
        "confidence": 0.71,
        "duration_seconds": 4.0,
        "severity": "low",
        "status": "open",
    },
    {
        "incident_ref": "eval-003",
        "event_type": "crowd_formation",
        "camera_name": "CAM 11",
        "zone_name": "Main lobby",
        "timestamp": "2026-09-20T17:40:00Z",
        "confidence": 0.88,
        "duration_seconds": 45.0,
        "severity": "medium",
        "status": "open",
    },
    {
        "incident_ref": "eval-004",
        "event_type": "restricted_area_entry",
        "camera_name": "CAM 02",
        "zone_name": "Server room",
        "timestamp": "2026-09-20T22:50:00Z",
        "confidence": 0.55,
        "duration_seconds": 3.5,
        "severity": "medium",
        "status": "open",
    },
]


def known_sections() -> set[str]:
    metadatas = get_collection().get(include=["metadatas"])["metadatas"]
    return {m["section"] for m in metadatas}


def main():
    valid_sections = known_sections()

    complete = 0
    real_citation = 0
    retrieved_citation = 0
    failures = []

    for incident in TEST_INCIDENTS:
        report = generate_report(incident)

        missing = [f for f in REQUIRED_FIELDS if not report.get(f)]
        if missing:
            failures.append(f"{incident['incident_ref']}: missing/empty {missing}")
        else:
            complete += 1

        cited = report.get("source_section", "")
        if cited in valid_sections:
            real_citation += 1
        else:
            failures.append(f"{incident['incident_ref']}: cited a section not in the KB -> {cited!r}")

        query = f"{incident['event_type']} in {incident['zone_name']} ({incident['severity']} severity)"
        retrieved = {hit["section"] for hit in retrieve(query, k=RETRIEVAL_TOP_K)}
        if cited in retrieved:
            retrieved_citation += 1

    total = len(TEST_INCIDENTS)
    print(f"Incidents:               {total}")
    print(f"All fields present:      {complete}/{total} ({complete / total:.1%})")
    print(f"Citation is a real section: {real_citation}/{total} ({real_citation / total:.1%})")
    print(f"Citation was retrieved:  {retrieved_citation}/{total} ({retrieved_citation / total:.1%})")

    if failures:
        print(f"\nFailures ({len(failures)}):")
        for failure in failures:
            print(f"  - {failure}")


if __name__ == "__main__":
    main()
