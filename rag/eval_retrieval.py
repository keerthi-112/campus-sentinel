"""Measure retrieval quality against a labelled query set.

Reports top-1 accuracy and recall@k — the "retrieval quality / source correctness"
row of the project evaluation table. Needs only the embedding model, so it runs
without the LLM.

    python eval_retrieval.py
"""
from config import RETRIEVAL_TOP_K
from retrieve import retrieve

# (query, section label that SHOULD be retrieved)
TEST_SET = [
    ("someone walked into a restricted lab after hours", "Section 4 - After-Hours Rules"),
    ("a person is in a restricted zone without authorization", "Section 2 - Unauthorized Entry"),
    ("what risk tier is a server room", "Section 1 - Zone Classification"),
    ("how fast must an operator acknowledge a tier 3 alert", "Section 3 - Response Time Targets"),
    ("the detection confidence is only 0.4, what do we do", "Section 2 - Confidence Thresholds"),
    ("do we log incidents that get dismissed", "Section 4 - Documentation"),
    ("a large group has gathered and is not moving", "Section 2 - Density vs. Count"),
    ("how many people count as a crowd", "Section 1 - Occupancy Limits"),
    ("when does an incident become an emergency", "Section 1 - Escalation Triggers"),
    ("what makes an incident high severity", "Section 2 - Severity Assignment"),
    ("same zone keeps getting alerts all day", "Section 3 - Repeat Incidents"),
    ("an unescorted guest is in a lab", "Section 2 - Unescorted Visitor Detection"),
    ("is a detection in the lobby an incident", "Section 3 - Public Areas"),
    ("who do we call for an after-hours emergency", "Section 2 - Immediate Actions"),
]


def main():
    top1_hits = 0
    topk_hits = 0
    misses = []

    for query, expected_section in TEST_SET:
        hits = retrieve(query, k=RETRIEVAL_TOP_K)
        retrieved = [hit["section"] for hit in hits]

        if retrieved and expected_section in retrieved[0]:
            top1_hits += 1
        if any(expected_section in section for section in retrieved):
            topk_hits += 1
        else:
            misses.append((query, expected_section, retrieved))

    total = len(TEST_SET)
    print(f"Queries:           {total}")
    print(f"Top-1 accuracy:    {top1_hits}/{total} ({top1_hits / total:.1%})")
    print(f"Recall@{RETRIEVAL_TOP_K}:         {topk_hits}/{total} ({topk_hits / total:.1%})")

    if misses:
        print(f"\nMisses ({len(misses)}):")
        for query, expected, retrieved in misses:
            print(f"  q: {query}")
            print(f"    expected: {expected}")
            print(f"    got:      {retrieved}")


if __name__ == "__main__":
    main()
