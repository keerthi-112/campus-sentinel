"""Turn a confirmed incident (or an operator question) into a sourced report/answer
using retrieved safety-policy sections and Llama 3.1 8B.

Report shape matches docs/schemas/report.json.
"""
import json
from datetime import datetime, timezone

import ollama

from config import LLM_MODEL, RETRIEVAL_TOP_K
from retrieve import retrieve

REPORT_SYSTEM_PROMPT = """You are a campus safety report writer. You are given a \
confirmed incident and relevant excerpts from the campus safety policy. Write a \
short, factual incident report grounded ONLY in the incident data and the provided \
policy excerpts — never invent a rule that isn't in the excerpts.

Respond with ONLY a JSON object with exactly these fields:
{
  "report_text": "2-3 sentence factual summary of what happened",
  "suggested_action": "the specific action the policy excerpts call for",
  "source_section": "the single most relevant section label from the excerpts"
}"""

CHAT_SYSTEM_PROMPT = """You are the campus safety knowledge assistant. Answer the \
operator's question using ONLY the provided policy excerpts. If the excerpts don't \
cover the question, say so plainly instead of guessing.

Respond with ONLY a JSON object with exactly these fields:
{
  "answer": "your answer, grounded in the excerpts",
  "source_section": "the single most relevant section label from the excerpts"
}"""


def _format_excerpts(hits: list[dict]) -> str:
    return "\n\n".join(f"[{hit['section']}]\n{hit['text']}" for hit in hits)


def _ask(system_prompt: str, user_prompt: str) -> dict:
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        format="json",
    )
    return json.loads(response["message"]["content"])


def generate_report(incident: dict, top_k: int = RETRIEVAL_TOP_K) -> dict:
    query = f"{incident['event_type']} in {incident['zone_name']} ({incident.get('severity', 'unknown')} severity)"
    hits = retrieve(query, k=top_k)

    user_prompt = (
        f"Incident:\n{json.dumps(incident, indent=2)}\n\n"
        f"Relevant policy excerpts:\n{_format_excerpts(hits)}"
    )
    parsed = _ask(REPORT_SYSTEM_PROMPT, user_prompt)

    return {
        "incident_ref": incident.get("incident_ref", incident.get("event_type")),
        "report_text": parsed["report_text"],
        "suggested_action": parsed["suggested_action"],
        "source_section": parsed.get("source_section", hits[0]["section"]),
        "created_time": datetime.now(timezone.utc).isoformat(),
    }


def generate_chat_answer(question: str, top_k: int = RETRIEVAL_TOP_K) -> dict:
    hits = retrieve(question, k=top_k)
    user_prompt = f"Question: {question}\n\nRelevant policy excerpts:\n{_format_excerpts(hits)}"
    parsed = _ask(CHAT_SYSTEM_PROMPT, user_prompt)

    return {
        "answer": parsed["answer"],
        "source_section": parsed.get("source_section", hits[0]["section"]),
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sample_incident_path = Path(__file__).parent.parent / "docs" / "schemas" / "incident.json"
    incident = json.loads(sample_incident_path.read_text(encoding="utf-8"))
    incident.pop("_description", None)

    if len(sys.argv) > 1:
        answer = generate_chat_answer(" ".join(sys.argv[1:]))
        print(json.dumps(answer, indent=2))
    else:
        report = generate_report(incident)
        print(json.dumps(report, indent=2))
