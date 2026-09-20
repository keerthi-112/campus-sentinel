# rag/

Owns: the safety knowledge base, chunking + embeddings, Chroma retrieval,
prompt assembly, and report generation.

Reads: `docs/schemas/incident.json`-shaped records (or a chat question) and the
curated safety documents in `documents/`.
Produces: `docs/schemas/report.json`-shaped records.

## Setup

```
cd rag
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Requires [Ollama](https://ollama.com) running locally with both models pulled:

```
ollama pull nomic-embed-text
ollama pull llama3.1:8b
```

## Usage

```
python ingest.py                          # chunk + embed documents/ into Chroma (run once, and after editing docs)
python retrieve.py "your query here"       # sanity-check retrieval on its own
python generate.py                         # generate a report for the sample incident in docs/schemas/incident.json
python generate.py "your question here"    # generate a chat-style answer instead
python chat.py                             # interactive chat loop
```

## Layout

```
rag/
  documents/        # curated safety docs (rules, SOPs, escalation, crowd mgmt)
  config.py          # model names, paths, retrieval top-k
  ingest.py          # chunk (by "## " section) + embed documents -> Chroma
  retrieve.py         # query embedding + top-k retrieval
  generate.py          # prompt assembly + Llama 3.1 8B call -> report.json / chat answer
  chat.py               # operator chat CLI (bypasses the incident pipeline)
```

Embeddings: `nomic-embed-text` via Ollama. LLM: `llama3.1:8b` via Ollama. Both
run fully locally — no incident data or document content leaves the machine.

## Notes for integration

- `generate_report(incident)` in `generate.py` is what `backend/`'s future
  `/reports` endpoint should call once an incident is confirmed.
- `generate_chat_answer(question)` is what `backend/`'s future `/chat`
  endpoint should call — it never needs an incident.
- Every document is chunked by its `## ` headings; keep that convention when
  adding new documents so `source_section` citations stay meaningful (e.g.
  "Restricted Area Rules, Section 2 - Unauthorized Entry").
