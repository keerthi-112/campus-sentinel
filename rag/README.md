# rag/

Owns: the safety knowledge base, chunking + embeddings, Chroma retrieval,
prompt assembly, and report generation.

Reads: `docs/schemas/incident.json`-shaped records (or a chat question) and the
curated safety documents.
Produces: `docs/schemas/report.json`-shaped records.

## Planned layout

```
rag/
  documents/        # curated safety docs (rules, SOPs, escalation, crowd mgmt)
  ingest.py         # chunk + embed documents into Chroma
  retrieve.py        # query embedding + top-k retrieval
  generate.py        # prompt assembly + Llama 3.1 8B call -> report.json
  chat.py            # operator chat entry point (bypasses incident pipeline)
```

Embeddings: `nomic-embed-text` via Ollama. LLM: `llama3.1:8b` via Ollama.
