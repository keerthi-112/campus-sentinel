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
ollama pull nomic-embed-text     # embeddings — required for ingest/retrieve
ollama pull llama3.2:3b          # generation — required for reports/chat
```

`config.py` points `LLM_MODEL` at `llama3.2:3b`, which is what the results below
were measured on. Any instruct model Ollama can serve works — set it to
`llama3.1:8b` on stronger hardware. Only the model name changes; nothing else does.

## Usage

```
python ingest.py                          # chunk + embed documents/ into Chroma (run once, and after editing docs)
python retrieve.py "your query here"       # sanity-check retrieval on its own
python generate.py                         # generate a report for the sample incident in docs/schemas/incident.json
python generate.py "your question here"    # generate a chat-style answer instead
python chat.py                             # interactive chat loop
python eval_retrieval.py                   # retrieval benchmark (embedding model only)
python eval_reports.py                     # report-quality benchmark (needs the LLM)
```

## Evaluation

Two harnesses cover this track's rows in the project evaluation table.

`eval_retrieval.py` runs 14 labelled queries and reports top-1 accuracy and
recall@k. Recall@k is the number that matters — every retrieved section goes
into the prompt, so a correct section anywhere in the top-k is a usable result.
Current: **100% recall@3**, 79% top-1.

The three top-1 misses are all near-ties where the section ranked first is also
a defensible answer — e.g. "someone walked into a restricted lab after hours"
returns *Unauthorized Entry* at 0.297 and *After-Hours Rules* at 0.299. Treat
top-1 here as a loose signal and recall@k as the real measure, since every
retrieved section reaches the prompt anyway.

`eval_reports.py` generates reports for four synthetic incidents and checks that
every required field is present, and that the cited `source_section` is both a
real section in the knowledge base and one that was actually retrieved — an
invented citation is the clearest automatable faithfulness failure.
Current: **4/4 on all three checks**.

Citations are correct by construction rather than by prompting: the model picks
*which* excerpt it relied on, and `_resolve_citation` in `generate.py` snaps that
answer onto one of the labels actually retrieved. Left to its own formatting,
`llama3.2:3b` variously shortened labels, wrapped them in brackets, and spliced
two together — all of which produced citations an operator couldn't look up.

## Known limitation

Retrieval is semantic, so **conditional numeric policies don't reliably surface**.
*Security Procedures, Section 2 - Confidence Thresholds* ("below 0.6 should be
reviewed manually") is not retrieved for a 0.55-confidence incident even at k=5,
because the section is phrased in operator-procedure language that doesn't sit
near incident language in embedding space. The generated report is therefore
silent about the manual-review requirement.

Worth fixing later with a rule pass that injects threshold-triggered sections
regardless of embedding distance — semantic search is the wrong tool for
"if confidence < 0.6". Noted here rather than papered over.

## Layout

```
rag/
  documents/        # curated safety docs (rules, SOPs, escalation, crowd mgmt)
  config.py          # model names, paths, retrieval top-k
  ingest.py          # chunk (by "## " section) + embed documents -> Chroma
  retrieve.py         # query embedding + top-k retrieval
  generate.py          # prompt assembly + LLM call -> report.json / chat answer
  chat.py               # operator chat CLI (bypasses the incident pipeline)
  api.py                 # FastAPI router: POST /reports, POST /chat
  eval_retrieval.py       # labelled retrieval benchmark
  eval_reports.py          # report field/citation benchmark
```

Embeddings: `nomic-embed-text` via Ollama. LLM: `llama3.1:8b` via Ollama. Both
run fully locally — no incident data or document content leaves the machine.

## Notes for integration

The Application track mounts this track's endpoints with one line in
`backend/app/main.py`:

```python
from rag.api import router as rag_router
app.include_router(rag_router)
```

That gives `POST /reports` (body: an `incident.json`-shaped object) and
`POST /chat` (body: `{"question": "..."}`). The handlers live here rather than
in `backend/` so prompt and signature changes don't need a cross-track edit.
If the LLM isn't available, both return a 503 whose message names the exact
`ollama pull` that fixes it.

Calling the functions directly works too:

- `generate_report(incident)` — once an incident is confirmed.
- `generate_chat_answer(question)` — never needs an incident.
- Every document is chunked by its `## ` headings; keep that convention when
  adding new documents so `source_section` citations stay meaningful (e.g.
  "Restricted Area Rules, Section 2 - Unauthorized Entry").
