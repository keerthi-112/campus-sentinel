# Campus Sentinel

Privacy-preserving, federated campus safety intelligence system. Watches selected
video feeds, detects restricted-area entry and crowd formation, and turns
confirmed events into sourced incident reports — without centralizing raw video.

Pipeline: `Video → OpenCV → YOLO26n → tracking/zone logic → event features →
event classifier (federated via Flower) → RAG (Chroma) → LLM (Llama 3.1 8B) →
FastAPI → React dashboard`.

## Folder map

| Folder | Owns | Track |
|---|---|---|
| `vision/` | Video capture, YOLO26n inference, zone/crowd logic | Vision & Zone |
| `federated/` | Simulated zone clients, event classifier, Flower server | Federated Learning |
| `rag/` | Knowledge base, embeddings, Chroma retrieval, LLM prompts | RAG & LLM |
| `backend/` | FastAPI + SQLite, live camera workers, incident decision stage, RAG bridge | Application |
| `frontend/` | React dashboard: live view, incidents, zone editor, FL status, chat | Application |
| `scripts/`, `.vscode/` | one-time setup scripts, VS Code launch configs | Application |
| `docs/schemas/` | Shared data contracts every track codes against | All tracks |

## Shared contracts

Before wiring tracks together, read `docs/schemas/`. These three JSON shapes are
the interface every module agrees on:

- `event_feature_vector.json` — what `vision/` hands to `federated/`
- `incident.json` — what the event classifier hands to `rag/` and `backend/`
- `report.json` — what `rag/` hands back to `backend/` and `frontend/`

Changing a schema affects every track — flag it to the team before merging.

## Running the whole system

**Prerequisites:** Python 3.10–3.12, Node 20+, and [Ollama](https://ollama.com/download)
(only for reports and chat).

**Quickest: one terminal does everything.** It checks what is installed, installs only what is
missing, then runs the backend and dashboard together (Ctrl+C stops both):

```bash
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1     # Windows   (add -CheckOnly to only check)
bash scripts/run_all.sh                                        # macOS / Linux (--check-only)
```

Or step by step:

```bash
# one-time setup (venv, Python + npm packages, FL data + first global model, RAG index)
bash scripts/setup.sh                                          # macOS / Linux
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1     # Windows

# run (two terminals, venv active)
cd backend  && uvicorn app.main:app --port 8000                # API + live vision workers
cd frontend && npm run dev                                     # dashboard
```

Open **http://localhost:5173** and sign in with **admin / admin123**. Then:

1. **Cameras & zones**: add a camera (upload a clip, or use webcam `0` / an RTSP URL),
   save it, and draw the restricted polygon on the frame.
2. **Monitor**: pick the camera and press *Start*. A person who stays inside the polygon past
   the dwell time becomes a candidate. The **federated classifier** escalates it or
   suppresses it, and an escalated incident gets a **RAG report** with a cited policy section.
3. **Federated learning**: see the convergence chart, compare local-only, FedAvg, FedProx
   and centralized, and start a new training run (simulation or real Flower).
4. **Safety assistant**: ask procedure questions, which are answered from the same knowledge base.

In VS Code: *Run and Debug → ▶ Full system (backend + dashboard)*.
With Docker: `docker compose up --build` (see the comments in `docker-compose.yml`).

### Tests per track

```bash
python -m vision.test_vision              # 46 checks  (repo root)
cd rag && python test_api.py              # 31 checks
python -m federated.test_federated        # 10 tests   (repo root)
cd backend && python -m tests.test_backend   # 11 tests
```

## Workflow

See `CONTRIBUTING.md` for the branch/PR process.
