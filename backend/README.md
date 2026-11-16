# backend/

The Application track's API: **FastAPI + SQLite**. It is also where the four tracks
meet:

```
vision/ camera workers ──► federated event classifier ──► incidents (SQLite)
   (YOLO + tracking +          escalate | suppress            │
    zone / crowd logic)                                       └─► rag/ report job ──► reports (SQLite)
React dashboard ◄── REST + MJPEG stream ── this API ── /api/chat ──► rag/ (retrieval + LLM)
```

## Setup and run

```bash
# from the repo root
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt                   # pulls in vision, rag, federated deps

python -m federated.simulate                              # trains the global classifier (~2 s)
cd rag && python ingest.py && cd ..                       # needs Ollama + nomic-embed-text

cd backend
uvicorn app.main:app --reload --port 8000
```

Open http://127.0.0.1:8000/docs for interactive API docs. Default login:
**admin / admin123** (change it with `SENTINEL_ADMIN_PASSWORD` before first start).

Tests (no Ollama, camera or GPU needed; the LLM call is stubbed):

```bash
cd backend && python -m tests.test_backend
```

## What it adds to the other tracks

| Concern | How |
|---|---|
| **Live monitoring** | `services/vision_worker.py` runs one thread per active camera, using the vision track's `PersonDetector`, `CentroidTracker`, `ZoneMonitor`, `CrowdMonitor` and `compute_features` unchanged. Video files are paced to their own FPS, so dwell timers measure video seconds. |
| **Event classifier** | `services/incidents.py`: every candidate + its feature vector goes to the federated `EventClassifier`. Escalated → `open` incident + RAG report. Suppressed → stored as `suppressed`, so you can evaluate false-alarm filtering, and an operator can still escalate it. |
| **Hot model swap** | When a federated run finishes, `global_model.json` is replaced atomically and the next decision uses the new model. No restart needed. |
| **RAG integration** | `services/rag_bridge.py` imports `rag/` (fixes the flat-import issue without touching their code), reuses their job queue and worker, and syncs finished reports into `reports`. Their own endpoints are also mounted at `/api/rag/...`. |
| **Privacy** | Frames exist only in memory for the live stream. The database stores configuration, event metadata, reports, FL round logs and users, never video. |

## Endpoints (all under `/api`, Bearer token unless noted)

| Method | Path | Purpose |
|---|---|---|
| POST | `/auth/login` | → `{token, user}` (no token needed) |
| GET | `/auth/me`, `/auth/users` · POST `/auth/users` | current user · admin user management |
| GET/POST | `/cameras` | list (with live runtime stats) / create camera + zone |
| GET/PUT/DELETE | `/cameras/{id}` | read / update (restarts the worker with the new zone) / delete |
| POST | `/cameras/{id}/start`, `/cameras/{id}/stop` | start / stop monitoring |
| GET | `/cameras/{id}/snapshot` | clean frame + `X-Frame-Width/Height` for the polygon editor |
| GET | `/cameras/{id}/stream?token=` | MJPEG live view with boxes, zone and dwell timers |
| POST | `/cameras/upload-video` · GET `/cameras/videos/list` | store a demo clip locally |
| GET | `/incidents`, `/incidents/stats`, `/incidents/{id}` | incident feed (+ report, classifier decision) |
| PATCH | `/incidents/{id}` | `open → acknowledged → resolved / false_alarm` |
| POST | `/incidents/{id}/report` | (re)generate report, or escalate a suppressed candidate |
| POST | `/incidents/ingest` | **no token** (optional `X-Ingest-Key`): external vision process posts `incident.json` (+ optional `features`) |
| GET | `/reports`, `/reports/{incident_ref}` | generated reports |
| GET | `/federated/model`, `/federated/rounds`, `/federated/summary`, `/federated/history`, `/federated/status` | FL status panel |
| POST | `/federated/train` | admin: start a simulation or Flower run in the background |
| POST | `/chat` | operator question → answer + cited section |
| GET | `/system/status` | vision / classifier / RAG / Ollama health |
| — | `/rag/reports`, `/rag/chat` | the RAG track's own router, unchanged |

Run the vision track's own CLI against this backend instead of the built-in workers:

```bash
python -m vision.run_demo --source clip.mp4 --type file --rag-url http://127.0.0.1:8000/api/incidents/ingest
```

## Configuration (environment variables)

| Variable | Default |
|---|---|
| `SENTINEL_ADMIN_USER` / `SENTINEL_ADMIN_PASSWORD` | `admin` / `admin123` |
| `SENTINEL_SECRET` | dev value. **Set a long random string in production** |
| `SENTINEL_YOLO_MODEL` | `yolo11n.pt` (set `yolo26n.pt` to compare) |
| `SENTINEL_DEVICE` | `cpu` (`cuda`, `mps`) |
| `SENTINEL_DATA_DIR` | `backend/data` (SQLite DB, uploaded clips, training log) |
| `SENTINEL_INGEST_KEY` | empty = ingest endpoint open on localhost |
| `SENTINEL_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` |
| `OLLAMA_HOST` | read by the `ollama` client, e.g. `http://ollama:11434` in Docker |

## Layout

```
backend/
  app/
    main.py            # app, startup (DB, admin, RAG sync, camera resume), routers
    config.py          # settings + sys.path for vision/ federated/ rag/
    db.py, models.py   # SQLite + the §06 data model (users, cameras, incidents, reports, federated_rounds)
    schemas.py, auth.py
    routers/           # auth, cameras, incidents, misc (reports, federated, chat, system)
    services/          # vision_worker, incidents (decision stage), rag_bridge, training
  tests/test_backend.py
  Dockerfile, docker-entrypoint.sh, requirements.txt
```
