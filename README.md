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
| `backend/` | FastAPI service, SQLite models, APIs | Application |
| `frontend/` | React dashboard + chat | Application |
| `docs/schemas/` | Shared data contracts every track codes against | All tracks |

## Shared contracts

Before wiring tracks together, read `docs/schemas/`. These three JSON shapes are
the interface every module agrees on:

- `event_feature_vector.json` — what `vision/` hands to `federated/`
- `incident.json` — what the event classifier hands to `rag/` and `backend/`
- `report.json` — what `rag/` hands back to `backend/` and `frontend/`

Changing a schema affects every track — flag it to the team before merging.

## Running locally

```
docker compose up
```

(Compose file currently stubs `backend` and `frontend`; each track adds its own
service block as it comes online.)

## Workflow

See `CONTRIBUTING.md` for the branch/PR process.
