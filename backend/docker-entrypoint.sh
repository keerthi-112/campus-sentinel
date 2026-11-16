#!/bin/sh
# Prepares the other tracks' artefacts on first start, then runs the API.
set -e
cd /workspace

# 1. Federated global classifier (seconds; only if none exists yet)
if [ ! -f federated/artifacts/global_model.json ]; then
  echo "[entrypoint] training initial federated classifier..."
  python -m federated.simulate --quiet || echo "[entrypoint] FL training failed — classifier will escalate everything"
fi

# 2. RAG knowledge base (needs Ollama with nomic-embed-text pulled)
if [ -n "$OLLAMA_HOST" ]; then
  for i in $(seq 1 30); do curl -sf "$OLLAMA_HOST/api/tags" >/dev/null && break; sleep 2; done
  if [ ! -d rag/chroma_db ]; then
    echo "[entrypoint] building RAG knowledge base..."
    (cd rag && python ingest.py) || echo "[entrypoint] ingest failed — pull the models into Ollama, then: docker compose exec backend sh -c 'cd /workspace/rag && python ingest.py'"
  fi
fi

cd /workspace/backend
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
