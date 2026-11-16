#!/usr/bin/env bash
# One-time setup on macOS / Linux. Run from the repo root:  bash scripts/setup.sh
set -e
cd "$(dirname "$0")/.."
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r backend/requirements.txt          # backend + vision + rag + federated
python -m federated.data --clients 5             # simulated zone datasets
python -m federated.simulate --quiet             # first global classifier
(cd frontend && npm install)
if command -v ollama >/dev/null 2>&1; then
  ollama pull nomic-embed-text && ollama pull llama3.2:3b && (cd rag && python ingest.py) \
    || echo "RAG ingest skipped. Start Ollama, then: cd rag && python ingest.py"
else
  echo "Ollama not found. Install it from https://ollama.com/download for reports + chat."
fi
echo "Done. Start: (1) cd backend && ../.venv/bin/uvicorn app.main:app --port 8000   (2) cd frontend && npm run dev"
