#!/usr/bin/env bash
# Campus Sentinel: check everything, install what is missing, run backend + dashboard
# in ONE terminal. Ctrl+C stops both.
#
#   bash scripts/run_all.sh                 # check + install + run
#   bash scripts/run_all.sh --check-only    # only report what is missing
#   MODEL=llama3.1:8b bash scripts/run_all.sh
#   SKIP_OLLAMA=1 bash scripts/run_all.sh
set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
MODEL="${MODEL:-llama3.2:3b}"
CHECK_ONLY=0; [ "$1" = "--check-only" ] && CHECK_ONLY=1
VPY="$ROOT/.venv/bin/python"
MISSING=()
ok()   { printf "  \033[32m[OK]\033[0m      %s\n" "$1"; }
miss() { printf "  \033[33m[MISSING]\033[0m %s\n" "$1"; MISSING+=("$1"); }
bad()  { printf "  \033[31m[ERROR]\033[0m   %s\n" "$1"; }
step() { printf "\n\033[36m== %s\033[0m\n" "$1"; }

step "Checking tools"
PY=""
for c in python3.11 python3.12 python3.10 python3; do
  if command -v $c >/dev/null 2>&1; then
    minor=$($c -c 'import sys;print(sys.version_info[1])')
    if [ "$minor" -ge 10 ] && [ "$minor" -le 13 ]; then PY=$c; ok "$($c --version) ($c)"; break; fi
  fi
done
[ -z "$PY" ] && [ ! -x "$VPY" ] && { bad "Python 3.10-3.13 not found"; exit 1; }
command -v node >/dev/null && ok "Node $(node -v)" || { bad "Node.js 20+ not found"; exit 1; }
HAS_OLLAMA=0
if [ -n "$SKIP_OLLAMA" ]; then echo "  [SKIP]    Ollama"
elif command -v ollama >/dev/null; then HAS_OLLAMA=1; ok "Ollama installed"
else echo "  [WARN]    Ollama not installed: reports + chat won't work (https://ollama.com/download)"; fi

step "Checking Python packages"
if [ -x "$VPY" ]; then ok ".venv exists"; else miss ".venv"; [ $CHECK_ONLY = 0 ] && $PY -m venv .venv && ok ".venv created"; fi
if [ -x "$VPY" ] && "$VPY" -c "import fastapi, uvicorn, sqlalchemy, multipart, httpx, cv2, numpy, ultralytics, chromadb, ollama, flwr, matplotlib" 2>/dev/null; then
  ok "all Python packages installed"
else
  miss "Python packages"
  if [ $CHECK_ONLY = 0 ]; then
    "$VPY" -m pip install --upgrade pip && "$VPY" -m pip install -r backend/requirements.txt && ok "Python packages installed"
  fi
fi

step "Checking dashboard packages"
if [ -d frontend/node_modules/vite ] && [ -d frontend/node_modules/recharts ] && [ -d frontend/node_modules/react-router-dom ]; then ok "node_modules present"
else miss "frontend/node_modules"; [ $CHECK_ONLY = 0 ] && (cd frontend && npm install) && ok "npm packages installed"; fi

step "Checking federated model"
if [ -f federated/artifacts/global_model.json ]; then ok "global model present"
else
  miss "federated global model"
  if [ $CHECK_ONLY = 0 ]; then "$VPY" -m federated.data --clients 5 >/dev/null && "$VPY" -m federated.simulate --quiet && ok "global model trained"; fi
fi

if [ $HAS_OLLAMA = 1 ]; then
  step "Checking Ollama + RAG knowledge base"
  if ! curl -sf http://127.0.0.1:11434/api/tags >/dev/null; then
    miss "Ollama server not running"
    if [ $CHECK_ONLY = 0 ]; then
      (ollama serve >/dev/null 2>&1 &)
      for i in $(seq 1 15); do curl -sf http://127.0.0.1:11434/api/tags >/dev/null && break; sleep 1; done
    fi
  else ok "Ollama server running"; fi
  if curl -sf http://127.0.0.1:11434/api/tags >/dev/null; then
    TAGS=$(curl -sf http://127.0.0.1:11434/api/tags)
    for m in nomic-embed-text "$MODEL"; do
      if echo "$TAGS" | grep -q "\"$m"; then ok "model $m"
      else miss "model $m"; [ $CHECK_ONLY = 0 ] && ollama pull "$m"; fi
    done
    if ! grep -q "LLM_MODEL = \"$MODEL\"" rag/config.py; then
      if [ $CHECK_ONLY = 0 ]; then
        sed -i.bak "s/^LLM_MODEL = \"[^\"]*\"/LLM_MODEL = \"$MODEL\"/" rag/config.py && rm -f rag/config.py.bak
        ok "rag/config.py now uses $MODEL"
      else miss "rag/config.py LLM_MODEL is not $MODEL"; fi
    fi
    KB=$(cd rag && "$VPY" -c "from retrieve import get_collection; print(get_collection().count())" 2>/dev/null || echo 0)
    if [ "${KB:-0}" -gt 0 ] 2>/dev/null; then ok "RAG knowledge base: $KB sections indexed"
    else miss "RAG knowledge base is empty"; [ $CHECK_ONLY = 0 ] && (cd rag && "$VPY" ingest.py); fi
  fi
fi

step "Summary"
if [ $CHECK_ONLY = 1 ]; then
  if [ ${#MISSING[@]} -eq 0 ]; then ok "everything is ready. Run without --check-only to start"
  else echo "  missing: ${MISSING[*]}. Run without --check-only to install them"; fi
  exit 0
fi

for port in 8000 5173; do
  if (command -v lsof >/dev/null && lsof -iTCP:$port -sTCP:LISTEN >/dev/null 2>&1); then bad "port $port already in use"; exit 1; fi
done

step "Starting backend on http://127.0.0.1:8000"
(cd backend && exec "$VPY" -m uvicorn app.main:app --host 0.0.0.0 --port 8000) &
BACKEND=$!
cleanup() { kill $BACKEND 2>/dev/null || true; wait $BACKEND 2>/dev/null || true; printf "\nStopped backend and dashboard.\n"; }
trap cleanup EXIT INT TERM
for i in $(seq 1 60); do curl -sf http://127.0.0.1:8000/health >/dev/null && break; kill -0 $BACKEND 2>/dev/null || { bad "backend crashed"; exit 1; }; sleep 1; done
curl -sf http://127.0.0.1:8000/health >/dev/null || { bad "backend did not start"; exit 1; }
ok "backend running"

step "Starting dashboard: open http://localhost:5173 (admin / admin123). Ctrl+C stops everything."
cd frontend && npm run dev
