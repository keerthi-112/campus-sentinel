"""
config.py — Settings and import paths for the Application track.

The backend is the integration point of all four tracks. It imports:
  * vision/     (repo root package)  — detection, tracking, zone + crowd logic
  * federated/  (repo root package)  — the federated event classifier
  * rag/        (flat modules)       — retrieval + report generation

rag/ uses flat imports (`import jobs`, `from config import ...`), so its folder
is put on sys.path directly. That is what makes `from api import router` work
without changing the RAG track's code.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = Path(os.getenv("SENTINEL_REPO_ROOT", BACKEND_DIR.parent)).resolve()
RAG_DIR = REPO_ROOT / "rag"
FEDERATED_DIR = REPO_ROOT / "federated"
DATA_DIR = Path(os.getenv("SENTINEL_DATA_DIR", BACKEND_DIR / "data")).resolve()
VIDEO_DIR = DATA_DIR / "videos"
DB_PATH = Path(os.getenv("SENTINEL_DB", DATA_DIR / "sentinel.db"))

for p in (str(REPO_ROOT),):
    if p not in sys.path:
        sys.path.insert(0, p)

SECRET_KEY = os.getenv("SENTINEL_SECRET", "change-me-in-production-campus-sentinel")
TOKEN_TTL_HOURS = int(os.getenv("SENTINEL_TOKEN_TTL_HOURS", "12"))
ADMIN_USER = os.getenv("SENTINEL_ADMIN_USER", "admin")
ADMIN_PASSWORD = os.getenv("SENTINEL_ADMIN_PASSWORD", "admin123")

# Optional shared key for POST /api/incidents/ingest from an external vision process.
INGEST_KEY = os.getenv("SENTINEL_INGEST_KEY", "")

YOLO_MODEL = os.getenv("SENTINEL_YOLO_MODEL", "yolo11n.pt")
DEVICE = os.getenv("SENTINEL_DEVICE", "cpu")

CORS_ORIGINS = [o.strip() for o in os.getenv(
    "SENTINEL_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()]

DATA_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
