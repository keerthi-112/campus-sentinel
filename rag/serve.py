"""Run the RAG endpoints as a standalone service.

Useful before backend/ mounts the router — this track can be exercised and demoed
on its own:

    python serve.py          # then POST to http://127.0.0.1:8100/reports

Once the Application track mounts the router, this stays handy for testing the
RAG track in isolation from the rest of the system.
"""
import uvicorn
from fastapi import FastAPI

from api import router

app = FastAPI(title="Campus Sentinel — RAG service")
app.include_router(router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8100)
