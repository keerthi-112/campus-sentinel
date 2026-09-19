from fastapi import FastAPI

app = FastAPI(title="Campus Sentinel API")


@app.get("/health")
def health():
    return {"status": "ok"}


# Planned routers, added as each track lands:
#   /cameras, /zones   -> vision/ config
#   /incidents          -> event classifier output (docs/schemas/incident.json)
#   /reports             -> rag/ output (docs/schemas/report.json)
#   /federated/rounds     -> federated/ round metrics
#   /chat                 -> proxies to rag/chat.py
