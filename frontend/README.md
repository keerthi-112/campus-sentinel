# frontend/

Owns: the React dashboard (video view, active incidents, past reports,
federated round status, operator chat).

Not scaffolded yet — the Application track owner should run, from inside this
folder:

```
npm create vite@latest . -- --template react
```

Then add a `Dockerfile` here (Vite dev server on port 5173) so it matches what
`docker-compose.yml` at the repo root already expects.

## Screens to build against `docs/schemas/`

- Video view: camera name, zone boundary, detections, confidence
- Incidents: list of `incident.json` records + current status
- Reports: `report.json` records (text, suggested action, source section)
- Federated status: round number, strategy, participating clients, metrics
- Chat: free-text box hitting the backend's `/chat` endpoint
