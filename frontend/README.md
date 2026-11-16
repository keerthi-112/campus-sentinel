# frontend/

The React dashboard (Vite + React 18 + React Router + Recharts).

## Run

```bash
cd frontend
npm install
npm run dev            # http://localhost:5173  (proxies /api to http://127.0.0.1:8000)
```

The backend must be running (`cd backend && uvicorn app.main:app --port 8000`).
Sign in with **admin / admin123**.

To let other laptops on the same Wi-Fi open the dashboard, Vite already listens
on all interfaces: open `http://<your-ip>:5173`. In Docker, `BACKEND_URL=http://backend:8000`
points the proxy at the backend container.

## Screens (top tabs)

| Tab | What it shows | Backend |
|---|---|---|
| **Monitor** | Live MJPEG view (boxes, red zone polygon, per-person dwell timer) with camera picker and Start/Stop. Live **zone cards** below it: people in zone, dwell timer vs threshold, max confidence, motion, and the **federated classifier's live escalate probability**, plus a whole-frame crowd card. Incident list (Active / All / Closed / **Filtered by FL**) with the selected incident's facts, classifier decision ("why?" shows the feature vector) and **AI report** (suggested action + cited section). Safety-assistant chat on the right | `/api/cameras`, `/stream`, `/api/incidents` |
| **Reports** | KPI tiles (open, acknowledged, resolved, false alarms, filtered by FL, reports, avg report time), searchable report-card grid | `/api/incidents` |
| **Cameras & zones** | Add / edit cameras (file upload, webcam, RTSP), tier / dwell / confidence / crowd thresholds, **draw the restricted polygon on a real frame** | `/api/cameras`, `/snapshot`, `/upload-video` |
| **Federated learning** | KPI tiles, convergence chart (accuracy / F1 / loss, or FedAvg vs FedProx with local-only and centralized reference lines) that **updates live during training**, "does federation help?" table, per-zone client table, model in use, **Start training run** (simulation or Flower), round log | `/api/federated/*` |
| **Safety assistant** | Full-page operator chat with cited sections | `/api/chat` |

Status pills in the top bar: API, Vision (cameras live), Classifier (strategy · rounds), RAG / LLM.
New escalated incidents pop up as toasts; click one to open it.

## Layout

```
src/
  main.jsx, App.jsx          # router, sidebar shell, auth gate, error boundary
  api.js                     # fetch wrapper (Bearer token, 401 → login), stream URL
  hooks.js                   # usePoll + formatters
  pages/                     # Login, Monitor, Reports, Cameras, Federated, Assistant
  components/                # IncidentDetail, PolygonEditor, Toasts, ErrorBoundary
  styles.css
```
