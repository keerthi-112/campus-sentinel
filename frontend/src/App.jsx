import { useEffect, useRef, useState } from "react";
import { Navigate, NavLink, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { auth } from "./api";
import { usePoll } from "./hooks";
import ErrorBoundary from "./components/ErrorBoundary";
import Toasts from "./components/Toasts";
import Login from "./pages/Login";
import Monitor from "./pages/Monitor";
import Reports from "./pages/Reports";
import Cameras from "./pages/Cameras";
import Federated from "./pages/Federated";
import Assistant from "./pages/Assistant";

const TABS = [
  ["/", "Monitor"],
  ["/reports", "Reports"],
  ["/cameras", "Cameras & zones"],
  ["/federated", "Federated learning"],
  ["/assistant", "Safety assistant"],
];

function Pill({ ok, warn, label, title }) {
  return <span className={`pill ${ok ? "ok" : warn ? "warn" : "bad"}`} title={title}><i />{label}</span>;
}

function Shell({ user, onLogout }) {
  const location = useLocation();
  const navigate = useNavigate();
  const { data: status, error: statusErr } = usePoll("/api/system/status", 5000);
  const { data: incidents, reload: reloadIncidents } = usePoll("/api/incidents?limit=200", 2000);
  const [selected, setSelected] = useState(null);
  const [toasts, setToasts] = useState([]);
  const seen = useRef(null);

  // a toast for every new escalated incident
  useEffect(() => {
    if (!incidents) return;
    const ids = new Set(incidents.map((i) => i.id));
    if (seen.current) {
      const fresh = incidents.filter((i) => !seen.current.has(i.id) && i.status !== "suppressed");
      if (fresh.length) {
        setToasts((t) => [...t, ...fresh.map((i) => ({ key: `${i.id}-${Date.now()}`, inc: i }))].slice(-3));
        setSelected((cur) => cur ?? fresh[0].id);
      }
    }
    seen.current = ids;
  }, [incidents]);

  const active = (incidents || []).filter((i) => i.status === "open").length;
  const clf = status?.classifier;
  const rag = status?.rag;
  const vision = status?.vision;
  const ragReady = rag?.available && rag?.kb_sections > 0;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo" aria-hidden>◉</span>
          <div>
            <h1>Campus Sentinel</h1>
            <p>Privacy-preserving federated campus safety</p>
          </div>
        </div>
        <nav className="tabs">
          {TABS.map(([to, label]) => (
            <NavLink key={to} to={to} end={to === "/"} className={({ isActive }) => (isActive ? "on" : "")}>
              {label}
              {to === "/" && active > 0 && <span className="badge">{active}</span>}
            </NavLink>
          ))}
        </nav>
        <div className="pills">
          <Pill ok={status && !statusErr} label={statusErr ? "API offline" : "API online"} />
          <Pill ok={vision?.running > 0} warn={vision?.available && !vision?.running}
                label={vision ? (vision.available ? `Vision: ${vision.running}/${vision.cameras} live` : "Vision: unavailable") : "Vision: …"}
                title={vision?.error || ""} />
          <Pill ok={clf?.loaded} warn={clf && !clf.loaded}
                label={clf?.loaded ? `Classifier: ${clf.strategy?.toUpperCase()} · ${clf.round_number} rds` : "Classifier: escalate-all fallback"} />
          <Pill ok={ragReady && rag?.ollama_reachable} warn={rag?.available && !(ragReady && rag?.ollama_reachable)}
                title={rag?.error || rag?.ollama_error || rag?.kb_error || ""}
                label={!rag ? "RAG: …" : !rag.available ? "RAG: not connected"
                  : !rag.ollama_reachable ? "RAG: start Ollama"
                  : !ragReady ? "RAG: run ingest.py" : `RAG: ${rag.llm_model}`} />
          <button className="ghost small" title={`Signed in as ${user.username} (${user.role})`} onClick={onLogout}>
            Sign out
          </button>
        </div>
      </header>

      <main>
        <ErrorBoundary resetKey={location.pathname}>
          <Routes>
            <Route path="/" element={<Monitor incidents={incidents || []} reloadIncidents={reloadIncidents}
                                              selected={selected} setSelected={setSelected} />} />
            <Route path="/reports" element={<Reports incidents={incidents || []}
                                                     open={(id) => { setSelected(id); navigate("/"); }} />} />
            <Route path="/cameras" element={<Cameras />} />
            <Route path="/federated" element={<Federated isAdmin={user.role === "admin"} />} />
            <Route path="/assistant" element={<div className="single"><Assistant /></div>} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </ErrorBoundary>
      </main>
      <Toasts toasts={toasts} setToasts={setToasts} onOpen={(id) => { setSelected(id); navigate("/"); }} />
    </div>
  );
}

export default function App() {
  const [user, setUser] = useState(auth.user);
  const navigate = useNavigate();
  const logout = () => {
    auth.clear();
    setUser(null);
    navigate("/login");
  };
  if (!user || !auth.token) {
    return (
      <Routes>
        <Route path="/login" element={<Login onLogin={setUser} />} />
        <Route path="*" element={<Navigate to="/login" replace />} />
      </Routes>
    );
  }
  return <Shell user={user} onLogout={logout} />;
}
