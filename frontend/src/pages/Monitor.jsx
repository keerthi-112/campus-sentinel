import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api, streamUrl } from "../api";
import { usePoll } from "../hooks";
import IncidentDetail, { EVENT_LABEL, Sev, StatusText, timeOf } from "../components/IncidentDetail";
import Assistant from "./Assistant";

const FILTERS = {
  active: ["Active", (i) => ["open", "acknowledged"].includes(i.status)],
  all: ["All", (i) => i.status !== "suppressed"],
  closed: ["Closed", (i) => ["resolved", "false_alarm"].includes(i.status)],
  filtered: ["Filtered", (i) => i.status === "suppressed"],
};

function Bar({ value, max = 1, tone = "accent" }) {
  const w = Math.max(0, Math.min(1, (value || 0) / max)) * 100;
  return <div className="bar"><div className={tone} style={{ width: `${w}%` }} /></div>;
}

function ZoneCards({ cam }) {
  const rt = cam.runtime;
  const live = rt?.live || {};
  const running = rt?.state === "running";
  const zoneState = running ? live.state || "clear" : "offline";
  const score = live.score;
  const crowdOn = (live.person_count || 0) >= cam.crowd_count;

  return (
    <div className="zones">
      <div className={`zone ${zoneState === "alert" && live.event_type === "restricted_area_entry" ? "alert" : ""}`}>
        <div className="zone-head">
          <span className="tag restricted">restricted · tier {cam.tier}</span>
          <b>{cam.zone_name}</b>
          <span className={`state ${zoneState}`}>{zoneState === "watching" ? "person in zone" : zoneState}</span>
        </div>
        <div className="kpis">
          <div><label>In zone</label><b>{running ? rt.in_zone : "—"}</b></div>
          <div><label>Dwell</label><b>{running ? `${(live.zone_dwell_seconds || 0).toFixed(1)}s` : "—"}</b><small>/ {cam.dwell_threshold}s</small></div>
          <div><label>Max conf</label><b>{running ? (live.max_confidence || 0).toFixed(2) : "—"}</b></div>
          <div><label>Motion</label><b>{running ? (live.motion_score || 0).toFixed(2) : "—"}</b></div>
        </div>
        <div className="barrow"><span>Dwell timer</span><Bar value={live.zone_dwell_seconds} max={cam.dwell_threshold} tone={zoneState === "alert" ? "bad" : "warn"} /><em>{running ? `${Math.min(100, Math.round(((live.zone_dwell_seconds || 0) / cam.dwell_threshold) * 100))}%` : ""}</em></div>
        <div className="barrow"><span>FL escalate prob.</span><Bar value={score ?? 0} tone={score >= 0.5 ? "bad" : "ok"} /><em>{score == null ? "—" : `${Math.round(score * 100)}%`}</em></div>
      </div>

      <div className={`zone ${crowdOn && running ? "alert crowd" : ""}`}>
        <div className="zone-head">
          <span className="tag crowd">crowd</span>
          <b>Whole frame</b>
          <span className={`state ${!running ? "offline" : crowdOn ? "alert" : "clear"}`}>{!running ? "offline" : crowdOn ? "crowd" : "normal"}</span>
        </div>
        <div className="kpis">
          <div><label>People</label><b>{running ? rt.persons : "—"}</b><small>/ {cam.crowd_count}</small></div>
          <div><label>Density</label><b>{running ? `${Math.round((live.density || 0) * 100)}%` : "—"}</b></div>
          <div><label>FPS</label><b>{running ? rt.fps : "—"}</b></div>
          <div><label>Incidents</label><b>{rt?.incidents ?? 0}</b></div>
        </div>
        <div className="barrow"><span>Crowd threshold</span><Bar value={live.person_count} max={cam.crowd_count} tone={crowdOn ? "bad" : "ok"} /><em>{running ? `${live.person_count || 0}/${cam.crowd_count}` : ""}</em></div>
        <div className="barrow"><span>Density</span><Bar value={live.density} tone="accent" /><em>{running ? `${Math.round((live.density || 0) * 100)}%` : ""}</em></div>
      </div>
    </div>
  );
}

function VideoPanel() {
  const { data: cameras, reload } = usePoll("/api/cameras", 1500);
  const [camId, setCamId] = useState(null);
  const [key, setKey] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (cameras?.length && !cameras.find((c) => c.id === camId))
      setCamId((cameras.find((c) => c.active_state) || cameras[0]).id);
  }, [cameras, camId]);
  const cam = cameras?.find((c) => c.id === camId);
  const running = cam?.active_state;
  const src = useMemo(() => (camId ? streamUrl(camId) : null), [camId, key]); // eslint-disable-line

  const toggle = async () => {
    setBusy(true);
    setError(null);
    try {
      await api(`/api/cameras/${camId}/${running ? "stop" : "start"}`, { method: "POST" });
      await reload();
      setKey((k) => k + 1);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Live view</h2>
        {!!cameras?.length && (
          <div className="head-tools">
            <select value={camId ?? ""} onChange={(e) => { setCamId(Number(e.target.value)); setKey((k) => k + 1); }}>
              {cameras.map((c) => <option key={c.id} value={c.id}>{c.camera_name} {c.active_state ? "● live" : "○ off"}</option>)}
            </select>
            <button className={running ? "danger" : "primary"} disabled={busy} onClick={toggle}>{busy ? "…" : running ? "Stop" : "Start"}</button>
          </div>
        )}
      </div>
      {!cameras ? <div className="frame" /> : !cameras.length ? (
        <div className="empty"><strong>No cameras yet</strong><span>Add one in <Link to="/cameras">Cameras &amp; zones</Link>: upload a clip, use a webcam or an RTSP stream.</span></div>
      ) : cam && (
        <>
          <div className="frame">
            {src && <img key={key} src={src} alt={`${cam.camera_name} live view`} />}
            <span className="cam-tag">{cam.camera_name} · {cam.source_type}</span>
            {running && <span className="rec">● LIVE</span>}
          </div>
          {cam.runtime?.error && <div className="alert">{cam.runtime.error}</div>}
          {error && <div className="alert">{error}</div>}
          <ZoneCards cam={cam} />
        </>
      )}
    </section>
  );
}

function IncidentsPanel({ incidents, selected, setSelected, reload }) {
  const [filter, setFilter] = useState("active");
  const list = incidents.filter(FILTERS[filter][1]);
  const sel = incidents.find((i) => i.id === selected);

  return (
    <section className="panel incidents">
      <div className="panel-head">
        <h2>Incidents</h2>
        <div className="seg">
          {Object.entries(FILTERS).map(([k, [label, fn]]) => (
            <button key={k} className={filter === k ? "on" : ""} onClick={() => setFilter(k)}>
              {label}{k === "filtered" && <small> {incidents.filter(fn).length}</small>}
            </button>
          ))}
        </div>
      </div>
      <ul className="inc-list">
        {list.map((i) => (
          <li key={i.id} className={`${i.id === selected ? "on" : ""} ${i.status === "suppressed" ? "dim" : ""}`} onClick={() => setSelected(i.id)}>
            <div className="row1">
              <Sev value={i.severity} />
              <b>{EVENT_LABEL[i.event_type] || i.event_type}</b>
              <span className="time">{timeOf(i.timestamp)}</span>
            </div>
            <div className="row2">
              <span>{i.zone_name} · {i.camera_name} · conf {i.confidence.toFixed(2)} · {i.duration_seconds.toFixed(1)}s</span>
              <StatusText value={i.status} />
            </div>
          </li>
        ))}
        {!list.length && (
          <li className="empty small">
            {filter === "filtered" ? "Nothing filtered yet. Candidates the federated classifier rejects as false alarms appear here."
              : "No incidents. They appear when someone stays in a restricted zone past the dwell timer."}
          </li>
        )}
      </ul>
      {sel && <IncidentDetail inc={sel} onChange={reload} />}
    </section>
  );
}

export default function Monitor({ incidents, reloadIncidents, selected, setSelected }) {
  return (
    <div className="monitor">
      <VideoPanel />
      <IncidentsPanel incidents={incidents} selected={selected} setSelected={setSelected} reload={reloadIncidents} />
      <Assistant compact />
    </div>
  );
}
