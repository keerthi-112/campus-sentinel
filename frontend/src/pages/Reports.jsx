import { useState } from "react";
import { usePoll } from "../hooks";
import { EVENT_LABEL, ReportBody, Sev, dateTimeOf } from "../components/IncidentDetail";

function Kpi({ label, value }) {
  return <div className="kpi"><label>{label}</label><b>{value}</b></div>;
}

const fmt = (s, unit, digits = 1) => (s ? `${s.mean.toFixed(digits)} ${unit} (p95 ${s.p95.toFixed(digits)})` : "—");

// Template §14 "System": end-to-end latency and basic resource usage, measured live.
function SystemPanel() {
  const { data: m } = usePoll("/api/system/metrics", 3000);
  if (!m) return null;
  const L = m.latency || {};
  return (
    <section className="panel">
      <div className="panel-head"><h2>System performance</h2><span className="muted small">live, since backend start</span></div>
      <div className="sys-grid">
        <div><label>Backend CPU</label><b>{m.process.cpu_percent}%</b><small>machine {m.system.cpu_percent}% of {m.system.cpu_count} cores</small></div>
        <div><label>Backend memory</label><b>{Math.round(m.process.rss_mb)} MB</b><small>RAM in use {m.system.ram_used_percent}% of {m.system.ram_total_gb} GB</small></div>
        <div><label>Frame processing</label><b>{m.cameras.length ? `${m.cameras[0].frame_ms} ms` : "—"}</b>
          <small>{m.cameras.length ? m.cameras.map((c) => `${c.camera_name}: ${c.fps} fps`).join(" · ") : "no camera running"}</small></div>
        <div><label>Alert decision</label><b>{L.decision_ms ? `${L.decision_ms.mean.toFixed(0)} ms` : "—"}</b><small>vision candidate → FL classifier → stored</small></div>
        <div><label>Report ready after</label><b>{L.report_total_s ? `${L.report_total_s.mean.toFixed(1)} s` : "—"}</b><small>{fmt(L.report_generation_s, "s")} generating</small></div>
        <div><label>Report queue wait</label><b>{L.report_queue_s ? `${L.report_queue_s.mean.toFixed(1)} s` : "—"}</b><small>{L.report_total_s ? `${L.report_total_s.n} reports measured` : "no reports yet"}</small></div>
      </div>
    </section>
  );
}

export default function Reports({ incidents, open }) {
  const [q, setQ] = useState("");
  const count = (s) => incidents.filter((i) => i.status === s).length;
  const withReport = incidents.filter((i) => i.report);
  const times = withReport
    .map((i) => (new Date(i.report.created_time) - new Date(i.created_at)) / 1000)
    .filter((t) => t >= 0 && t < 3600);
  const avg = times.length ? (times.reduce((a, b) => a + b, 0) / times.length).toFixed(1) : "—";

  const needle = q.trim().toLowerCase();
  const cards = incidents
    .filter((i) => i.status !== "suppressed")
    .filter((i) => !needle || [i.zone_name, i.camera_name, i.event_type, i.report?.report_text, i.report?.source_section]
      .some((t) => (t || "").toLowerCase().includes(needle)));

  return (
    <div className="page">
      <div className="kpis-row">
        <Kpi label="Open" value={count("open")} />
        <Kpi label="Acknowledged" value={count("acknowledged")} />
        <Kpi label="Resolved" value={count("resolved")} />
        <Kpi label="False alarms" value={count("false_alarm")} />
        <Kpi label="Filtered by FL" value={count("suppressed")} />
        <Kpi label="Reports" value={withReport.length} />
        <Kpi label="Avg report time" value={avg === "—" ? avg : `${avg} s`} />
      </div>
      <SystemPanel />
      <section className="panel">
        <div className="panel-head">
          <h2>Incident reports</h2>
          <input className="search" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search reports…" />
        </div>
        {!cards.length ? <div className="empty small">No reports yet.</div> : (
          <div className="cards">
            {cards.map((i) => (
              <article key={i.id} className={`card-report sev-${i.severity}`} onClick={() => open(i.id)}>
                <div className="row1">
                  <Sev value={i.severity} />
                  <b>{EVENT_LABEL[i.event_type] || i.event_type}</b>
                  <span className="muted">incident-{String(i.id).padStart(5, "0")}</span>
                </div>
                <div className="muted small">{i.zone_name} · {i.camera_name} · {dateTimeOf(i.timestamp)} · {i.status.replace("_", " ")}</div>
                <ReportBody inc={i} />
              </article>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
