import { useState } from "react";
import { api } from "../api";

export const EVENT_LABEL = {
  restricted_area_entry: "Restricted Area Entry",
  crowd_formation: "Crowd Formation",
};

export const timeOf = (iso) => {
  const d = new Date(iso);
  return isNaN(d) ? iso : d.toLocaleTimeString();
};
export const dateTimeOf = (iso) => {
  const d = new Date(iso);
  return isNaN(d) ? iso : d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
};

export function Sev({ value }) {
  return <span className={`sev ${value}`}>{value}</span>;
}

export function StatusText({ value }) {
  return <span className={`status ${value}`}>{value.replace("_", " ")}</span>;
}

export function ReportBody({ inc }) {
  if (inc.report)
    return (
      <>
        <p className="report-text">{inc.report.report_text}</p>
        <div className="action">
          <span>Suggested action</span>
          {inc.report.suggested_action}
        </div>
        <div className="source">Source: {inc.report.source_section}</div>
      </>
    );
  if (inc.status === "suppressed")
    return <p className="muted">The federated classifier judged this candidate a false alarm, so no report was written. You can still escalate it.</p>;
  if (inc.report_status === "failed")
    return <p className="bad-text">{inc.report_error || "Report generation failed."}</p>;
  return <p className="generating">Retrieving the safety procedure and writing the report…</p>;
}

// Selected incident: actions, facts, classifier decision and the RAG report.
export default function IncidentDetail({ inc, onChange }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [showFeatures, setShowFeatures] = useState(false);

  const act = async (fn) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
      onChange?.();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };
  const setStatus = (status) => act(() => api(`/api/incidents/${inc.id}`, { method: "PATCH", body: { status } }));
  const regenerate = () => act(() => api(`/api/incidents/${inc.id}/report`, { method: "POST" }));
  const f = inc.features || {};
  const pct = inc.classifier_score == null ? "n/a" : `${Math.round(inc.classifier_score * 100)}%`;

  return (
    <div className="detail">
      <div className="detail-title">
        <h3>incident-{String(inc.id).padStart(5, "0")}</h3>
        <StatusText value={inc.status} />
      </div>
      <div className="actions">
        {inc.status === "suppressed" ? (
          <button className="primary" disabled={busy} onClick={regenerate}>Escalate anyway</button>
        ) : (
          <>
            <button disabled={busy || inc.status !== "open"} onClick={() => setStatus("acknowledged")}>Acknowledge</button>
            <button className="primary" disabled={busy || !["open", "acknowledged"].includes(inc.status)} onClick={() => setStatus("resolved")}>Resolve</button>
            <button disabled={busy || !["open", "acknowledged"].includes(inc.status)} onClick={() => setStatus("false_alarm")}>False alarm</button>
            {["resolved", "false_alarm"].includes(inc.status) && <button disabled={busy} onClick={() => setStatus("open")}>Re-open</button>}
          </>
        )}
      </div>
      {error && <div className="alert">{error}</div>}

      <div className="facts">
        <div><label>Event</label><b>{EVENT_LABEL[inc.event_type] || inc.event_type}</b></div>
        <div><label>Location</label><b>{inc.zone_name}</b></div>
        <div><label>Camera</label><b>{inc.camera_name}</b></div>
        <div><label>Time</label><b>{timeOf(inc.timestamp)}</b></div>
        <div><label>Confidence</label><b>{inc.confidence.toFixed(2)}</b></div>
        <div><label>Duration</label><b>{inc.duration_seconds.toFixed(1)} s</b></div>
        <div><label>People</label><b>{f.person_count ?? "—"}</b></div>
        <div><label>Classifier</label><b className={inc.classifier_decision === "suppressed" ? "muted" : ""}>{pct}</b></div>
      </div>

      <div className="clf-line">
        <span className={`decision ${inc.classifier_decision}`}>{inc.classifier_decision}</span>
        <div className="meter"><div style={{ width: `${(inc.classifier_score ?? 1) * 100}%` }} /><i /></div>
        <button className="ghost small" onClick={() => setShowFeatures((v) => !v)}>{showFeatures ? "hide" : "why?"}</button>
      </div>
      {showFeatures && (
        <div className="why">
          <p className="muted small">{inc.decision_reason}{inc.model_version ? ` · model ${inc.model_version}` : ""}</p>
          <div className="facts compact">
            <div><label>Density</label><b>{f.density ?? "—"}</b></div>
            <div><label>Motion</label><b>{f.motion_score ?? "—"}</b></div>
            <div><label>Dwell</label><b>{f.zone_dwell_seconds ?? "—"} s</b></div>
            <div><label>Max conf</label><b>{f.max_confidence ?? "—"}</b></div>
            <div><label>Occupancy</label><b>{f.occupancy ? "yes" : "no"}</b></div>
            <div><label>Source</label><b>{inc.source}</b></div>
          </div>
        </div>
      )}

      <div className="report">
        <div className="report-head">
          <span>AI report</span>
          {inc.report && !["pending", "running"].includes(inc.report_status) && inc.status !== "suppressed" && (
            <button className="ghost small" disabled={busy} onClick={regenerate}>regenerate</button>
          )}
          {inc.report_status === "failed" && <button className="ghost small" disabled={busy} onClick={regenerate}>retry</button>}
        </div>
        <ReportBody inc={inc} />
      </div>
    </div>
  );
}
