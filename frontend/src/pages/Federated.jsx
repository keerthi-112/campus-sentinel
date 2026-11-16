import { useEffect, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api";
import { usePoll } from "../hooks";

const C = { acc: "#4f9cf9", f1: "#36c28f", loss: "#e8a33d", fedavg: "#4f9cf9", fedprox: "#c084fc" };
const pct = (v) => (v == null || isNaN(v) ? "—" : `${(v * 100).toFixed(1)}%`);
const kib = (b) => (b == null ? "—" : b > 1024 * 1024 ? `${(b / 1048576).toFixed(2)} MB` : `${(b / 1024).toFixed(1)} KB`);

function Kpi({ label, value }) {
  return <div className="kpi"><label>{label}</label><b>{value}</b></div>;
}

const tooltipStyle = { background: "#151b23", border: "1px solid #2a3340", borderRadius: 8, fontSize: 12 };

function Convergence({ rounds, summary }) {
  const strategies = Object.keys(rounds?.strategies || {});
  const [view, setView] = useState(null);
  const current = view && (view === "compare" || strategies.includes(view)) ? view : strategies[0];
  if (!strategies.length) return <div className="empty small">No rounds yet. Start a training run on the right.</div>;

  let data, lines, refs = [];
  if (current === "compare") {
    const n = Math.max(...strategies.map((s) => rounds.strategies[s].length));
    data = Array.from({ length: n }, (_, k) => {
      const row = { round: k + 1 };
      strategies.forEach((s) => { row[s] = rounds.strategies[s][k]?.metrics.accuracy; });
      return row;
    });
    lines = strategies.map((s) => <Line key={s} type="monotone" dataKey={s} name={`${s.toUpperCase()} accuracy`} stroke={C[s] || "#999"} dot={false} strokeWidth={2.2} />);
    if (summary?.local_only_mean_accuracy) refs.push(["local-only mean", summary.local_only_mean_accuracy]);
    if (summary?.centralized) refs.push(["centralized", summary.centralized.global_test.accuracy]);
  } else {
    data = rounds.strategies[current].map((r) => ({ round: r.round_number, acc: r.metrics.accuracy, f1: r.metrics.f1, loss: r.metrics.loss }));
    lines = [
      <Line key="acc" type="monotone" dataKey="acc" name="Global test accuracy" stroke={C.acc} dot={false} strokeWidth={2.2} />,
      <Line key="f1" type="monotone" dataKey="f1" name="F1" stroke={C.f1} dot={false} strokeWidth={2.2} />,
      <Line key="loss" type="monotone" dataKey="loss" name="Test loss" stroke={C.loss} dot={false} strokeWidth={2.2} />,
    ];
  }

  return (
    <>
      <div className="panel-head">
        <h2>Convergence</h2>
        <select value={current} onChange={(e) => setView(e.target.value)}>
          {strategies.map((s) => <option key={s} value={s}>{s.toUpperCase()} · {rounds.strategies[s].length} rounds · {rounds.engine}</option>)}
          {strategies.length > 1 && <option value="compare">FedAvg vs FedProx (accuracy)</option>}
        </select>
      </div>
      <div className="chart-box">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={data} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
            <CartesianGrid stroke="#2a3340" strokeDasharray="3 3" />
            <XAxis dataKey="round" stroke="#8b97a6" tick={{ fontSize: 12 }} />
            <YAxis stroke="#8b97a6" tick={{ fontSize: 12 }} domain={current === "compare" ? ["auto", "auto"] : [0, 1]} />
            <Tooltip contentStyle={tooltipStyle} formatter={(v) => (typeof v === "number" ? v.toFixed(3) : v)} labelFormatter={(l) => `Round ${l}`} />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            {refs.map(([label, y]) => (
              <ReferenceLine key={label} y={y} stroke="#8b97a6" strokeDasharray="5 4"
                             label={{ value: `${label} ${y.toFixed(3)}`, fill: "#8b97a6", fontSize: 11, position: "insideBottomRight" }} />
            ))}
            {lines}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </>
  );
}

function TrainForm({ isAdmin, status, onDone }) {
  const [form, setForm] = useState({ engine: "simulation", strategy: "both", rounds: 20, clients: 5, local_epochs: 3, mu: 0.05 });
  const [error, setError] = useState(null);
  const [wasRunning, setWasRunning] = useState(false);
  const running = status?.process_running;

  useEffect(() => {
    if (running && !wasRunning) setWasRunning(true);
    if (status && !running && wasRunning) { setWasRunning(false); onDone(); }
  }, [status, running, wasRunning, onDone]);

  const start = async () => {
    setError(null);
    try {
      await api("/api/federated/train", { method: "POST", body: { ...form, rounds: +form.rounds, clients: +form.clients, local_epochs: +form.local_epochs, mu: +form.mu } });
    } catch (e) {
      setError(e.message);
    }
  };
  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value });
  const progress = status?.total_rounds ? (status.current_round / status.total_rounds) * 100 : 0;

  return (
    <div className="pad train">
      <div className="form-grid">
        <label>Engine
          <select value={form.engine} onChange={(e) => setForm({ ...form, engine: e.target.value, strategy: e.target.value === "flower" && form.strategy === "both" ? "fedavg" : form.strategy })}>
            <option value="simulation">simulation (fast)</option>
            <option value="flower">Flower gRPC</option>
          </select>
        </label>
        <label>Strategy
          <select value={form.strategy} onChange={set("strategy")}>
            {form.engine === "simulation" && <option value="both">FedAvg + FedProx</option>}
            <option value="fedavg">FedAvg</option>
            <option value="fedprox">FedProx</option>
          </select>
        </label>
        <label>Zones<select value={form.clients} onChange={set("clients")}>{[3, 4, 5].map((n) => <option key={n}>{n}</option>)}</select></label>
        <label>Rounds<input type="number" min="1" max="200" value={form.rounds} onChange={set("rounds")} /></label>
        <label>Local epochs<input type="number" min="1" max="50" value={form.local_epochs} onChange={set("local_epochs")} /></label>
        <label>FedProx μ<input type="number" min="0" step="0.01" value={form.mu} onChange={set("mu")} /></label>
      </div>
      {running ? (
        <div className="progress">
          <div className="bar"><div className="accent" style={{ width: `${progress}%` }} /></div>
          <span className="muted small">{(status.current_strategy || status.strategy || "").toUpperCase()} · round {status.current_round}/{status.total_rounds}
            {status.last_accuracy != null && ` · accuracy ${pct(status.last_accuracy)}`} · chart updates live</span>
        </div>
      ) : (
        <button className="primary" disabled={!isAdmin} onClick={start}>{isAdmin ? "Start training run" : "Admins only"}</button>
      )}
      {status?.state === "failed" && <div className="alert">Last run failed. {status.error}{status.log_tail && <pre>{status.log_tail}</pre>}</div>}
      {error && <div className="alert">{error}</div>}
    </div>
  );
}

export default function Federated({ isAdmin }) {
  const { data: status } = usePoll("/api/federated/status", 1500);
  const live = status?.process_running;
  const { data: model, reload: reloadModel } = usePoll("/api/federated/model", 10000);
  const { data: rounds, reload: reloadRounds } = usePoll("/api/federated/rounds", live ? 1500 : 10000, [live]);
  const { data: summary, reload: reloadSummary } = usePoll("/api/federated/summary", 10000);

  const deployed = rounds?.strategies?.[rounds?.config?.deployed] || Object.values(rounds?.strategies || {})[0] || [];
  const last = deployed[deployed.length - 1];
  const zones = summary?.zone_accuracy || {};
  const fedRows = Object.entries(summary?.federated || {});

  return (
    <div className="page">
      <div className="kpis-row">
        <Kpi label="Strategy" value={(model?.strategy || "—").toUpperCase()} />
        <Kpi label="Rounds" value={model?.round_number ?? "—"} />
        <Kpi label="Clients" value={model?.participating_clients?.length ?? "—"} />
        <Kpi label="Global accuracy" value={pct(model?.metrics?.accuracy)} />
        <Kpi label="F1" value={model?.metrics?.f1?.toFixed(3) ?? "—"} />
        <Kpi label="AUC" value={model?.metrics?.auc?.toFixed(3) ?? "—"} />
        <Kpi label="Weights exchanged" value={kib(last?.cumulative_comm_bytes)} />
      </div>

      <div className="fl-grid">
        <div className="col">
          <section className="panel"><Convergence rounds={rounds} summary={summary} /></section>

          {summary?.local_only && (
            <section className="panel">
              <div className="panel-head"><h2>Does federation help?</h2><span className="muted small">same campus-wide held-out test set</span></div>
              <table>
                <thead><tr><th>Setting</th><th>Training data</th><th>Accuracy</th><th>F1</th><th>AUC</th><th>Data leaves zone?</th></tr></thead>
                <tbody>
                  {Object.entries(summary.local_only).map(([cid, v]) => (
                    <tr key={cid}><td>Local only · {cid}</td><td>{v.n_train} own events</td><td>{pct(v.global_test.accuracy)}</td><td>{v.global_test.f1.toFixed(3)}</td><td>{v.global_test.auc?.toFixed(3)}</td><td className="ok-text">no</td></tr>
                  ))}
                  <tr className="sub"><td>Local only · mean</td><td /><td>{pct(summary.local_only_mean_accuracy)}</td><td /><td /><td className="ok-text">no</td></tr>
                  {fedRows.map(([s, v]) => (
                    <tr key={s} className="hl"><td><b>Federated {s.toUpperCase()}</b></td><td>all zones · weights only</td><td><b>{pct(v.final.accuracy)}</b></td><td>{v.final.f1.toFixed(3)}</td><td>{v.final.auc?.toFixed(3)}</td><td className="ok-text">no ({kib(v.total_comm_bytes)} of weights)</td></tr>
                  ))}
                  {summary.centralized && (
                    <tr><td>Centralized (upper bound)</td><td>{summary.centralized.n_train} pooled events</td><td>{pct(summary.centralized.global_test.accuracy)}</td><td>{summary.centralized.global_test.f1.toFixed(3)}</td><td>{summary.centralized.global_test.auc?.toFixed(3)}</td><td className="bad-text">yes, all raw data</td></tr>
                  )}
                </tbody>
              </table>
            </section>
          )}

          <section className="panel">
            <details>
              <summary className="panel-head"><h2>Round log · {(rounds?.config?.deployed || "—").toUpperCase()}</h2><span className="muted small">click to expand</span></summary>
              <table>
                <thead><tr><th>Round</th><th>Participating clients</th><th>Accuracy</th><th>F1</th><th>Loss</th><th>This round</th><th>Cumulative</th></tr></thead>
                <tbody>
                  {[...deployed].reverse().map((r) => (
                    <tr key={r.round_number}><td>{r.round_number}</td><td className="small">{r.participating_clients.join(", ")}</td><td>{pct(r.metrics.accuracy)}</td><td>{r.metrics.f1.toFixed(3)}</td><td>{r.metrics.loss.toFixed(3)}</td><td>{kib(r.comm_bytes)}</td><td>{kib(r.cumulative_comm_bytes)}</td></tr>
                  ))}
                </tbody>
              </table>
            </details>
          </section>
        </div>

        <div className="col">
          <section className="panel">
            <div className="panel-head"><h2>Zone clients</h2></div>
            <table>
              <thead><tr><th>Client (zone)</th><th>Samples</th><th>Own data: global model</th><th>Alone</th></tr></thead>
              <tbody>
                {Object.entries(zones).map(([cid, z]) => (
                  <tr key={cid}><td>{cid}</td><td>{z.n}</td><td><b>{pct(z.federated_own)}</b></td><td className="muted">{pct(z.local_only_own)}</td></tr>
                ))}
                {!Object.keys(zones).length && <tr><td colSpan={4} className="muted">Run a simulation to see per-zone results.</td></tr>}
              </tbody>
            </table>
            <p className="pad muted small">
              Each zone trained on its own data. Only model weights ({model?.architecture?.includes("7-16") ? "580 bytes" : "a few hundred bytes"} per transfer) reached the server. No video or event data left the zones.
            </p>
          </section>

          <section className="panel">
            <div className="panel-head"><h2>Model in use</h2><button className="ghost small" onClick={() => { reloadModel(); reloadRounds(); reloadSummary(); }}>Refresh</button></div>
            {model?.loaded ? (
              <div className="kv pad">
                <label>Mode</label><span>federated global model</span>
                <label>Strategy</label><span>{model.strategy?.toUpperCase()} ({model.engine})</span>
                <label>Version</label><span className="mono">{model.model_version}</span>
                <label>Clients</label><span>{(model.participating_clients || []).join(", ")}</span>
                <label>Model</label><span>{model.architecture}</span>
                <label>Threshold</label><span>escalate if P ≥ {model.threshold}</span>
                <label>Trained</label><span>{new Date(model.created_at).toLocaleString()}</span>
              </div>
            ) : <div className="pad muted">No global model yet, so every candidate is escalated until the first run finishes.</div>}
          </section>

          <section className="panel">
            <div className="panel-head"><h2>Run training</h2><span className="muted small">never blocks monitoring</span></div>
            <TrainForm isAdmin={isAdmin} status={status} onDone={() => { reloadModel(); reloadRounds(); reloadSummary(); }} />
          </section>
        </div>
      </div>
    </div>
  );
}
