import { useState } from "react";
import { api } from "../api";
import { usePoll } from "../hooks";
import PolygonEditor from "../components/PolygonEditor";

const EMPTY = {
  camera_name: "", zone_name: "", source_type: "file", source: "", restricted_region: [],
  tier: 2, dwell_threshold: 4, confidence_threshold: 0.4, crowd_count: 5, loop_video: true,
};

export default function Cameras() {
  const { data: cameras, reload } = usePoll("/api/cameras", 4000);
  const { data: videos, reload: reloadVideos } = usePoll("/api/cameras/videos/list", 0);
  const [form, setForm] = useState(null); // null = closed, {} = editing
  const [error, setError] = useState(null);
  const [msg, setMsg] = useState(null);
  const [uploading, setUploading] = useState(false);

  const edit = (cam) => {
    setError(null);
    setMsg(null);
    setForm(cam ? { ...cam } : { ...EMPTY });
  };
  const set = (k, v) => setForm((f) => ({ ...f, [k]: v }));

  const save = async (e) => {
    e?.preventDefault();
    setError(null);
    const body = {
      camera_name: form.camera_name, zone_name: form.zone_name, source_type: form.source_type,
      source: String(form.source), restricted_region: form.restricted_region,
      tier: Number(form.tier), dwell_threshold: Number(form.dwell_threshold),
      confidence_threshold: Number(form.confidence_threshold), crowd_count: Number(form.crowd_count),
      loop_video: !!form.loop_video,
    };
    try {
      const saved = form.id
        ? await api(`/api/cameras/${form.id}`, { method: "PUT", body })
        : await api("/api/cameras", { method: "POST", body });
      setForm({ ...saved });
      setMsg(form.id ? "Saved. A running camera restarts with the new zone." : "Camera created. Now draw its restricted zone below.");
      reload();
    } catch (err) {
      setError(err.message);
    }
  };

  const remove = async () => {
    if (!form?.id) return;
    try {
      await api(`/api/cameras/${form.id}`, { method: "DELETE" });
      setForm(null);
      reload();
    } catch (err) {
      setError(err.message);
    }
  };

  const upload = async (file) => {
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await api("/api/cameras/upload-video", { method: "POST", form: fd });
      set("source", res.path);
      set("source_type", "file");
      reloadVideos();
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h2>Cameras &amp; zones</h2>
          <span className="muted small">Each camera has a named restricted zone drawn on its own frame, a risk tier and a dwell timer.</span>
        </div>
        <button className="primary" onClick={() => edit(null)}>+ Add camera</button>
      </div>

      <div className={`split ${form ? "with-detail wide" : ""}`}>
        <div className="panel table-wrap">
          <table>
            <thead><tr><th>Camera</th><th>Zone</th><th>Source</th><th>Tier</th><th>Dwell</th><th>Region</th><th>State</th></tr></thead>
            <tbody>
              {(cameras || []).map((c) => (
                <tr key={c.id} className={form?.id === c.id ? "on" : ""} onClick={() => edit(c)}>
                  <td><b>{c.camera_name}</b></td>
                  <td>{c.zone_name}</td>
                  <td className="small mono">{c.source_type}: {String(c.source).split("/").pop()}</td>
                  <td>{c.tier}</td>
                  <td>{c.dwell_threshold}s</td>
                  <td>{c.restricted_region?.length ? `${c.restricted_region.length}-pt polygon` : "centre box"}</td>
                  <td><span className={`dot ${c.active_state ? "ok" : ""}`} /> {c.runtime?.state || (c.active_state ? "running" : "stopped")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {form && (
          <form className="panel cam-form pad" onSubmit={save}>
            <div className="detail-head">
              <h3>{form.id ? `Edit ${form.camera_name}` : "New camera"}</h3>
              <button type="button" className="ghost small" onClick={() => setForm(null)}>✕</button>
            </div>
            <div className="grid2">
              <label>Camera name<input required value={form.camera_name} onChange={(e) => set("camera_name", e.target.value)} placeholder="Gate 1" /></label>
              <label>Zone name<input required value={form.zone_name} onChange={(e) => set("zone_name", e.target.value)} placeholder="Lab 2 restricted area" /></label>
              <label>Source type
                <select value={form.source_type} onChange={(e) => set("source_type", e.target.value)}>
                  <option value="file">video file</option>
                  <option value="webcam">webcam</option>
                  <option value="rtsp">RTSP stream</option>
                </select>
              </label>
              <label>Source
                <input required value={form.source} onChange={(e) => set("source", e.target.value)}
                  placeholder={form.source_type === "webcam" ? "0" : form.source_type === "rtsp" ? "rtsp://user:pass@ip:554/stream" : "/path/to/clip.mp4"} />
              </label>
            </div>
            {form.source_type === "file" && (
              <div className="upload">
                <label className="filebtn">{uploading ? "Uploading…" : "Upload a clip"}
                  <input type="file" accept="video/*" onChange={(e) => upload(e.target.files?.[0])} hidden />
                </label>
                {!!videos?.length && (
                  <select value="" onChange={(e) => e.target.value && set("source", e.target.value)}>
                    <option value="">…or pick an uploaded clip</option>
                    {videos.map((v) => <option key={v.path} value={v.path}>{v.filename}</option>)}
                  </select>
                )}
                <label className="inline"><input type="checkbox" checked={!!form.loop_video} onChange={(e) => set("loop_video", e.target.checked)} /> loop</label>
              </div>
            )}
            <div className="grid4">
              <label>Risk tier
                <select value={form.tier} onChange={(e) => set("tier", e.target.value)}>
                  <option value={1}>1 · low</option><option value={2}>2 · medium</option><option value={3}>3 · high</option>
                </select>
              </label>
              <label>Dwell (s)<input type="number" step="0.5" min="0.5" value={form.dwell_threshold} onChange={(e) => set("dwell_threshold", e.target.value)} /></label>
              <label>Min confidence<input type="number" step="0.05" min="0.05" max="0.95" value={form.confidence_threshold} onChange={(e) => set("confidence_threshold", e.target.value)} /></label>
              <label>Crowd ≥ people<input type="number" min="2" value={form.crowd_count} onChange={(e) => set("crowd_count", e.target.value)} /></label>
            </div>

            <h4>Restricted region</h4>
            <PolygonEditor cameraId={form.id} value={form.restricted_region} onChange={(pts) => set("restricted_region", pts)} />

            {error && <div className="alert">{error}</div>}
            {msg && <div className="note">{msg}</div>}
            <div className="actions">
              <button type="submit" className="primary">{form.id ? "Save changes" : "Create camera"}</button>
              {form.id && <button type="button" className="ghost danger-text" onClick={remove}>Delete</button>}
            </div>
          </form>
        )}
      </div>
    </div>
  );
}
