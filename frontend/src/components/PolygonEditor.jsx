import { useEffect, useRef, useState } from "react";
import { api } from "../api";

// Draw the restricted region on a real frame from the camera.
// Points are stored in the frame's native pixel coordinates, which is what
// the vision track's ZoneConfig expects.
export default function PolygonEditor({ cameraId, value, onChange }) {
  const [img, setImg] = useState(null); // { url, w, h }
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const svgRef = useRef(null);
  const points = value || [];

  const load = async () => {
    if (!cameraId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api(`/api/cameras/${cameraId}/snapshot`, { raw: true });
      const blob = await res.blob();
      const w = parseInt(res.headers.get("X-Frame-Width") || "640", 10);
      const h = parseInt(res.headers.get("X-Frame-Height") || "480", 10);
      setImg((old) => {
        if (old) URL.revokeObjectURL(old.url);
        return { url: URL.createObjectURL(blob), w, h };
      });
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cameraId]);

  const addPoint = (e) => {
    if (!img) return;
    const rect = svgRef.current.getBoundingClientRect();
    const x = Math.round(((e.clientX - rect.left) / rect.width) * img.w);
    const y = Math.round(((e.clientY - rect.top) / rect.height) * img.h);
    onChange([...points, [x, y]]);
  };

  const defaultRect = () => {
    if (!img) return;
    const { w, h } = img;
    onChange([[w / 4, h / 4], [(3 * w) / 4, h / 4], [(3 * w) / 4, (3 * h) / 4], [w / 4, (3 * h) / 4]].map((p) => p.map(Math.round)));
  };

  if (!cameraId) return <div className="empty">Save the camera first, then draw its restricted zone on a live frame.</div>;

  return (
    <div className="poly">
      <div className="poly-tools">
        <span className="muted">Click on the frame to add corners ({points.length} points{points.length > 0 && points.length < 3 ? ": need at least 3" : ""})</span>
        <span>
          <button type="button" className="ghost small" onClick={() => onChange(points.slice(0, -1))} disabled={!points.length}>Undo</button>
          <button type="button" className="ghost small" onClick={() => onChange([])} disabled={!points.length}>Clear</button>
          <button type="button" className="ghost small" onClick={defaultRect} disabled={!img}>Centre box</button>
          <button type="button" className="ghost small" onClick={load}>{loading ? "Loading…" : "Refresh frame"}</button>
        </span>
      </div>
      {error && <div className="alert">Could not grab a frame: {error}</div>}
      {img && (
        <div className="poly-stage">
          <img src={img.url} alt="camera frame" />
          <svg ref={svgRef} viewBox={`0 0 ${img.w} ${img.h}`} preserveAspectRatio="none" onClick={addPoint}>
            {points.length >= 3 && <polygon points={points.map((p) => p.join(",")).join(" ")} className="zone-fill" />}
            {points.length >= 2 && points.length < 3 && <polyline points={points.map((p) => p.join(",")).join(" ")} className="zone-line" />}
            {points.map(([x, y], i) => (
              <circle key={i} cx={x} cy={y} r={Math.max(img.w / 120, 4)} className="zone-pt" />
            ))}
          </svg>
        </div>
      )}
      <small className="muted">Frame size {img ? `${img.w}×${img.h}` : "—"}. Leave empty to use the centre box automatically.</small>
    </div>
  );
}
