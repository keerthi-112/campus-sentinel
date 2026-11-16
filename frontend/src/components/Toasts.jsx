import { useEffect } from "react";
import { EVENT_LABEL } from "./IncidentDetail";

// Pops up for every new escalated incident; click to open it on the Monitor tab.
export default function Toasts({ toasts, setToasts, onOpen }) {
  useEffect(() => {
    if (!toasts.length) return;
    const id = setTimeout(() => setToasts((t) => t.slice(1)), 8000);
    return () => clearTimeout(id);
  }, [toasts, setToasts]);

  return (
    <div className="toasts">
      {toasts.map(({ key, inc }) => (
        <button key={key} className={`toast sev-${inc.severity}`}
                onClick={() => { onOpen(inc.id); setToasts((t) => t.filter((x) => x.key !== key)); }}>
          <b>New incident · {EVENT_LABEL[inc.event_type] || inc.event_type}</b>
          <span>{inc.zone_name} · {inc.camera_name} · conf {inc.confidence.toFixed(2)}</span>
        </button>
      ))}
    </div>
  );
}
