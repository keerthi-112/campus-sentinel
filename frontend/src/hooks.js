import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";

// Poll an API path every `ms` milliseconds. Returns { data, error, loading, reload }.
export function usePoll(path, ms = 3000, deps = []) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const alive = useRef(true);

  const load = useCallback(async () => {
    if (!path) return;
    try {
      const d = await api(path);
      if (alive.current) {
        setData(d);
        setError(null);
      }
    } catch (e) {
      if (alive.current) setError(e.message);
    } finally {
      if (alive.current) setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, ...deps]);

  useEffect(() => {
    alive.current = true;
    load();
    if (!ms) return () => { alive.current = false; };
    const id = setInterval(load, ms);
    return () => {
      alive.current = false;
      clearInterval(id);
    };
  }, [load, ms]);

  return { data, error, loading, reload: load };
}

export const fmtTime = (iso) => {
  if (!iso) return "—";
  const d = new Date(iso);
  return isNaN(d) ? iso : d.toLocaleString();
};

export const pct = (v) => (v == null || isNaN(v) ? "—" : `${(v * 100).toFixed(1)}%`);
