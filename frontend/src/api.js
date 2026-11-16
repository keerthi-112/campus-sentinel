// Tiny fetch wrapper: adds the Bearer token, parses JSON, logs out on 401.

const TOKEN_KEY = "sentinel_token";
const USER_KEY = "sentinel_user";

export const auth = {
  get token() {
    try { return localStorage.getItem(TOKEN_KEY); } catch { return null; }
  },
  get user() {
    try { return JSON.parse(localStorage.getItem(USER_KEY) || "null"); } catch { return null; }
  },
  save(token, user) {
    try {
      localStorage.setItem(TOKEN_KEY, token);
      localStorage.setItem(USER_KEY, JSON.stringify(user));
    } catch { /* storage unavailable */ }
  },
  clear() {
    try {
      localStorage.removeItem(TOKEN_KEY);
      localStorage.removeItem(USER_KEY);
    } catch { /* ignore */ }
  },
};

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

export async function api(path, { method = "GET", body, raw = false, form } = {}) {
  const headers = {};
  if (auth.token) headers.Authorization = `Bearer ${auth.token}`;
  let payload;
  if (form) payload = form;
  else if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    payload = JSON.stringify(body);
  }
  const res = await fetch(path, { method, headers, body: payload });
  if (res.status === 401 && !path.endsWith("/auth/login")) {
    auth.clear();
    window.location.assign("/login");
    throw new ApiError(401, "Session expired");
  }
  if (raw) {
    if (!res.ok) throw new ApiError(res.status, (await res.text()) || res.statusText);
    return res;
  }
  if (res.status === 204) return null;
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = data?.detail;
    const msg = Array.isArray(detail) ? detail.map((d) => d.msg).join("; ") : detail || res.statusText;
    throw new ApiError(res.status, msg);
  }
  return data;
}

export const streamUrl = (cameraId) =>
  `/api/cameras/${cameraId}/stream?token=${encodeURIComponent(auth.token || "")}&t=${Date.now()}`;
