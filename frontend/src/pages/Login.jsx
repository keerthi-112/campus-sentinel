import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, auth } from "../api";

export default function Login({ onLogin }) {
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const navigate = useNavigate();

  const submit = async (e) => {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const { token, user } = await api("/api/auth/login", { method: "POST", body: { username, password } });
      auth.save(token, user);
      onLogin(user);
      navigate("/");
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="login-wrap">
      <form className="panel login" onSubmit={submit}>
        <span className="logo big" aria-hidden>◉</span>
        <h1>Campus Sentinel</h1>
        <p className="muted">Privacy-preserving campus safety: operator sign-in</p>
        <label>Username<input value={username} onChange={(e) => setUsername(e.target.value)} autoFocus /></label>
        <label>Password<input type="password" value={password} onChange={(e) => setPassword(e.target.value)} /></label>
        {error && <div className="alert">{error}</div>}
        <button className="primary" disabled={busy}>{busy ? "Signing in…" : "Sign in"}</button>
        <small className="muted">Default: admin / admin123</small>
      </form>
    </div>
  );
}
