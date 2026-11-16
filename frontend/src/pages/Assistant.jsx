import { useEffect, useRef, useState } from "react";
import { api } from "../api";

const SUGGESTIONS = [
  "What should an operator do when someone enters a restricted lab?",
  "When should a crowd be escalated?",
  "What are the visitor rules after 8 pm?",
];

// Operator chat over the safety knowledge base (retrieval + local LLM).
export default function Assistant({ compact = false }) {
  const [messages, setMessages] = useState([]);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [messages, busy]);

  const ask = async (question) => {
    const text = (question ?? q).trim();
    if (!text || busy) return;
    setQ("");
    setMessages((m) => [...m, { role: "user", text }]);
    setBusy(true);
    try {
      const res = await api("/api/chat", { method: "POST", body: { question: text } });
      setMessages((m) => [...m, { role: "bot", text: res.answer, source: res.source_section }]);
    } catch (e) {
      setMessages((m) => [...m, { role: "bot", error: true, text: e.message }]);
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className={`panel chat ${compact ? "compact" : "full"}`}>
      <div className="panel-head">
        <h2>Safety assistant</h2>
        <span className="muted small">answers from the safety knowledge base</span>
      </div>
      <div className="msgs">
        {!messages.length && (
          <div className="suggest">
            {SUGGESTIONS.map((s) => <button key={s} onClick={() => ask(s)}>{s}</button>)}
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role} ${m.error ? "err" : ""}`}>
            <p>{m.text}</p>
            {m.source && <div className="source">Source: {m.source}</div>}
          </div>
        ))}
        {busy && <div className="msg bot"><p className="generating">Searching the knowledge base…</p></div>}
        <div ref={endRef} />
      </div>
      <form className="ask" onSubmit={(e) => { e.preventDefault(); ask(); }}>
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask about a safety procedure…" disabled={busy} />
        <button className="primary" disabled={busy || !q.trim()}>Ask</button>
      </form>
    </section>
  );
}
