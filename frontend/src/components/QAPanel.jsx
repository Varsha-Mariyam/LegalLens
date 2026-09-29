import { useEffect, useRef, useState } from "react";
import { api } from "../api.js";

const SUGGESTIONS = ["What is the notice period?", "Who can terminate?", "Is that risky?", "How much is the payment?"];

export default function QAPanel({ documentId, onOpenClause }) {
  const [history, setHistory] = useState([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const endRef = useRef();

  useEffect(() => { api.qaHistory(documentId).then(setHistory).catch((e) => setError(e.message)); }, [documentId]);
  useEffect(() => { endRef.current?.scrollIntoView({ block: "nearest" }); }, [history, busy]);

  const ask = async (q) => {
    const text = (q ?? question).trim();
    if (!text) return;
    setBusy(true);
    setError("");
    setQuestion("");
    try {
      const turn = await api.ask(documentId, text);
      setHistory((h) => [...h, turn]);
    } catch (e) {
      setError(e.message);
      setQuestion(text);
    } finally {
      setBusy(false);
    }
  };

  const clear = async () => {
    if (!window.confirm("Clear this document's conversation?")) return;
    await api.clearQa(documentId);
    setHistory([]);
  };

  return (
    <section className="qa">
      <div className="qa-log" aria-live="polite">
        {history.length === 0 && (
          <div className="qa-empty">
            <p>Ask about anything in this document. Follow-up questions like “Is that risky?” use the previous answer.</p>
            <div className="chips">
              {SUGGESTIONS.map((s) => <button key={s} className="chip" onClick={() => ask(s)} disabled={busy}>{s}</button>)}
            </div>
          </div>
        )}
        {history.map((t) => (
          <div key={t.conversation_id} className="qa-turn">
            <p className="q">{t.question}</p>
            <div className="a">
              {t.answer.split("\n").map((l, i) => (l.trim() ? <p key={i}>{l}</p> : null))}
              <p className="sources">
                {t.sources?.filter((s) => s.clause_number).map((s) => (
                  <button key={s.clause_number} className="chip small" onClick={() => onOpenClause(s.clause_number)}>
                    Clause {s.clause_label ?? s.clause_number}{s.title ? ` · ${s.title}` : ""}
                  </button>
                ))}
                {t.sources?.filter((s, i, all) => s.reference && all.findIndex((x) => x.reference === s.reference) === i).map((s, i) => (
                  <span key={i} className="ref-chip" title={s.citation || ""}>{s.reference}</span>
                ))}
              </p>
              <p className="method">{t.method}</p>
            </div>
          </div>
        ))}
        {busy && <p className="muted">Looking through the document…</p>}
        <div ref={endRef} />
      </div>
      {error && <p className="error">{error}</p>}
      <form className="qa-input" onSubmit={(e) => { e.preventDefault(); ask(); }}>
        <input value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="Ask a question about this document"
               aria-label="Question" maxLength={1000} />
        <button className="primary" disabled={busy || !question.trim()}>Send</button>
      </form>
      {history.length > 0 && <button className="linklike muted" onClick={clear}>Clear conversation</button>}
    </section>
  );
}
