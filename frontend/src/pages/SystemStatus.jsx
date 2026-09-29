import { useEffect, useState } from "react";
import { api } from "../api.js";

function Row({ name, ok, children }) {
  return (
    <tr>
      <th>{name}</th>
      <td><span className={`status-dot ${ok ? "ok" : "warn"}`} aria-hidden="true" />{children}</td>
    </tr>
  );
}

export default function SystemStatus() {
  const [s, setS] = useState(null);
  const [sources, setSources] = useState([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const load = () => Promise.all([api.systemStatus(), api.knowledge()]).then(([a, b]) => { setS(a); setSources(b); })
    .catch((e) => setError(e.message));
  useEffect(() => { load(); }, []);

  const reindex = async () => {
    setBusy(true);
    try { await api.reindex(); await load(); } catch (e) { setError(e.message); } finally { setBusy(false); }
  };

  if (error) return <main className="page"><p className="error">{error}</p></main>;
  if (!s) return <main className="page"><p className="muted">Loading…</p></main>;
  const m = s.classifier_metrics;
  return (
    <main className="page system">
      <h1>System components</h1>
      <table className="status-table">
        <tbody>
          <Row name="Language model" ok={s.llm.available}>
            {s.llm.available ? `${s.llm.provider} · ${s.llm.model}` : "Not configured — explanations, persona summaries and answers use the rule-based/extractive mode. Set an API key in .env to enable an LLM."}
          </Row>
          <Row name="Embeddings" ok={!s.embedding.fallback_reason}>
            {s.embedding.backend} ({s.embedding.dim} dimensions){s.embedding.fallback_reason ? ` — fallback in use: ${s.embedding.fallback_reason}` : ""}
          </Row>
          <Row name="Vector database (RAG)" ok={Boolean(s.rag_index)}>
            {s.rag_index ? `${s.rag_index.vector_store}: ${s.rag_index.knowledge_chunks} knowledge chunks, ${s.rag_index.standard_clauses} reference clauses` : "Not built yet"}
          </Row>
          <Row name="Document cache (CAG)" ok>
            {s.cache.backend} · {s.cache.context_hits ?? 0} context hits, {s.cache.llm_hits ?? 0} cached LLM responses
          </Row>
          <Row name="OCR" ok={s.ocr.available}>{s.ocr.available ? `Tesseract ${s.ocr.version}` : `Unavailable: ${s.ocr.error}`}</Row>
          <Row name="Clause classifier" ok={s.classifier.ml_model}>
            {s.classifier.ml_model ? `Trained model at ${s.classifier.model_path}` : (s.classifier.load_error || "No trained model — run scripts/train_classifier.py")}
          </Row>
        </tbody>
      </table>
      <button className="secondary" onClick={reindex} disabled={busy}>{busy ? "Rebuilding…" : "Rebuild knowledge index"}</button>

      {m && (
        <section>
          <h2>Classifier evaluation (held-out test set)</h2>
          <div className="table-wrap">
            <table className="metrics">
              <thead><tr><th>Method</th><th>Accuracy</th><th>Macro F1</th><th>Gold-label accuracy</th><th>Gold-label macro F1</th></tr></thead>
              <tbody>
                {Object.entries(m.methods || {}).map(([k, v]) => (
                  <tr key={k}>
                    <td>{k}</td><td>{v.all?.accuracy?.toFixed(3)}</td><td>{v.all?.macro_f1?.toFixed(3)}</td>
                    <td>{v.gold_only?.accuracy?.toFixed(3)}</td><td>{v.gold_only?.macro_f1?.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="muted small">Gold labels = project-authored seed clauses and CUAD expert annotations (heading-derived weak labels excluded).</p>
        </section>
      )}

      <section>
        <h2>Reference knowledge base</h2>
        <p className="muted small">Educational summaries prepared for this project. Verify against the cited primary sources.</p>
        <ul className="sources-list">
          {sources.map((src) => (
            <li key={src.file}><strong>{src.title}</strong> <span className="muted">({src.file})</span><br /><span className="small">{src.sources}</span></li>
          ))}
        </ul>
      </section>
    </main>
  );
}
