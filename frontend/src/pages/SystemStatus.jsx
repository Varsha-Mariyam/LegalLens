import { useEffect, useState } from "react";
import { api } from "../api.js";

const fmt = (x) => (typeof x === "number" ? x.toFixed(3) : "–");
const ci = (c) => (Array.isArray(c) ? `${c[0].toFixed(3)}–${c[1].toFixed(3)}` : "–");

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
          <Row name="Clause classifier" ok={["ensemble", "transformer"].includes(s.classifier.primary) || s.classifier.ml_model}>
            {s.classifier.primary === "ensemble"
              ? `Ensemble: fine-tuned ${s.classifier.transformer.base_model} (weight ${s.classifier.transformer.ensemble_weight}) + TF-IDF Logistic Regression, on ${s.classifier.transformer.device}; fallback: Logistic Regression alone`
              : s.classifier.primary === "transformer"
                ? `Fine-tuned ${s.classifier.transformer.base_model} on ${s.classifier.transformer.device} (${s.classifier.transformer.path}); fallback: ${s.classifier.ml_model ? "TF-IDF + Logistic Regression" : "keyword/embedding baselines"}`
                : s.classifier.ml_model ? `TF-IDF + Logistic Regression (${s.classifier.model_path}) — Transformer not in use: ${s.classifier.transformer?.error || "unavailable"}`
                  : (s.classifier.load_error || "No trained model — run scripts/train_classifier.py")}
          </Row>
          <Row name="Risk detector" ok={Boolean(s.risk_detector?.detector)}>
            {s.risk_detector?.detector
              ? `Rule signals + learned detector: ${{ ensemble: `fine-tuned ${s.risk_detector.base_model} + TF-IDF ensemble (weight ${s.risk_detector.ensemble_weight})`, transformer: `fine-tuned ${s.risk_detector.base_model}`, logreg: "TF-IDF + Logistic Regression" }[s.risk_detector.detector]} — 12 CUAD clause types`
              : `Rule signals only — learned detector not in use: ${s.risk_detector?.error || "disabled or not trained"}`}
          </Row>
          <Row name="Q&A retrieval" ok={s.qa_retrieval?.mode === "hybrid" && s.qa_retrieval?.calibrated_not_found}>
            {s.qa_retrieval?.mode === "hybrid"
              ? `Hybrid BM25 + dense (dense weight ${s.qa_retrieval.alpha}${s.qa_retrieval.fusion === "rrf" ? ", reciprocal rank fusion" : ""}); "not found" ${s.qa_retrieval.calibrated_not_found ? "calibrated on validation contracts" : "uncalibrated rule"}`
              : `${s.qa_retrieval?.mode} ranking`}
          </Row>
        </tbody>
      </table>
      <button className="secondary" onClick={reindex} disabled={busy}>{busy ? "Rebuilding…" : "Rebuild knowledge index"}</button>

      {m?.classifier && (
        <section>
          <h2>Classifier evaluation (held-out test {m.dataset === "contract_split" ? "contracts" : "set"})</h2>
          <div className="table-wrap">
            <table className="metrics">
              <thead><tr><th>Model</th><th>Accuracy</th><th>Macro precision</th><th>Macro recall</th><th>Macro F1</th><th>95% CI (macro F1)</th><th>Gold-label accuracy</th><th>Gold-label macro F1</th></tr></thead>
              <tbody>
                <tr>
                  <td className="best">{m.model_name} ({m.base_model})</td>
                  <td>{fmt(m.classifier.accuracy)}</td><td>{fmt(m.classifier.macro_precision)}</td><td>{fmt(m.classifier.macro_recall)}</td>
                  <td>{fmt(m.classifier.macro_f1)}</td><td>{ci(m.classifier.macro_f1_ci95)}</td>
                  <td>{fmt(m.classifier.gold_only?.accuracy)}</td><td>{fmt(m.classifier.gold_only?.macro_f1)}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div className="table-wrap">
            <table className="metrics">
              <thead><tr><th>Category</th><th>Precision</th><th>Recall</th><th>F1</th><th>Test clauses</th></tr></thead>
              <tbody>
                {Object.entries(m.classifier.per_class || {}).map(([k, v]) => (
                  <tr key={k}><td>{k}</td><td>{fmt(v.precision)}</td><td>{fmt(v.recall)}</td><td>{fmt(v["f1-score"])}</td><td>{v.support}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
          {m.classifier.confusion_matrix && (
            <div className="table-wrap">
              <table className="metrics">
                <thead><tr><th>Actual ↓ / predicted →</th>{m.classifier.confusion_matrix.labels.map((l) => <th key={l}>{l}</th>)}</tr></thead>
                <tbody>
                  {m.classifier.confusion_matrix.matrix.map((row, i) => (
                    <tr key={m.classifier.confusion_matrix.labels[i]}>
                      <td>{m.classifier.confusion_matrix.labels[i]}</td>
                      {row.map((n, j) => <td key={j} className={i === j ? "best" : ""}>{n}</td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <p className="muted small metrics-note">
            {m.dataset === "contract_split"
              ? `${m.test_size} test clauses from ${m.test_contracts} contracts never seen in training (split by contract); metrics are computed from the deployed model's own predictions. Gold labels = CUAD expert annotations (heading-derived weak labels excluded). Full report, including the comparison with other methods: ${m.report}.`
              : "Gold labels = project-authored seed clauses and CUAD expert annotations (heading-derived weak labels excluded)."}
          </p>

          {m.risk_detector && (
            <>
              <h2>Risk detector evaluation (12 CUAD clause types)</h2>
              <div className="table-wrap">
                <table className="metrics">
                  <thead><tr><th>Model</th><th>Micro precision</th><th>Micro recall</th><th>Micro F1</th><th>95% CI (micro F1)</th><th>Macro F1</th><th>Macro ROC-AUC</th><th>Macro average precision</th></tr></thead>
                  <tbody>
                    <tr>
                      <td className="best">{m.model_name} ({m.base_model})</td>
                      <td>{fmt(m.risk_detector.micro_precision)}</td><td>{fmt(m.risk_detector.micro_recall)}</td><td>{fmt(m.risk_detector.micro_f1)}</td>
                      <td>{ci(m.risk_detector.micro_f1_ci95)}</td><td>{fmt(m.risk_detector.macro_f1)}</td>
                      <td>{fmt(m.risk_detector.macro_roc_auc)}</td><td>{fmt(m.risk_detector.macro_average_precision)}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </>
          )}

          {m.retrieval && (
            <>
              <h2>Q&amp;A retrieval evaluation (CUAD questions, test contracts)</h2>
              <div className="table-wrap">
                <table className="metrics">
                  <thead><tr><th>Ranker</th><th>Hit@1</th><th>Hit@3</th><th>MRR@10</th><th>nDCG@5</th></tr></thead>
                  <tbody>
                    {Object.entries(m.retrieval).map(([k, v]) => (
                      <tr key={k}><td>{k}</td><td>{fmt(v["hit@1"])}</td><td>{fmt(v["hit@3"])}</td><td>{fmt(v["mrr@10"])}</td><td>{fmt(v["ndcg@5"])}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
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
