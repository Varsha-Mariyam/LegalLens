import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api.js";
import ClauseDetail from "../components/ClauseDetail.jsx";
import ComparisonPanel from "../components/ComparisonPanel.jsx";
import PersonaPanel from "../components/PersonaPanel.jsx";
import ProcessingSteps from "../components/ProcessingSteps.jsx";
import QAPanel from "../components/QAPanel.jsx";
import RiskBadge from "../components/RiskBadge.jsx";
import RiskRail from "../components/RiskRail.jsx";

const TABS = [
  ["clauses", "Clauses"],
  ["overview", "Summary & recommendations"],
  ["comparison", "Reference comparison"],
  ["persona", "Persona summary"],
  ["qa", "Ask LegalLens"],
];

export default function DocumentView() {
  const { id } = useParams();
  const [status, setStatus] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState("");
  const [tab, setTab] = useState("clauses");
  const [selected, setSelected] = useState(null);
  const [filter, setFilter] = useState("ALL");
  const [downloading, setDownloading] = useState(false);

  const loadAnalysis = useCallback(() => api.analysis(id).then((a) => {
    setAnalysis(a);
    setSelected((cur) => cur ?? a.highest_risk[0]?.clause_id ?? a.clauses[0]?.clause_id ?? null);
  }), [id]);

  useEffect(() => {
    let timer;
    let cancelled = false;
    const poll = async () => {
      try {
        const s = await api.status(id);
        if (cancelled) return;
        setStatus(s);
        if (s.status === "completed") await loadAnalysis();
        else if (s.status !== "failed") timer = setTimeout(poll, 1200);
      } catch (e) {
        if (!cancelled) setError(e.message);
      }
    };
    poll();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [id, loadAnalysis]);

  const comparisonByClause = useMemo(() => {
    const m = {};
    analysis?.comparisons.forEach((r) => { if (r.user_clause_number && !m[r.user_clause_number]) m[r.user_clause_number] = r; });
    return m;
  }, [analysis]);

  // On narrow screens the detail panel is below the list: bring it into view when a clause is chosen.
  const selectClause = (clauseId) => {
    setSelected(clauseId);
    if (window.matchMedia("(max-width: 1000px)").matches) {
      requestAnimationFrame(() => document.querySelector(".clause-detail")?.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" }));
    }
  };

  const openClauseNumber = (num) => {
    const c = analysis.clauses.find((x) => x.clause_number === num);
    if (c) { setSelected(c.clause_id); setTab("clauses"); }
  };

  const downloadReport = async () => {
    setDownloading(true);
    setError("");
    try {
      const res = await api.report(id);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `LegalLens_Report_${analysis.document.filename.replace(/\.[^.]+$/, "")}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(e.message);
    } finally {
      setDownloading(false);
    }
  };

  const reprocess = async () => {
    await api.reprocess(id);
    setAnalysis(null);
    setSelected(null);
    setStatus({ status: "uploaded", progress: 0, current_step: "Queued", steps: status.steps });
    const poll = async () => {
      const s = await api.status(id);
      setStatus(s);
      if (s.status === "completed") loadAnalysis();
      else if (s.status !== "failed") setTimeout(poll, 1200);
    };
    setTimeout(poll, 800);
  };

  if (error && !analysis) return <main className="page"><p className="error">{error}</p><Link to="/">Back to documents</Link></main>;
  if (!status) return <main className="page"><p className="muted">Loading…</p></main>;
  if (!analysis) {
    return (
      <main className="page narrow">
        <ProcessingSteps status={status} />
        {status.status === "failed" && <button className="primary" onClick={reprocess}>Try again</button>}
      </main>
    );
  }

  const d = analysis.document;
  const clauses = analysis.clauses;
  const shown = filter === "ALL" ? clauses : clauses.filter((c) => c.risk_level === filter || c.clause_type === filter);
  const selectedClause = clauses.find((c) => c.clause_id === selected);
  const meta = analysis.analysis_meta;

  return (
    <main className="page document">
      <header className="doc-header">
        <div>
          <Link to="/" className="muted small">All documents</Link>
          <h1>{d.filename}</h1>
          <p className="muted">
            {d.detected_type} agreement · {d.page_count} page{d.page_count === 1 ? "" : "s"} · text via {d.extraction_method}
          </p>
          <p className="counts">
            <strong>{d.clause_count} clauses</strong>
            <span className="c-high">{d.risk_counts.HIGH} high risk</span>
            <span className="c-medium">{d.risk_counts.MEDIUM} medium risk</span>
            <span className="c-low">{d.risk_counts.LOW} low risk</span>
          </p>
        </div>
        <div className="doc-actions">
          <button className="primary" onClick={downloadReport} disabled={downloading}>
            {downloading ? "Preparing report…" : "Download PDF report"}
          </button>
          <button className="secondary" onClick={reprocess}>Re-run analysis</button>
        </div>
      </header>
      {error && <p className="error">{error}</p>}

      <nav className="tabs" role="tablist">
        {TABS.map(([k, label]) => (
          <button key={k} role="tab" aria-selected={tab === k} className={tab === k ? "active" : ""} onClick={() => setTab(k)}>
            {label}
          </button>
        ))}
      </nav>

      {tab === "clauses" && (
        <div className="clause-layout">
          <RiskRail clauses={clauses} selectedId={selected} onSelect={selectClause} />
          <div className="clause-list">
            <label className="inline filter">
              Show
              <select value={filter} onChange={(e) => setFilter(e.target.value)}>
                <option value="ALL">All clauses</option>
                <option value="HIGH">High risk</option>
                <option value="MEDIUM">Medium risk</option>
                <option value="LOW">Low risk</option>
                {Object.keys(analysis.category_counts).map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </label>
            <ul>
              {shown.map((c) => (
                <li key={c.clause_id}>
                  <button className={`clause-item ${selected === c.clause_id ? "selected" : ""}`} onClick={() => selectClause(c.clause_id)}>
                    <span className="clause-num">{c.label || ""}</span>
                    <span className="clause-title">
                      {c.title || c.clause_type}
                      <small>{c.clause_type}</small>
                    </span>
                    <RiskBadge level={c.risk_level} />
                  </button>
                </li>
              ))}
            </ul>
          </div>
          <ClauseDetail clause={selectedClause} comparison={selectedClause && comparisonByClause[selectedClause.clause_number]} />
        </div>
      )}

      {tab === "overview" && (
        <div className="overview">
          <section>
            <h2>Overall summary</h2>
            <div className="summary-text">
              {analysis.overall_summary.split("\n").map((l, i) => (l.trim() ? <p key={i}>{l}</p> : null))}
            </div>
            <p className="method">Generated by: {analysis.summary_method}</p>
          </section>
          <section>
            <h2>Recommendations</h2>
            <ol className="recs">
              {analysis.recommendations.map((r, i) => (
                <li key={i} className={`rec-${r.priority.toLowerCase()}`}>
                  {r.priority !== "INFO" && <RiskBadge level={r.priority} />}
                  <span>{r.text}</span>
                  {r.clause_number && <button className="linklike" onClick={() => openClauseNumber(r.clause_number)}>View clause</button>}
                </li>
              ))}
            </ol>
          </section>
          <section>
            <h2>Clause categories</h2>
            <table className="cat-table">
              <tbody>
                {Object.entries(analysis.category_counts).map(([k, v]) => (
                  <tr key={k}><th>{k}</th><td>{v}</td></tr>
                ))}
              </tbody>
            </table>
          </section>
          <section className="how">
            <h2>How this analysis was produced</h2>
            <p className="small">
              Text generation: {meta.generation}. Embeddings: {meta.embedding_backend}. Vector store: {meta.vector_store}.
              Document context cache (CAG): {meta.cache_backend}. Classifier: {meta.classifier?.ml_model ? "TF-IDF + Logistic Regression with keyword and embedding baselines" : "keyword and embedding baselines"}.
            </p>
          </section>
        </div>
      )}

      {tab === "comparison" && <ComparisonPanel documentId={id} onOpenClause={openClauseNumber} />}
      {tab === "persona" && <PersonaPanel documentId={id} />}
      {tab === "qa" && <QAPanel documentId={id} onOpenClause={openClauseNumber} />}
    </main>
  );
}
