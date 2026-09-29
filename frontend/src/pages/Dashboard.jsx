import { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api.js";
import RiskBadge from "../components/RiskBadge.jsx";

const TYPES = [
  ["auto", "Detect automatically"],
  ["employment", "Employment agreement / offer letter"],
  ["rental", "Rental / lease agreement"],
  ["nda", "Non-disclosure agreement"],
  ["service", "Service / business agreement"],
];
const ACCEPT = ".pdf,.docx,.txt,.png,.jpg,.jpeg,.tif,.tiff,.bmp";

function RiskSummary({ counts }) {
  if (!counts) return null;
  const total = counts.HIGH + counts.MEDIUM + counts.LOW;
  if (!total) return null;
  return (
    <div className="risk-bar" aria-label={`${counts.HIGH} high, ${counts.MEDIUM} medium, ${counts.LOW} low potential risk`}>
      {["HIGH", "MEDIUM", "LOW"].map((l) => counts[l] > 0 && (
        <span key={l} className={`seg risk-${l.toLowerCase()}`} style={{ flexGrow: counts[l] }} />
      ))}
    </div>
  );
}

export default function Dashboard() {
  const [docs, setDocs] = useState(null);
  const [file, setFile] = useState(null);
  const [type, setType] = useState("auto");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef();
  const navigate = useNavigate();

  const load = () => api.listDocuments().then(setDocs).catch((e) => setError(e.message));
  useEffect(() => { load(); }, []);
  useEffect(() => {
    if (!docs?.some((d) => d.status === "processing" || d.status === "uploaded")) return;
    const t = setTimeout(load, 2000);
    return () => clearTimeout(t);
  }, [docs]);

  const upload = async (e) => {
    e.preventDefault();
    if (!file) return;
    setBusy(true);
    setError("");
    try {
      const doc = await api.upload(file, type);
      navigate(`/documents/${doc.document_id}`);
    } catch (err) {
      setError(err.message);
      setBusy(false);
    }
  };

  const remove = async (d) => {
    if (!window.confirm(`Delete "${d.filename}" and its analysis?`)) return;
    try { await api.deleteDocument(d.document_id); load(); } catch (err) { setError(err.message); }
  };

  return (
    <main className="page dashboard">
      <section className="upload-panel">
        <h1>Analyse a document</h1>
        <form onSubmit={upload}>
          <div
            className={`dropzone ${dragging ? "dragging" : ""} ${file ? "has-file" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => { e.preventDefault(); setDragging(false); if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]); }}
            onClick={() => inputRef.current.click()}
            onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current.click()}
            role="button"
            tabIndex={0}
          >
            <input ref={inputRef} type="file" accept={ACCEPT} hidden onChange={(e) => setFile(e.target.files[0] || null)} />
            {file ? (
              <><strong>{file.name}</strong><span>{(file.size / 1024).toFixed(0)} KB · click to choose a different file</span></>
            ) : (
              <><strong>Drop a file here or click to choose</strong><span>PDF, Word (.docx), text, or a scanned image</span></>
            )}
          </div>
          <label className="inline">
            Document type
            <select value={type} onChange={(e) => setType(e.target.value)}>
              {TYPES.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
            </select>
          </label>
          {error && <p className="error" role="alert">{error}</p>}
          <button className="primary" disabled={!file || busy}>{busy ? "Uploading…" : "Upload and analyse"}</button>
        </form>
        <p className="muted small">
          Try the synthetic samples in <code>data/raw_documents/samples/</code> if you don't have a document to hand.
        </p>
      </section>

      <section className="doc-list">
        <h2>Your documents</h2>
        {docs === null && <p className="muted">Loading…</p>}
        {docs?.length === 0 && <p className="muted">Nothing analysed yet. Upload a document to see its clauses and risks here.</p>}
        <ul>
          {docs?.map((d) => (
            <li key={d.document_id} className="doc-row">
              <Link to={`/documents/${d.document_id}`} className="doc-link">
                <span className="doc-name">{d.filename}</span>
                <span className="doc-meta">
                  {d.detected_type || "type pending"} · {new Date(d.upload_date).toLocaleDateString()} ·{" "}
                  {d.status === "completed" ? `${d.clause_count} clauses` : d.status === "failed" ? "failed" : `${d.current_step || "queued"} (${d.progress}%)`}
                </span>
                <RiskSummary counts={d.status === "completed" ? d.risk_counts : null} />
              </Link>
              {d.status === "completed" && d.risk_counts.HIGH > 0 && <RiskBadge level="HIGH" />}
              <button className="linklike danger" onClick={() => remove(d)} aria-label={`Delete ${d.filename}`}>Delete</button>
            </li>
          ))}
        </ul>
      </section>
    </main>
  );
}
