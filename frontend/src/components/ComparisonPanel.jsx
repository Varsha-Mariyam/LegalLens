import { Fragment, useEffect, useState } from "react";
import { api } from "../api.js";

const STATUS_TEXT = { equivalent: "Matches", differs: "Differs", missing: "Missing", additional: "Not in reference" };

export default function ComparisonPanel({ documentId, onOpenClause }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [open, setOpen] = useState(null);
  useEffect(() => { api.comparison(documentId).then(setData).catch((e) => setError(e.message)); }, [documentId]);
  if (error) return <p className="error">{error}</p>;
  if (!data) return <p className="muted">Loading comparison…</p>;
  return (
    <section className="comparison">
      <p className="notice">
        Compared with <strong>{data.reference}</strong>. {data.reference_label}
      </p>
      <p className="muted">
        {data.summary.equivalent} matching · {data.summary.differs} different · {data.summary.missing} missing from this document ·{" "}
        {data.summary.additional} not in the reference
      </p>
      {data.rows.length === 0 && <p className="muted">No reference template exists for this document type.</p>}
      <div className="table-wrap">
        <table className="compare-table">
          <thead>
            <tr><th>Clause</th><th>Reference</th><th>This document</th><th>Status</th></tr>
          </thead>
          <tbody>
            {data.rows.map((r) => (
              <Fragment key={r.id}>
                <tr className={`status-${r.status}`} onClick={() => setOpen(open === r.id ? null : r.id)}>
                  <td>
                    <button className="linklike" aria-expanded={open === r.id}>{r.category}</button>
                    <div className="muted small">{r.standard_title || r.user_title}</div>
                  </td>
                  <td>{r.standard_value}</td>
                  <td>{r.user_value}</td>
                  <td><span className={`status-tag ${r.status}`}>{STATUS_TEXT[r.status]}</span></td>
                </tr>
                {open === r.id && (
                  <tr className="compare-detail">
                    <td colSpan={4}>
                      <p><strong>Difference:</strong> {r.difference}</p>
                      <div className="side-by-side">
                        <div>
                          <h5>Reference template</h5>
                          <blockquote className="reference">{r.standard_text || "—"}</blockquote>
                        </div>
                        <div>
                          <h5>This document{r.user_clause_number ? `, clause ${r.user_clause_label ?? r.user_clause_number}` : ""}</h5>
                          <blockquote className="original">{r.user_text || "No matching clause found."}</blockquote>
                          {r.user_clause_number && (
                            <button className="linklike" onClick={() => onOpenClause(r.user_clause_number)}>Open this clause</button>
                          )}
                        </div>
                      </div>
                      <p className="method">Similarity {r.similarity?.toFixed?.(2) ?? "—"} · {r.method}</p>
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
