// The document drawn as a vertical strip: one segment per clause, height proportional to clause length,
// colour = potential risk level. Clicking a segment selects that clause.
export default function RiskRail({ clauses, selectedId, onSelect, orientation = "vertical" }) {
  const lengths = clauses.map((c) => Math.max(1, (c.preview || c.original_text || "").length));
  return (
    <div className={`risk-rail ${orientation}`} role="list" aria-label="Clauses by position and potential risk">
      {clauses.map((c, i) => (
        <button
          key={c.clause_id}
          role="listitem"
          className={`seg risk-${(c.risk_level || "none").toLowerCase()} ${selectedId === c.clause_id ? "selected" : ""}`}
          style={{ flexGrow: Math.sqrt(lengths[i]) }}
          onClick={onSelect ? () => onSelect(c.clause_id) : undefined}
          tabIndex={onSelect ? 0 : -1}
          title={`Clause ${c.label || c.clause_number}: ${c.title || c.clause_type} — ${c.risk_level || "not assessed"} risk`}
          aria-label={`Clause ${c.label || c.clause_number}, ${c.title || c.clause_type}, ${c.risk_level || "not assessed"} risk`}
        />
      ))}
    </div>
  );
}
