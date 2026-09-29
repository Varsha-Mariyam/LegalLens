export default function ProcessingSteps({ status }) {
  const steps = status.steps || [];
  const current = status.current_step || "";
  const idx = steps.findIndex((s) => current.startsWith(s));
  return (
    <section className="processing" aria-live="polite">
      <h2>{status.status === "failed" ? "Analysis failed" : "Analysing your document"}</h2>
      <div className="progress" role="progressbar" aria-valuenow={status.progress} aria-valuemin={0} aria-valuemax={100}>
        <span style={{ width: `${status.progress || 0}%` }} />
      </div>
      <ol className="steps">
        {steps.map((s, i) => (
          <li key={s} className={i < idx || status.status === "completed" ? "done" : i === idx ? "active" : ""}>
            {i === idx && current !== s ? current : s}
          </li>
        ))}
      </ol>
      {status.error && <p className="error">{status.error}</p>}
    </section>
  );
}
