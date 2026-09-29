export default function RiskBadge({ level }) {
  const l = (level || "none").toLowerCase();
  return <span className={`risk-badge risk-${l}`}>{level ? level.charAt(0) + level.slice(1).toLowerCase() : "—"}</span>;
}
