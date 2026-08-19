"use client"
export default function TraitsTable({ traits }: { traits: Record<string, unknown> }) {
  const keys = Object.keys(traits)
  if (keys.length === 0) return <p className="muted">No traits yet — computed after each checkpoint replay.</p>
  return (
    <table className="tbl">
      <tbody>
        {keys.map((k) => (
          <tr key={k}><td>{k}</td><td>{String(traits[k])}</td></tr>
        ))}
      </tbody>
    </table>
  )
}