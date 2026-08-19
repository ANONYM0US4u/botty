"use client"
import type { LeaderboardRow } from "@/lib/api"

export default function LeaderboardTable({ rows }: { rows: LeaderboardRow[] }) {
  if (rows.length === 0) return <p className="muted">No checkpoints yet — start a run first.</p>
  return (
    <table className="tbl">
      <thead>
        <tr><th>#</th><th>Checkpoint</th><th>Sharpe</th><th>Win rate</th><th>Δ Equity</th></tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={r.path}>
            <td>{i + 1}</td>
            <td>{r.path.split(/[\\/]/).pop()}</td>
            <td>{r.sharpe.toFixed(3)}</td>
            <td>{r.win_rate.toFixed(3)}</td>
            <td>{r.mean_reward.toFixed(0)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}