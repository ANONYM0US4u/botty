"use client"
import { useQuery } from "@tanstack/react-query"
import { getEquity, getTrades } from "@/lib/api"
import Nav from "@/components/Nav"
import HelpPanel from "@/components/HelpPanel"

function sharpe(eq: number[]) {
  if (eq.length < 3) return 0
  const rets = eq.slice(1).map((v, i) => v / eq[i] - 1)
  const mean = rets.reduce((a, b) => a + b, 0) / rets.length
  const var_ = rets.reduce((a, b) => a + (b - mean) ** 2, 0) / rets.length
  return var_ === 0 ? 0 : (mean / Math.sqrt(var_)) * Math.sqrt(252 * 75)
}
function maxDD(eq: number[]) {
  let peak = -Infinity, mdd = 0
  for (const v of eq) { peak = Math.max(peak, v); mdd = Math.min(mdd, (v - peak) / peak) }
  return mdd
}

export default function Performance() {
  const eq = useQuery({ queryKey: ["equity"], queryFn: getEquity, refetchInterval: 15_000 })
  const tr = useQuery({ queryKey: ["trades"], queryFn: getTrades, refetchInterval: 15_000 })
  const equity = (eq.data ?? []).map((p) => p.equity)
  const trades = tr.data ?? []
  const wins = trades.filter((t) => t.side === "sell").length
  return (
    <main>
      <Nav />
      <h1>Performance</h1>
      <HelpPanel title="Metrics">
        Sharpe &gt; 1 and max drawdown &gt; -20% are our success targets.
      </HelpPanel>
      <ul>
        <li>Sharpe: {sharpe(equity).toFixed(3)}</li>
        <li>Max drawdown: {(maxDD(equity) * 100).toFixed(2)}%</li>
        <li>Trades: {trades.length}</li>
        <li>Sells (closed): {wins}</li>
      </ul>
    </main>
  )
}