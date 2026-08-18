"use client"
import { useQuery } from "@tanstack/react-query"
import { getTrades } from "@/lib/api"
import Nav from "@/components/Nav"
import HelpPanel from "@/components/HelpPanel"

export default function Trades() {
  const tr = useQuery({ queryKey: ["trades"], queryFn: getTrades, refetchInterval: 15_000 })
  const trades = tr.data ?? []
  return (
    <main>
      <Nav />
      <h1>Trades</h1>
      <HelpPanel title="Trade ledger">
        Every executed paper trade. Green = buy (opening), red = sell (closing).
        Win/loss stats live on the Performance tab.
      </HelpPanel>
      <table>
        <thead><tr><th>Time</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Price</th></tr></thead>
        <tbody>
          {trades.map((t, i) => (
            <tr key={i} className={t.side === "buy" ? "win" : "loss"}>
              <td>{t.ts}</td><td>{t.symbol}</td><td>{t.side}</td><td>{t.qty}</td><td>{t.price}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </main>
  )
}