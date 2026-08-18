"use client"
import { useQuery } from "@tanstack/react-query"
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts"
import { getEquity, getPositions, getStatus } from "@/lib/api"
import Nav from "@/components/Nav"
import HelpPanel from "@/components/HelpPanel"

export default function Overview() {
  const eq = useQuery({ queryKey: ["equity"], queryFn: getEquity, refetchInterval: 10_000 })
  const pos = useQuery({ queryKey: ["positions"], queryFn: getPositions, refetchInterval: 10_000 })
  const st = useQuery({ queryKey: ["status"], queryFn: getStatus, refetchInterval: 10_000 })
  const data = (eq.data ?? []).map((p) => ({ t: p.ts, Equity: p.equity }))
  return (
    <main>
      <Nav />
      <h1>Overview</h1>
      <HelpPanel title="Equity curve">
        Your paper-trading account value over time. Upward trend = bot making money.
      </HelpPanel>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <XAxis dataKey="t" /><YAxis domain={["auto", "auto"]} /><Tooltip />
          <Line type="monotone" dataKey="Equity" stroke="#C9A962" dot={false} />
        </LineChart>
      </ResponsiveContainer>
      <h2>Status: {st.data?.killed ? "KILL SWITCHED" : "Running"}</h2>
      <h2>Equity: ₹{st.data?.equity?.toFixed(2)}</h2>
      <h2>Positions</h2>
      <ul>{(pos.data ?? []).map((p) => <li key={p.symbol}>{p.symbol} qty={p.qty}</li>)}</ul>
    </main>
  )
}