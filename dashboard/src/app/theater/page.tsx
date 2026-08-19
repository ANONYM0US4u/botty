"use client"
import { useState } from "react"
import { useQuery } from "@tanstack/react-query"
import { getDecisions, getLeaderboard, getTheaterState, theaterCommand } from "@/lib/api"
import ActionProbsChart, { type ProbPoint } from "@/components/ActionProbsChart"
import HelpPanel from "@/components/HelpPanel"
import LeaderboardTable from "@/components/LeaderboardTable"
import Nav from "@/components/Nav"
import TraitsTable from "@/components/TraitsTable"
import { useMetricsSocket } from "@/lib/useMetricsSocket"

export default function Theater() {
  const [symbol, setSymbol] = useState("BTCUSDT")
  const [probs, setProbs] = useState<ProbPoint[]>([])
  const [traits, setTraits] = useState<Record<string, unknown>>({})
  const st = useQuery({ queryKey: ["theater-state"], queryFn: getTheaterState, refetchInterval: 5_000 })
  const lb = useQuery({ queryKey: ["theater-lb"], queryFn: getLeaderboard, refetchInterval: 10_000 })
  const dec = useQuery({ queryKey: ["theater-dec"], queryFn: getDecisions, refetchInterval: 10_000 })

  useMetricsSocket((e) => {
    if (e.name === "probs") {
      const p = e.payload as { ts?: string; probs?: number[] }
      if (Array.isArray(p.probs) && p.probs.length >= 3) {
        const [buy, hold, sell] = p.probs
        setProbs((prev) => [...prev, {
          t: (p.ts ?? "").slice(11, 19),
          buy,
          hold,
          sell,
        }].slice(-120))
      }
    } else if (e.name === "theater/traits") {
      setTraits(e.payload)
    }
  })

  const state = st.data ?? { status: "unknown", symbol: null, run_id: null, steps: 0, phase: "", error: "" }
  const running = state.status === "running" || state.status === "starting"
  const latest = dec.data?.[0]
  const [msg, setMsg] = useState("")
  const cmd = async (c: "start" | "stop" | "reset") => {
    try {
      const r = await theaterCommand(c, c === "start" ? symbol : undefined)
      const body = await r.json().catch(() => null)
      if (!r.ok) setMsg(body?.detail ?? body?.error ?? `${c}: HTTP ${r.status}`)
      else setMsg(body?.status ? `${c}: ${body.status}` : `${c}: ok`)
    } catch (e) {
      setMsg(`${c} failed: ${(e as Error).message}`)
    }
  }

  return (
    <main>
      <Nav />
      <h1>Theater</h1>
      <HelpPanel title="Live PPO Theater">
        Trains PPO live on recent bars. Each checkpoint is replayed over the last 300 bars — curves, probabilities and traits come from that replay (latest-policy replay), not live trading.
      </HelpPanel>
      <section>
        <h2>Control</h2>
        <p>
          Status: <strong>{state.status}</strong> · Symbol: {state.symbol ?? "—"} · Run: {state.run_id ?? "—"} · Steps: {state.steps}
        </p>
        <p className="muted">{state.phase || " "}</p>
        {state.error && <p className="err">{state.error}</p>}
        <input value={symbol} onChange={(e) => setSymbol(e.target.value)} placeholder="Symbol (e.g. BTCUSDT)" disabled={running} />
        <button onClick={() => cmd("start")} disabled={running}>Start</button>
        <button onClick={() => cmd("stop")} disabled={!running}>Stop</button>
        <button onClick={() => cmd("reset")} disabled={running}>Reset</button>
        {msg && <p className="err">{msg}</p>}
      </section>
      <section>
        <h2>Action probabilities</h2>
        <ActionProbsChart points={probs} />
        {latest && <p className="muted">Last replay decision: {latest.ts} · action {latest.action} · {latest.probs}</p>}
      </section>
      <section>
        <h2>Traits (latest replay)</h2>
        <TraitsTable traits={traits} />
      </section>
      <section>
        <h2>Checkpoint leaderboard</h2>
        <LeaderboardTable rows={lb.data ?? []} />
      </section>
    </main>
  )
}