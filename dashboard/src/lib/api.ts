const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000"

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`, { cache: "no-store" })
  if (!r.ok) throw new Error(`${path}: ${r.status}`)
  return r.json() as Promise<T>
}

export interface EquityPoint { ts: string; equity: number }
export interface Trade { order_id: string; symbol: string; side: string; qty: number; price: number; ts: string }
export interface Checkpoint { path: string; reward: number; sharpe: number; ts: string }
export interface MetricPoint { name: string; value: number; ts: string }
export interface Status { killed: boolean; equity: number; day_pnl: number }

export const getEquity = () => get<EquityPoint[]>("/api/equity")
export const getTrades = () => get<Trade[]>("/api/trades")
export const getCheckpoints = () => get<Checkpoint[]>("/api/checkpoints")
export const getMetrics = (name: string) => get<MetricPoint[]>(`/api/metrics/${name}`)
export const getPositions = () => get<{ symbol: string; qty: number }[]>("/api/positions")
export const getStatus = () => get<Status>("/api/status")
export const setKillSwitch = (active: boolean) =>
  fetch(`${BASE}/api/killswitch`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ active }) })

export function useMetricsWs(onMessage: (m: MetricPoint) => void) {
  if (typeof window === "undefined") return
  const ws = new WebSocket(`${BASE.replace(/^http/, "ws")}/ws/metrics`)
  ws.onmessage = (e) => onMessage(JSON.parse(e.data as string))
  return () => ws.close()
}