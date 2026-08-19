"use client"
import { Area, AreaChart, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"

export interface ProbPoint { t: string; buy: number; hold: number; sell: number }

export default function ActionProbsChart({ points }: { points: ProbPoint[] }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <AreaChart data={points}>
        <XAxis dataKey="t" />
        <YAxis domain={[0, 1]} />
        <Tooltip />
        <Legend />
        <Area type="monotone" dataKey="buy" stackId="1" stroke="#22c55e" fill="#22c55e" fillOpacity={0.5} />
        <Area type="monotone" dataKey="hold" stackId="1" stroke="#eab308" fill="#eab308" fillOpacity={0.5} />
        <Area type="monotone" dataKey="sell" stackId="1" stroke="#ef4444" fill="#ef4444" fillOpacity={0.5} />
      </AreaChart>
    </ResponsiveContainer>
  )
}