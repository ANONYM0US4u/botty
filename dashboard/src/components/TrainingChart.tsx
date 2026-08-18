"use client"
import { useQuery } from "@tanstack/react-query"
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts"
import { getMetrics } from "@/lib/api"

export default function TrainingChart({ name }: { name: string }) {
  const q = useQuery({ queryKey: ["metrics", name], queryFn: () => getMetrics(name), refetchInterval: 10_000 })
  const data = (q.data ?? []).map((m) => ({ t: m.ts, v: m.value }))
  return (
    <div>
      <h3>{name}</h3>
      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={data}>
          <XAxis dataKey="t" /><YAxis /><Tooltip />
          <Line type="monotone" dataKey="v" stroke="#C9A962" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}