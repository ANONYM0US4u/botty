"use client"
import { useEffect, useState } from "react"
import Nav from "@/components/Nav"
import { useMetricsSocket } from "@/lib/useMetricsSocket"

export default function Logs() {
  const [rows, setRows] = useState<string[]>([])
  useMetricsSocket((m) => setRows((r) => [JSON.stringify(m), ...r].slice(0, 200)))
  return (
    <main>
      <Nav />
      <h1>Logs</h1>
      <pre>{rows.join("\n")}</pre>
    </main>
  )
}