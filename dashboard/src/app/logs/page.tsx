"use client"
import { useEffect, useState } from "react"
import Nav from "@/components/Nav"
import { useMetricsWs } from "@/lib/api"

export default function Logs() {
  const [rows, setRows] = useState<string[]>([])
  useEffect(() => {
    const close = useMetricsWs((m) => setRows((r) => [JSON.stringify(m), ...r].slice(0, 200)))
    return close
  }, [])
  return (
    <main>
      <Nav />
      <h1>Logs</h1>
      <pre>{rows.join("\n")}</pre>
    </main>
  )
}