"use client"
import { useEffect, useRef } from "react"

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000"

export interface TheaterEvent { name: string; payload: Record<string, unknown> }

export function useMetricsSocket(onEvent: (e: TheaterEvent) => void) {
  const cb = useRef(onEvent)
  cb.current = onEvent
  useEffect(() => {
    let ws: WebSocket | null = null
    let closed = false
    let retry = 0
    const open = () => {
      ws = new WebSocket(`${BASE.replace(/^http/, "ws")}/ws/metrics`)
      ws.onopen = () => { retry = 0 }
      ws.onmessage = (e) => {
        try {
          const m = JSON.parse(e.data as string)
          if (m.name && m.payload) cb.current(m)
        } catch { /* ignore malformed frames */ }
      }
      ws.onclose = () => {
        if (!closed) setTimeout(open, Math.min(1000 * 2 ** retry++, 10_000))
      }
    }
    open()
    return () => { closed = true; ws?.close() }
  }, [])
}