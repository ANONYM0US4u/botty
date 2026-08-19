"use client"
import { useEffect } from "react"

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000"

export default function Heartbeat() {
  useEffect(() => {
    const beat = () => {
      try { fetch(`${BASE}/api/heartbeat`, { method: "POST", keepalive: true }) } catch { /* backend may be down */ }
    }
    beat()
    const id = setInterval(beat, 5000)
    const onHide = () => {
      // pagehide fires on tab close, navigate-away and reload; a reload is
      // cancelled by the next heartbeat within the backend's grace window.
      try { navigator.sendBeacon(`${BASE}/api/heartbeat?closing=1`) } catch { /* ignore */ }
    }
    window.addEventListener("pagehide", onHide)
    return () => {
      clearInterval(id)
      window.removeEventListener("pagehide", onHide)
    }
  }, [])
  return null
}