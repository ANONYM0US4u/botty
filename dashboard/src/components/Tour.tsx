"use client"
import { useEffect, useState } from "react"

const STEPS = [
  { tab: "Overview", text: "Your account value over time. Watch the equity curve — it should trend up." },
  { tab: "Brain", text: "The neural network's training curves. Reward shows what the agent is optimizing — check Performance and Risk to see whether that learning is actually useful." },
  { tab: "Trades", text: "Every paper trade the bot made. Green = buy, red = sell." },
  { tab: "Performance", text: "Sharpe, drawdown, win rate. Targets: Sharpe > 1, drawdown > -20%." },
  { tab: "Risk", text: "Kill switch and exposure. Hit it if the bot misbehaves." },
  { tab: "Logs", text: "Live event stream — orders, fills, errors." },
  { tab: "Done", text: "Check docs/dashboard-guide.md for the full manual." },
]

export default function Tour() {
  const [step, setStep] = useState<number | null>(null)
  useEffect(() => {
    if (!localStorage.getItem("tour_done")) setStep(0)
  }, [])
  if (step === null) return null
  const s = STEPS[step]
  return (
    <div className="tour-overlay">
      <div className="tour-card">
        <h2>Welcome to your trading bot</h2>
        <p><strong>{s.tab}:</strong> {s.text}</p>
        <button onClick={() => {
          if (step === STEPS.length - 1) { localStorage.setItem("tour_done", "1"); setStep(null) }
          else setStep(step + 1)
        }}>{step === STEPS.length - 1 ? "Start trading" : "Next"}</button>
      </div>
    </div>
  )
}