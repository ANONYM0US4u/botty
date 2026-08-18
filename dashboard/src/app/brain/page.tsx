"use client"
import { useQuery } from "@tanstack/react-query"
import Nav from "@/components/Nav"
import HelpPanel from "@/components/HelpPanel"
import TrainingChart from "@/components/TrainingChart"
import { getCheckpoints } from "@/lib/api"

export default function Brain() {
  const ck = useQuery({ queryKey: ["checkpoints"], queryFn: getCheckpoints, refetchInterval: 15_000 })
  return (
    <main>
      <Nav />
      <h1>Brain — Neural Network Monitor</h1>
      <HelpPanel title="Training curves">
        Reward trending up = the policy is learning. Flat/noisy = reward function
        needs work. Entropy dropping = policy becoming certain about its actions
        (uncertainty shown as action probabilities + entropy, never fake confidence).
      </HelpPanel>
      <TrainingChart name="ep_rew_mean" />
      <TrainingChart name="policy_loss" />
      <TrainingChart name="entropy" />
      <h2>Model registry</h2>
      <HelpPanel title="Checkpoints">
        Each saved policy snapshot with its training metrics. The active live
        policy is shown in the scheduler config.
      </HelpPanel>
      <table>
        <thead><tr><th>Checkpoint</th><th>Reward</th><th>Sharpe</th><th>Saved</th></tr></thead>
        <tbody>
          {(ck.data ?? []).map((c) => (
            <tr key={c.path}><td>{c.path}</td><td>{c.reward}</td><td>{c.sharpe}</td><td>{c.ts}</td></tr>
          ))}
        </tbody>
      </table>
    </main>
  )
}