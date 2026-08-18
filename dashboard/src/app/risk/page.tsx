"use client"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { getStatus, setKillSwitch } from "@/lib/api"
import Nav from "@/components/Nav"
import HelpPanel from "@/components/HelpPanel"

export default function Risk() {
  const st = useQuery({ queryKey: ["status"], queryFn: getStatus, refetchInterval: 10_000 })
  const qc = useQueryClient()
  const kill = useMutation({ mutationFn: setKillSwitch, onSuccess: () => qc.invalidateQueries({ queryKey: ["status"] }) })
  return (
    <main>
      <Nav />
      <h1>Risk</h1>
      <HelpPanel title="Kill switch">
        Stops all new orders immediately. Use it if the bot behaves unexpectedly.
      </HelpPanel>
      <p>Status: {st.data?.killed ? "KILLED" : "Live"}</p>
      <button onClick={() => kill.mutate(!st.data?.killed)}>
        {st.data?.killed ? "Restart" : "Kill bot"}
      </button>
    </main>
  )
}