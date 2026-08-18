import Link from "next/link"

const tabs = ["overview", "trades", "performance", "risk", "brain", "logs"]

export default function Nav() {
  return (
    <nav className="nav">
      {tabs.map((t) => (
        <Link key={t} href={`/${t}`}>{t}</Link>
      ))}
    </nav>
  )
}