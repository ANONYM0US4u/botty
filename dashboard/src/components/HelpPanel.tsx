export default function HelpPanel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <details className="help">
      <summary>? {title}</summary>
      <p>{children}</p>
    </details>
  )
}