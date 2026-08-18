# Dashboard Guide — Plain English

## Tabs
- **Overview** — equity curve (your paper account value), current positions, kill status.
- **Brain** — how the neural network is learning. Reward shows what the agent is optimizing, not how good it is — a rising reward can mean turnover explosion or overfitting. Performance and Risk tabs tell you whether the learning is useful.
- **Trades** — every paper trade.
- **Performance** — Sharpe (risk-adjusted return; >1 = good), max drawdown (how far from peak; better than -20%), win rate.
- **Risk** — kill switch, exposure limits.
- **Logs** — live events.

## Daily routine (5 minutes)
1. Open Overview — is equity trending up?
2. Open Brain — reward trending up is not automatically good (it can mean turnover explosion or overfitting). Decompose it via the logged reward terms; if flat for days, tune the reward function.
3. Open Performance — check Sharpe and drawdown vs targets.
4. Check Risk — kill switch off, no limit breaches.

## Metric definitions
- **Sharpe**: (mean return / return volatility) annualized. >1 = profit per unit risk is solid.
- **Max drawdown**: largest peak-to-trough drop in equity. -20% = your stop line.
- **Win rate**: fraction of closed trades that made money.
- **Entropy**: policy uncertainty. High = exploring; low = confident.