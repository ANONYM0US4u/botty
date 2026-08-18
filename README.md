# Self-Learning Intraday Trading Bot

RL-driven intraday trading bot for NSE stocks (Dhan sandbox paper trading) and
crypto futures (Gate testnet), with a Next.js dashboard. PPO trains on
historical bars; a risk gateway owns all execution; promotion to live is a
multi-step state machine (`paper -> staged -> live`), never a single button.

## Architecture

`Scheduler -> RiskGateway -> BrokerAdapter` — the RiskGateway is the only
component that may touch a broker. Every order is validated (daily loss limit,
position cap, exposure cap, stale data, NaN, kill switch) and executes
fail-closed with capped retries on transient errors. Simulator fills on the
**next bar open** (no lookahead). See
`docs/superpowers/specs/2026-08-19-trading-bot-design.md` for the full design.

## Quickstart

```bash
# 1. Environment (Python 3.11+)
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"

# 2. Config
copy config\.env.example config\.env   # add broker keys (Dhan sandbox, Gate testnet)

# 3. Tests
.venv\Scripts\python.exe -m pytest

# 4. Train (probe-verified window from config)
.venv\Scripts\python.exe -m engine.agents.ppo

# 5. Run the engine API (http://127.0.0.1:8000)
.venv\Scripts\python.exe -m uvicorn engine.api.main:create_app --factory

# 6. Dashboard (http://localhost:3000)
cd dashboard
npm i
npm run dev
```

## Smoke test

```bash
.venv\Scripts\python.exe scripts\smoke.py   # prints SMOKE OK {report}
```

See `docs/dashboard-guide.md` for the dashboard tour.
