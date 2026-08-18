# Trading Bot — Design Document

Date: 2026-08-19
Status: Approved for implementation (V1) — AMENDED 2026-08-19 after external review (`trading-bot/Problems.txt`)

## 0. Amendments (2026-08-19, from Problems.txt review)

The V1 plan was amended to incorporate the review's critical points. These
override any conflicting statement elsewhere in this document:

1. **V1 TradingEnv = single instrument** per env, `Discrete(3)` actions
   `{0=flat, 1=long, 2=short}`. No multi-symbol 3^n encoding / base_repr hacks.
2. **Reward = Δequity** (mark-to-market equity change, not realized P&L), with
   every term (`equity_delta`, `cost`, `drawdown`, `holding`) logged separately
   in `info["reward_terms"]` and persisted to the ledger.
3. **Holding penalty weight = 0 initially**, configurable; raised only after
   the V1 baseline exists.
4. **RiskGateway structurally owns broker access** — it is the ONLY component
   holding the `BrokerAdapter` and calling `place_order`/`cancel_order`;
   scheduler/policy/API/dashboard go through it. Not advisory: unbypassable.
5. **Fail closed**: every execution failure (transient retry w/ capped backoff,
   permanent → reject) leaves positions unchanged.
6. **Causal-feature invariant**: features at index i use bars [0..i] only; a
   dedicated test asserts dropping later bars does not change earlier obs.
7. **Next-bar fill rule**: orders placed at bar t close fill at bar t+1 OPEN
   (no lookahead); decision-bar close is never the fill price.
8. **Data availability gate**: yfinance 5m ≈ 60 days max (CANNOT do 2019–2023);
   a probe script determines the real training window before training; Dhan
   historical API is the fallback for longer intraday history.
9. **"Promote to live" = staged state machine** (`paper → staged → live`),
   never a single button; every transition logged.
10. **Separate retryable vs permanent broker errors** (BrokerError taxonomy).
11. **Per-instrument cost model** (brokerage, STT, stamp duty, exchange txn,
    SEBI, GST, slippage, spread) — no single cost assumption for all instruments.
12. **NSE intraday flatten at 15:15** enforced by RiskGateway.
13. **Experiment tracking**: run_id/model_id + git commit + config hash stored
    with every checkpoint (`run_meta.json`, ledger).
14. **Property-based risk tests** (hypothesis): invariants never violated —
    position/exposure caps, loss limit, stale/killed/NaN → no trade.
15. **Dashboard priority**: V1.0 = Overview/Trades/Performance/Risk/Logs;
    Brain tab (uncertainty = action probs + entropy, never fake "confidence")
    is V1.1 after e2e smoke.
16. **Crypto broker = Gate testnet** (2026-08-19, user decision): Bybit testnet
    key creation was blocked, so the crypto leg uses a GENERIC CCXT adapter
    (`CcxtAdapter(exchange_id, ...)`, default `gate`) instead of BybitAdapter.
    Clock-skew auto-compensation via `options["timeDifference"]` measured from
    `fetch_time()` at init (user PC clock runs ~22s slow). Fetcher default
    `exchange_id` also switched bybit → gate. Keys: `GATE_TESTNET_API_KEY/SECRET`.

## 1. Vision

A self-learning intraday trading bot for Indian equities (NSE via Dhan) and
crypto futures (Bybit/Binance testnets) that trains on historical data,
paper-trades on broker sandbox/testnet environments, and is monitored through
a feature-rich web dashboard that exposes both trading results and the neural
network's inner workings.

## 2. Success Criteria

The bot is considered successful when, in paper trading:

1. **Learning progress** — training curves (reward, win rate, policy entropy)
   show measurable improvement over time.
2. **Beat Nifty** — risk-adjusted return exceeds the NSE index benchmark.
3. **Sharpe ratio > 1** (annualized, intraday).
4. **Max drawdown < 20%**.
5. **Positive P&L over time** — small, consistent profits rather than
   occasional large wins.

Honest caveat: these are a high bar for any retail bot. The design's virtue is
that walk-forward evaluation surfaces the truth quickly, and the paper-money
simulator lets us iterate on reward functions with zero risk.

## 3. Market Scope (V1)

| Asset class | Instruments | Data source | Paper execution |
|---|---|---|---|
| NSE equities (intraday) | NIFTY 50 constituents — start with 1-3 liquid tickers (e.g. RELIANCE.NS, NIFTY index) | yfinance (.NS) historical; Dhan API live quotes; nsepython/jugaad-data fallback | Dhan sandbox keys |
| Crypto futures | BTCUSDT (1 pair to start) | CCXT mainnet OHLCV (testnet history is thin) | Bybit testnet via CCXT |

Frequency: intraday, 1–15 minute bars, decisions on bar close.

## 4. Architecture

Monorepo at `D:\OpenCodeDevelopement\trading-bot\`:

```
trading-bot/
├── engine/                      # Python 3.11+
│   ├── data/
│   │   ├── fetchers/            #   yfinance_nse.py, ccxt_crypto.py, dhan.py
│   │   ├── indicators.py        #   Polars: EMA, RSI, ATR, VWAP, session band
│   │   └── store.py             #   SQLite (WAL) + parquet; append-only ledger
│   ├── env/
│   │   └── trading_env.py       #   gymnasium environment (SINGLE instrument)
│   ├── agents/
│   │   ├── ppo.py               #   stable-baselines3 PPO trainer
│   │   ├── optimize.py          #   Optuna auto-optimizer (V2)
│   │   └── llm_overlay.py       #   LLM risk modulation (V3)
│   ├── brokers/
│   │   ├── base.py              #   BrokerAdapter ABC
│   │   ├── dhan.py              #   Dhan sandbox (NSE)
│   │   ├── bybit.py             #   Bybit testnet via CCXT
│   │   └── simulator.py         #   Simulated fills + slippage (fast training)
│   ├── live/
│   │   ├── scheduler.py         #   Bar-close loop: state → policy → order → ledger
│   │   └── risk.py              #   Kill switch, daily loss limit, exposure caps
│   ├── eval/
│   │   └── metrics.py           #   Sharpe, DD, win rate, profit factor, vs Nifty
│   └── api/
│       └── main.py              #   FastAPI + WebSocket (dashboard backend)
├── dashboard/                   # Next.js 14 + TS + TanStack Query + Recharts
├── config/
│   └── config.yaml              #   Instruments, timeframe, risk limits, reward weights
├── references/                  # 10 cloned reference repos (study only)
├── docs/
│   └── superpowers/specs/       #   Design + plans
└── tests/
```

### Design rules

- Every subsystem communicates through a narrow interface: `BrokerAdapter`,
  `TradingEnv`, `DataStore`, `MetricsEmitter`. Swapping simulator ↔ Dhan, or
  migrating PC → VM, never touches learning code.
- Append-only ledger; nothing in the trading path is mutated.
- Config-driven (YAML); no hardcoded instruments or limits.

## 5. Data Layer

- **Fetchers**: yfinance for NSE minute bars (intraday backfill limited:
  5m ≈ 60 days — see data availability gate, Section 0.8); CCXT for
  crypto mainnet history (read mainnet, trade testnet); Dhan API for live
  stock quotes once keys are provisioned. Fallbacks: nsepython, jugaad-data.
- **Pipeline**: raw OHLCV → parquet; indicators via Polars → feature columns;
  clean bars → SQLite `bars` table (WAL mode).
- **Schema**: `symbols`, `bars`, `orders`, `fills`, `equity_curve`,
  `checkpoints`, `eval_results`, `decisions`, `metrics_snapshots`.
- Cost model: Indian transaction costs (STT + brokerage + GST) and crypto
  taker fees applied in env reward and in simulator fills — per
  anvisgit/Algorithmic-Trading-with-RL findings, naive strategies die on
  Indian costs.

## 6. Learning Engine

### V1 — Reinforcement learning (core)

- **State**: last N=120 bars of features per instrument + current position +
  cash ratio. Feature set: OHLCV-derived (EMA, RSI, ATR, VWAP,
  session band, realized vol) — causal only (bars [0..i], no lookahead).
- **Actions**: per instrument discrete `{long, flat, short}` (Discrete(3));
  expand to size tiers after baseline. V1 runs ONE instrument per
  environment/run: PPO #1 → RELIANCE.NS, PPO #2 → BTCUSDT, separate
  checkpoints and eval per run — never one joint action space.
- **Reward (mathematical, per component, all logged separately)**:

  ```
  reward = w_e · Δequity − w_c · cost − w_dd · drawdown − w_hold · |position|
  Δequity = equity_t − equity_{t−1}   (mark-to-market change, NOT realized P&L)
  cost    = turnover · per-instrument cost rate (brokerage/STT/stamp/GST/slippage)
  drawdown = (peak − equity) / peak
  ```
  Defaults: `w_e = 1.0, w_c = 1.0, w_dd = 0.1, w_hold = 0.0`. Holding penalty
  starts at 0 so the agent is never pre-committed to "don't trade" — raise it
  only after real overtrading is observed. Every term is emitted in
  `info["reward_terms"]` and persisted, so a flat reward curve can be
  decomposed into its causes.
- **Agent**: stable-baselines3 PPO, MLP policy; LSTM policy as experiment.
- **Training protocol**: walk-forward on the PROBE-VERIFIED data window (not
  assumed 2019–2023 — gate: Section 0.8), validate on the most recent period,
  out-of-sample paper test. Deterministic seeds. Checkpoints versioned in
  `checkpoints/` with eval report + run metadata (run_id, git commit,
  config hash) per checkpoint. Training is BLOCKED unless the dataset
  validation gate passes (duplicate/future timestamps, gaps, OHLC
  consistency, timezone normalization, corporate-action policy).

### V2 — Strategy auto-optimizer

- Optuna parameter search over classic strategies (MA cross, RSI, breakout)
  using backtesting.py/vectorbt (AGPL — study patterns; personal use via pip
  is fine, no code copied).
- Regime detector (volatility clustering) selects best strategy per regime.

### V3 — LLM overlay

- LLM scores market regime from news/context (FastAPI streaming endpoint) →
  scales position size via risk multiplier 0.5×–1.5×. Never generates orders
  directly. Blueprint: benstaf/FinRL_DeepSeek.

## 7. Paper Execution (BrokerAdapters)

Interface: `place_order / cancel_order / get_positions / get_orders / get_balance / subscribe_quotes`.

- **DhanAdapter**: `dhan-oss/DhanHQ-py` SDK, sandbox credentials.
- **BybitAdapter**: CCXT with testnet config, USDT-M futures.
- **SimulatorAdapter**: fills at next-bar price + configurable slippage/latency;
  used to train and backtest faster than realtime.
- Order rejection → logged + re-queued (max 3 retries, backoff). Kill switch
  per adapter.

## 8. Orchestration & Risk

- `scheduler.py`: on each bar close → build state → policy action → risk
  checks → order → fill → ledger → emit metrics. NSE hours 09:15–15:30 IST;
  crypto 24/7 (config-gated).
- **Risk module (non-negotiable)**:
  - Daily loss limit (default −3% equity → flatten and halt)
  - Max position size per symbol; max total exposure
  - Stale-data guard (no trades on bars older than 2 min)
  - Manual kill-switch flag (dashboard toggle)

## 9. Evaluation Harness

- Metrics: cumulative return, annualized Sharpe, max drawdown, win rate,
  profit factor, vs-Nifty benchmark (same-period index returns).
- quantstats (Apache-2.0) used for tear sheets in dashboard.
- Every checkpoint → eval report → `eval_results/` → dashboard table with
  "promote to live" action.

## 10. Dashboard (V1 scope — feature-rich)

Next.js 14 + TS; consumes FastAPI REST + WebSocket.

| Tab | Content |
|---|---|
| Overview | Equity curve vs Nifty overlay, P&L today, open positions, exposure |
| Brain | Training curves (reward/losses/entropy, live WS), action-distribution histogram, decision attribution (input perturbation on policy), rollout replay per episode, model registry + promote-to-live, inference latency + uncertainty stats |
| Trades | Full ledger with filters, daily P&L heatmap, per-symbol breakdown |
| Performance | Sharpe, max DD, win rate, profit factor, monthly returns, rolling Sharpe, return histogram |
| Risk | Exposure gauges, daily-loss tracker, limits, kill switch |
| Logs | Live event stream (orders, fills, rejections, retries, data gaps) |

### Tutorial

- In-app onboarding tour (7 steps, one per tab) + "?" help panel on every
  widget explaining what it shows and what good/bad looks like.
- `docs/dashboard-guide.md`: plain-English guide, metric definitions, and a
  "what should I look at daily" quick-start.

## 11. Repo Strategy (installed 2026-08-19)

Cloned to `references/` (MIT = study+use; GPL/AGPL/unlicensed = read-only):

| Repo | Role | License |
|---|---|---|
| dhan-oss/DhanHQ-py | Dhan SDK → adapter | MIT |
| AminHP/gym-anytrading | Env design | MIT |
| AI4Finance-Foundation/FinRL | RL patterns, walk-forward | MIT |
| microsoft/qlib | Production RL+data platform | MIT |
| benstaf/FinRL_DeepSeek | V3 LLM overlay | MIT |
| anvisgit/Algorithmic-Trading-with-RL | NSE costs + Nifty benchmark | none |
| freqtrade/freqtrade | Live loop / risk blueprint | GPL-3.0 |
| polakowo/vectorbt | V2 optimizer engine | other/AGPL |
| kernc/backtesting.py | V2 backtesting | AGPL-3.0 |
| aeron7/nsepython | NSE data fallback | ref only |

Pip libraries (no clone): quantstats (Apache-2.0), jugaad-data, gymnasium,
stable-baselines3, ccxt, optuna, polars, fastapi, pydantic v2.

## 12. Testing

- pytest: indicator math vs known values; env determinism (seed → identical
  rollout); reward sanity; broker adapter round-trips (simulator + testnet);
  risk-module unit tests; walk-forward reproducibility (same seed, same
  checkpoints).
- TDD for all new modules per plan.

## 13. Phases

- **V1**: data pipeline → TradingEnv → PPO training + eval → SimulatorAdapter
  live loop → dashboard (all tabs) + tutorial. Runs on user's PC in 3-4h
  sessions; models persisted, resumable.
- **V2**: Optuna optimizer + regime detector + strategy switching.
- **V3**: LLM overlay + news context (risk modulation only).

## 14. Tech Decisions

- Python 3.11+ engine; FastAPI + Pydantic v2 + Uvicorn; Polars for data;
  SQLite WAL + parquet; structlog.
- Dashboard: Next.js 14 (App Router), TypeScript, TanStack Query v5,
  Recharts, custom heatmaps; `localhost:8000` API, `localhost:3001` UI.
- Hosting: local PC (3-4h sessions, resumable checkpoints); architecture
  allows later migration to a free Oracle Cloud Always-Free ARM VM for 24/7.
- Secrets: `.env` (gitignored) for Dhan sandbox keys and testnet keys.