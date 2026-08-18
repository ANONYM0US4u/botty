# Live PPO Theater — Design (V2.1)

Date: 2026-08-19
Status: approved design (Stage 1); Stage 2 is a sketch only
Repo: trading-bot (V1 complete: 62 tests green, Dhan sandbox + Gate testnet live)

## 1. Vision

A dashboard section where the user watches a policy train in real time:
training curves, the actions/probabilities the policy takes on recent bars,
behavior "traits" (hold time, frequency, aggression), a checkpoint
leaderboard, and a paper replay of the latest saved policy over a fixed
validation window — all driven from the browser with Start/Stop/Reset.

## 2. Scope

- **Stage 1 (this spec)**: Live PPO Theater — backend theater module, API
  endpoints, dashboard `/theater` page. Built now.
- **Stage 2 (V2.2, sketch only)**: true evolutionary population trainer
  (N seeds, generations = eval rounds, elite selection, trait vectors
  compared across generations). Documented in the appendix, NOT built now.

## 3. Design decisions (from review, all binding)

1. **Responsive stop/reset**: theater trains in chunks of `n_steps` (2048)
   instead of `save_every` (50k). Stop flag checked between chunks → stop
   responds in seconds, not minutes.
2. **Atomic checkpoint saves**: model saved to `*.zip.tmp` then renamed.
   Replay thread only loads complete files — no half-written-zip crashes.
3. **SQLite concurrency**: `DataStore._conn` gains `PRAGMA busy_timeout=5000`
   (WAL already on). Train thread + replay thread + API share one DB.
4. **Honest naming**: UI labels the live leg "latest-policy replay on recent
   bars", never "live trading". ONE fixed validation window (last 300 bars)
   is used for replay + traits + leaderboard so all numbers are comparable.
5. **Action probabilities**: `decisions.probs` column (already exists) is
   populated by Scheduler for real; WS gets an `emit_json` channel for
   structured payloads (probs/decisions/fills). Metrics table stays scalar.
6. **Isolated runs**: theater runs under `checkpoints/theater/<run_id>`
   (run_id tracking already exists). Reset clears the current run's in-memory
   state and checkpoint files, never the historical DB; past runs remain
   browsable in the leaderboard by run_id.
7. **Testability**: `create_app(store, risk, cfg, theater=None)` — theater
   endpoints return 503 when absent; tests inject a fake theater; zero
   network in tests.
8. **Leaderboard cost**: cached; recomputed only when a new checkpoint
   lands (or on demand), never on every 10s poll.

## 4. Architecture

```
Browser (/theater)                     Backend (single process)
┌──────────────────────────┐           ┌──────────────────────────────────────┐
│ Controls (start/stop/    │─POST────▶│ /api/theater/start|stop|reset|state    │
│  reset + symbol select)  │◀─GET─────│ /api/theater/leaderboard               │
│ Curves (rew/entropy/loss)│           │ TrainingTheater (engine/training/     │
│ Action-prob bars         │◀──WS────▶│   theater.py)                          │
│ Paper replay equity/trades│          │   ├─ train thread: PPO chunks +       │
│ Traits table             │           │   │   _TheaterCallback (probs, hist)  │
│ Leaderboard table        │           │   ├─ replay thread: on new checkpoint │
└──────────────────────────┘           │   │   → Scheduler over window bars    │
                                       │   └─ traits + leaderboard per ckpt    │
                                       │ DataStore (sqlite+parquet, busy_timeout)│
                                       └──────────────────────────────────────┘
```

Threading: train thread (chunked `model.learn`), replay thread (waits for
new complete checkpoint, simulates window, appends fills/equity/decisions),
API/WS in the event loop. No shared mutable state between threads except
the stop flag and the store.

## 5. Components

### 5.1 `engine/training/theater.py` — TrainingTheater

```
TrainingTheater(store, emitter, cfg, fetch_bars, checkpoint_root)
```

State machine: `idle → starting → running → stopping → stopped/error → idle`.
Exposed as `state()` dict: status, symbol, run_id, steps, phase, error.

- `start(symbol)`: resolves instrument kind (stock → `yfinance_nse` via the
  fetch callback injected from `run_live.py`; crypto → `ccxt_crypto` gate),
  fetches bars, saves to store, builds `TradingEnv`, spawns train thread.
  Returns 409 if already running, 400 (with message) on fetch/validation
  failure — e.g. the Task-4 `validate.py` gate must pass.
- train thread loop: `while steps < total_timesteps and not stop_flag`:
  `model.learn(chunk=n_steps, callback=_TheaterCallback)` → atomic save →
  `append_checkpoint` → recompute traits + leaderboard (cache) → emit
  `theater/progress` WS event.
- `stop()`: set flag; thread exits after current chunk; state `stopped`.
- `reset()`: stop; delete `checkpoints/theater/<run_id>` files; state
  `idle`. Never touches `data/`, `data/trading.db`, or other runs.
- `leaderboard()`: for each checkpoint in the current run: replay on the
  fixed window → `compute_eval_report` → rows sorted by sharpe desc, with
  win_rate/mean_reward/traits as columns. Cached; invalidated on new
  checkpoint.
- `traits(checkpoint)`: from the replay's fills/decisions: avg hold bars,
  trade frequency (trades/bar), win rate, max position notional vs cap,
  long/short bias (net fill qty sign). Pure computation in `traits.py`.

### 5.2 `engine/training/traits.py`

Pure function `compute_traits(fills: list[dict], decisions: list[dict],
risk_cfg: dict) -> dict` — no I/O, unit-tested directly.

### 5.3 `_TheaterCallback` (extends `_MetricCallback` in ppo.py)

Adds to the existing 100-step metric emission:
- **Action probabilities**: every step, keep `deque(maxlen=100)` of
  `policy.get_distribution(obs)` probs; every 100 steps emit the rolling
  mean prob vector `{p_flat, p_long, p_short}` via `emit_json("probs", ...)`
  and `append_decision` with probs JSON (live action bar chart source).
- Reuses existing `ep_rew_mean / entropy / policy_loss / value_loss`.

### 5.4 API (`engine/api/main.py`)

- `POST /api/theater/start` `{symbol}` → 200 {state} | 400 {error} | 409
  {error} if running | 503 if theater not configured
- `POST /api/theater/stop` → 200 {state} | 503
- `POST /api/theater/reset` → 200 {state} | 503
- `GET /api/theater/state` → 200 state dict | 503
- `GET /api/theater/leaderboard` → 200 rows | 503
- `create_app(store, risk, cfg, theater=None)` — all theater routes guard
  on `theater is None`.
- `MetricsEmitter.emit_json(name: str, payload: dict)` — sends
  `{"name": ..., "payload": ...}` to WS clients (scalar `emit` unchanged).

### 5.5 `engine/live/scheduler.py` (small change)

`on_bar_close` currently writes `"probs": "[]"`. Change: after
`policy.predict`, compute `policy.policy.get_distribution(obs)` probs and
write the real JSON array into `append_decision` (and include in the
returned summary). Existing tests assert decision rows exist, not probs
content — verify at implementation time.

### 5.6 `engine/data/store.py` (small change)

`_conn()`: add `PRAGMA busy_timeout=5000` after WAL pragma.

### 5.7 `scripts/run_live.py` (replaces run_api.py as the live entry)

Wires: config → DataStore → SimulatorAdapter → RiskGateway → theater
(fetch_bars callback choosing yfinance_nse vs ccxt_crypto by symbol) →
`create_app(..., theater=theater)` → uvicorn. `run_api.py` (watch-only,
no theater) stays for minimal demos.

### 5.8 Dashboard (`dashboard/src`)

- `Nav.tsx`: add "Theater" tab.
- `src/app/theater/page.tsx`: layout with:
  - **Controls**: symbol selector (RELIANCE.NS / BTCUSDT), Start / Stop /
    Reset buttons, status pill (state + steps), error banner.
  - **Training curves**: reuse `TrainingChart` (ep_rew_mean, entropy,
    policy_loss) — already polls `/api/metrics/{name}` every 10s.
  - **Action-prob bars**: stacked bar (flat/long/short shares) from
    `decisions.probs` via a new `GET /api/decisions` read or WS
    `probs` events (see 5.9).
  - **Paper replay**: equity line + recent fills table (from
    `/api/equity` + `/api/trades`).
  - **Traits**: table from leaderboard rows (trait columns).
  - **Leaderboard**: table from `GET /api/theater/leaderboard`, refresh
    on state change / new-checkpoint event.
- `src/lib/api.ts`: `getTheaterState/startTheater/stopTheater/resetTheater/
  getLeaderboard` typed calls.
- WS client (`useMetricsSocket` hook): connects `/ws/metrics`, dispatches
  `probs`/`theater/*` events to local state (curves stay on the 10s poll;
  WS powers the action bars + leaderboard refresh).

### 5.9 `GET /api/decisions?symbol=&limit=`

New read endpoint on DataStore (`get_decisions(symbol, limit)`) — needed
because the dashboard must rebuild the action-prob chart from history on
load (WS only carries events since connection). Returns rows with
`ts, action, probs`.

## 6. Data flow

1. User clicks Start → `POST /api/theater/start {symbol}`.
2. Theater fetches+validates bars, saves to store, spawns train thread.
3. Train thread: chunked learn; every 100 steps metrics → store + WS;
   per-step probs → rolling mean → `decisions` + WS `probs` event.
4. On each checkpoint: atomic save, then replay thread replays the
   checkpoint over the fixed 300-bar window (Scheduler + SimulatorAdapter
   + RiskGateway), appending fills/equity/decisions; traits + leaderboard
   recomputed (cache invalidated); `theater/checkpoint` WS event.
5. Dashboard: curves via 10s poll; action bars via WS events + history
   via `GET /api/decisions` on load; leaderboard/traits refreshed on
   checkpoint events.
6. Stop → flag → thread exits after chunk → state `stopped`. Reset →
   delete current run's checkpoints → `idle`.

## 7. Error handling

| Failure | Behavior |
|---|---|
| Fetch/validation fail (yfinance offline, gate down) | 400 + message, state `idle` |
| Training exception (OOM, NaN, API change) | state `error` + message, WS `theater/error` event; Stop/Reset still work |
| Theater not configured (run_api.py) | all theater routes → 503 |
| Start while running | 409 |
| Replay loads half-written zip | impossible (atomic rename); loader skips files not matching `*.zip` if tmp rename raced |
| SQLite contention | busy_timeout 5000ms + WAL |
| Stop during chunk | graceful, ≤ n_steps late |

## 8. Testing (TDD, no network)

- `tests/training/test_theater.py`: state machine transitions with a fake
  trainer (idle→running→stopping→stopped; double-start → 409-equivalent;
  reset cleans only current run), fetch-failure → error state, stop
  latency bounded (fake trainer checks flag per chunk).
- `tests/training/test_traits.py`: pure function — hold bars, frequency,
  win rate, max-notional-vs-cap, long/short bias on synthetic fills.
- `tests/training/test_leaderboard.py`: ranking order + cache
  invalidation on new checkpoint.
- `tests/api/test_theater_api.py`: TestClient + fake theater — 503 without,
  200/400/409 with, decisions endpoint shape.
- `tests/data/test_store.py`: busy_timeout set (PRAGMA queryable);
  `get_decisions` filtering.
- `tests/live/test_scheduler.py`: probs column now populated with a 3-float
  JSON array (update existing assertions if they assert `"[]"`).
- Existing 62 tests stay green.

## 9. Out of scope (Stage 1)

- Live market streaming (training only on fetched historical bars).
- Multi-symbol concurrent runs (one theater run at a time).
- Evolutionary population trainer (V2.2 — appendix).
- Broker live order routing from the theater (replay is simulator-only).

## Appendix: Stage 2 sketch — Evolutionary Trainer (V2.2, NOT built)

- Population of N PPO seeds trained in parallel; a **generation** = one
  eval round where all candidates replay the fixed window and are ranked.
- Elite selection: top-k survive; new seeds derived from elites (new seed,
  higher LR, weight perturbation) fill the next generation.
- **Trait vectors** (the Stage-1 traits dict) are the observable genome:
  compare trait distributions across generations, show per-generation
  trait evolution charts.
- Leaderboard becomes a lineage tree (who beat whom, which traits spread).
- All Stage-1 theater plumbing (runs, checkpoints, traits, replay,
  leaderboard) is reused; only the training driver changes.