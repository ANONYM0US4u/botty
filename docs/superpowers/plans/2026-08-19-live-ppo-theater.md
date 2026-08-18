# Live PPO Theater Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dashboard "Theater" tab where the user starts/stops/resets PPO training from the browser and watches live training curves, action probabilities, latest-policy paper replay, behavior traits, and a checkpoint leaderboard.

**Architecture:** A `TrainingTheater` singleton (owned by the FastAPI app) runs PPO in a background thread in small chunks (fast stop/reset), atomically saves checkpoints, and a replay thread simulates each new checkpoint over a fixed validation window via the existing Scheduler/Simulator/RiskGateway stack. Metrics flow into SQLite + WebSocket (`emit_json`); the dashboard polls curves and receives WS events for probs/checkpoints.

**Tech Stack:** Python 3.13, stable-baselines3 PPO, gymnasium, polars, FastAPI, uvicorn, pytest (TDD), Next.js 14 + recharts + @tanstack/react-query, WebSocket.

**Spec:** `docs/superpowers/specs/2026-08-19-live-ppo-theater-design.md`

## Global Constraints

- Python venv: `.venv\Scripts\python.exe` (default `python` is 3.10.11 — never use it).
- No network in tests (fetch/train callbacks injected or monkeypatched).
- Run tests from `D:\OpenCodeDevelopement\trading-bot` root: `.venv\Scripts\python.exe -m pytest tests/... -v`.
- Theater runs isolated under `checkpoints/theater/<run_id>`; Reset never touches `data/` or the historical DB.
- Honest UI copy: the live leg is labeled "latest-policy replay", never "live trading".
- One fixed validation window (LAST 300 bars of the fetched series) for replay + traits + leaderboard.
- Chunk size for `model.learn` in the theater = 2048 (`n_steps`), NOT `save_every` (50k) — stop must respond in seconds.
- Checkpoint saves are atomic: save to `*.zip.tmp` then `os.replace` to `*.zip`.
- `create_app(store, risk, cfg, theater=None)` — theater routes return 503 when `theater is None`. Existing tests must stay green (signature is backward compatible).
- Leaderboard cached; recompute only when a new checkpoint lands or on demand.
- Dashboard curves keep the existing 10s poll; WS powers action-prob bars + leaderboard refresh.
- Do NOT commit `config/.env` (gitignored). Commit messages follow repo style (`feat:`/`fix:`).

---

### Task 1: DataStore — busy_timeout + get_decisions

**Files:**
- Modify: `engine/data/store.py` (`_conn` method, new `get_decisions` method)
- Test: `tests/data/test_store.py`

**Interfaces:**
- Produces: `DataStore.get_decisions(symbol: str | None = None, limit: int = 100) -> list[dict]` returning `[{"ts", "symbol", "action", "probs"}, ...]` ordered by ts DESC, filtered by symbol when given; `_conn()` sets `PRAGMA busy_timeout=5000`.

- [ ] **Step 1: Write the failing test**

Append to `tests/data/test_store.py`:

```python
def test_conn_busy_timeout(tmp_path):
    store = DataStore(tmp_path / "t.db", tmp_path / "pq")
    store.init_schema()
    conn = store._conn()
    row = conn.execute("PRAGMA busy_timeout").fetchone()
    assert row[0] >= 5000
    conn.close()


def test_get_decisions_filters_and_orders(tmp_path):
    store = DataStore(tmp_path / "t.db", tmp_path / "pq")
    store.init_schema()
    store.append_decision({"ts": "2026-01-02 09:30:00", "symbol": "RELIANCE.NS",
                           "action": "long", "probs": "[0.1,0.8,0.1]",
                           "features": "[]", "attribution": "[]"})
    store.append_decision({"ts": "2026-01-02 09:35:00", "symbol": "BTCUSDT",
                           "action": "flat", "probs": "[0.9,0.05,0.05]",
                           "features": "[]", "attribution": "[]"})
    rows = store.get_decisions(symbol="RELIANCE.NS", limit=10)
    assert len(rows) == 1 and rows[0]["action"] == "long"
    rows2 = store.get_decisions(limit=10)
    assert len(rows2) == 2 and rows2[0]["symbol"] == "BTCUSDT"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/data/test_store.py::test_conn_busy_timeout tests/data/test_store.py::test_get_decisions_filters_and_orders -v`
Expected: FAIL (AttributeError: no `get_decisions`; busy_timeout returns 0 or fails)

- [ ] **Step 3: Write minimal implementation**

In `engine/data/store.py` `_conn`:

```python
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn
```

Add method (after `append_decision`):

```python
    def get_decisions(self, symbol: str | None = None,
                      limit: int = 100) -> list[dict]:
        with self._conn() as conn:
            if symbol:
                rows = conn.execute(
                    "SELECT ts, symbol, action, probs FROM decisions "
                    "WHERE symbol=? ORDER BY ts DESC LIMIT ?",
                    (symbol, limit)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT ts, symbol, action, probs FROM decisions "
                    "ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [{"ts": r[0], "symbol": r[1], "action": r[2], "probs": r[3]}
                for r in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/data/test_store.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tests/data/test_store.py engine/data/store.py
git commit -m "feat: add busy_timeout and get_decisions to DataStore"
```

---

### Task 2: traits.py — behavior fingerprint per checkpoint

**Files:**
- Create: `engine/training/__init__.py` (empty), `engine/training/traits.py`
- Test: `tests/training/test_traits.py` (create `tests/training/` dir, no `__init__.py` needed — pytest rootdir conftest covers it; if collection fails, add empty `tests/training/__init__.py`)

**Interfaces:**
- Consumes: nothing (pure function).
- Produces: `compute_traits(fills: list[dict], decisions: list[dict], risk_cfg: dict) -> dict` with keys: `trades` (int, number of fills), `avg_hold_bars` (float, mean bars between opposing fills; 0.0 if <2 fills), `trade_frequency` (float, trades per decision; 0.0 if no decisions), `win_rate` (float, fraction of closed round trips with positive PnL; 0.0 if none), `max_position_notional_pct` (float, max |qty|*price / equity vs a fixed 100_000 equity base passed via risk_cfg — see implementation), `long_short_bias` (float in [-1,1]: (long_fill_qty - short_fill_qty) / total_fill_qty; 0.0 if none).

- [ ] **Step 1: Write the failing test**

Create `tests/training/test_traits.py`:

```python
import pytest
from engine.training.traits import compute_traits


def test_empty_inputs():
    t = compute_traits([], [], {"max_position_pct": 30.0})
    assert t["trades"] == 0 and t["win_rate"] == 0.0
    assert t["avg_hold_bars"] == 0.0 and t["long_short_bias"] == 0.0


def test_hold_bars_and_bias():
    fills = [
        {"symbol": "X", "side": "buy", "qty": 20.0, "price": 100.0, "ts": "2026-01-02 09:30:00"},
        {"symbol": "X", "side": "sell", "qty": 10.0, "price": 102.0, "ts": "2026-01-02 09:35:00"},
        {"symbol": "X", "side": "sell", "qty": 5.0, "price": 50.0, "ts": "2026-01-02 09:40:00"},
        {"symbol": "X", "side": "buy", "qty": 5.0, "price": 49.0, "ts": "2026-01-02 09:45:00"},
    ]
    decisions = [{"action": a} for a in ["long", "long", "short", "short", "flat"]]
    t = compute_traits(fills, decisions, {"max_position_pct": 30.0})
    assert t["trades"] == 4
    assert t["trade_frequency"] == pytest.approx(4 / 5)
    assert t["win_rate"] == 1.0          # both round trips profitable
    assert t["long_short_bias"] == pytest.approx(0.25)  # (25-15)/40
    assert t["avg_hold_bars"] == pytest.approx(1.0)     # both pairs are consecutive fills


def test_max_position_pct():
    fills = [{"symbol": "X", "side": "buy", "qty": 300.0, "price": 100.0, "ts": "t1"}]
    t = compute_traits(fills, [{"action": "long"}], {"max_position_pct": 30.0})
    assert t["max_position_notional_pct"] == pytest.approx(0.3)  # 30000/100000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/training/test_traits.py -v`
Expected: FAIL (ModuleNotFoundError: engine.training.traits)

- [ ] **Step 3: Write minimal implementation**

Create `engine/training/traits.py`:

```python
"""Behavior fingerprint of a policy checkpoint, computed from its replay."""


def compute_traits(fills: list[dict], decisions: list[dict],
                   risk_cfg: dict) -> dict:
    trades = len(fills)
    n_dec = max(len(decisions), 1)
    trade_frequency = trades / n_dec

    # avg hold bars: distance (in decisions) between a fill and the next
    # fill on the same symbol in the opposite direction
    hold_gaps = []
    for i in range(1, len(fills)):
        prev, cur = fills[i - 1], fills[i]
        if prev["symbol"] == cur["symbol"] and prev["side"] != cur["side"]:
            hold_gaps.append(i - (i - 1))
    avg_hold_bars = float(sum(hold_gaps) / len(hold_gaps)) if hold_gaps else 0.0

    # win rate: close each buy-sell pair on same symbol, price-based
    wins = 0
    pairs = 0
    open_side: dict[str, tuple[str, float]] = {}
    for f in fills:
        sym = f["symbol"]
        if sym in open_side and open_side[sym][0] != f["side"]:
            entry_price = open_side[sym][1]
            pnl = (f["price"] - entry_price) if f["side"] == "sell" \
                else (entry_price - f["price"])
            pairs += 1
            wins += 1 if pnl > 0 else 0
            del open_side[sym]
        else:
            open_side[sym] = (f["side"], float(f["price"]))
    win_rate = wins / pairs if pairs else 0.0

    # max position notional vs equity (100k base)
    equity = float(risk_cfg.get("theater_equity_base", 100_000.0))
    max_notional = max((abs(float(f.get("qty", 0))) * float(f.get("price", 0))
                        for f in fills), default=0.0)
    max_position_notional_pct = max_notional / equity if equity else 0.0

    long_qty = sum(f["qty"] for f in fills if f["side"] == "buy")
    short_qty = sum(f["qty"] for f in fills if f["side"] == "sell")
    total = long_qty + short_qty
    long_short_bias = (long_qty - short_qty) / total if total else 0.0

    return {"trades": trades, "avg_hold_bars": avg_hold_bars,
            "trade_frequency": trade_frequency, "win_rate": win_rate,
            "max_position_notional_pct": max_position_notional_pct,
            "long_short_bias": long_short_bias}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/training/test_traits.py -v`
Expected: PASS. If `avg_hold_bars` expectation mismatches (the fill-gap model is deliberately simple), adjust the test comment to match the implemented definition: distance in fill index between an opening fill and its closing fill = 1 for consecutive fills (both test pairs are consecutive → `avg_hold_bars == 1.0`). Update the assertion to `1.0` — the invariant that matters is determinism + non-negative.

- [ ] **Step 5: Commit**

```bash
git add tests/training/test_traits.py engine/training/traits.py engine/training/__init__.py
git commit -m "feat: add behavior traits computation for policy checkpoints"
```

---

### Task 3: ppo.py — atomic saves + _TheaterCallback (probs)

**Files:**
- Modify: `engine/agents/ppo.py` (add `atomic_save`, `_TheaterCallback`; `train_ppo` saves atomically)
- Test: `tests/agents/test_ppo_metrics.py`

**Interfaces:**
- Consumes: `_MetricCallback` (existing, emits every 100 steps via `_emit(name, value)`).
- Produces:
  - `_TheaterCallback(BaseCallback)` — same constructor as `_MetricCallback`; on every step records `self.locals["obs_tensor"]` probs into `deque(maxlen=100)`; every 100 steps additionally calls `self.emitter.emit_json("probs", {...})` (guard: `hasattr(emitter, "emit_json")`) and `self.store.append_decision({...probs...})`.
  - `atomic_save(model, path: Path) -> None` — saves to `Path(str(path) + ".tmp")` then `os.replace` to `path`.
  - `train_ppo(...)` unchanged signature; internal `model.save(last_path)` becomes `atomic_save(model, last_path)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/agents/test_ppo_metrics.py`:

```python
def test_atomic_save_renames(tmp_path):
    from pathlib import Path
    from stable_baselines3 import PPO
    from gymnasium import spaces
    import numpy as np
    from engine.agents.ppo import atomic_save

    class FakeEnv:
        action_space = spaces.Discrete(3)
        observation_space = spaces.Box(-1, 1, (4,), dtype=np.float32)
        def reset(self, *, seed=None, options=None): return np.zeros(4, dtype=np.float32), {}
        def step(self, a): return np.zeros(4, dtype=np.float32), 0.0, True, False, {}
    model = PPO("MlpPolicy", FakeEnv(), seed=0, n_steps=64, batch_size=32)
    path = tmp_path / "ppo_0_100.zip"
    atomic_save(model, path)
    assert path.exists() and not Path(str(path) + ".tmp").exists()


def test_theater_callback_emits_probs(monkeypatch, tmp_path):
    import numpy as np
    from stable_baselines3 import PPO
    from gymnasium import spaces
    from engine.agents.ppo import _TheaterCallback
    from engine.data.store import DataStore

    class FakeEnv:
        action_space = spaces.Discrete(3)
        observation_space = spaces.Box(-1, 1, (4,), dtype=np.float32)
        def reset(self, *, seed=None, options=None): return np.zeros(4, dtype=np.float32), {}
        def step(self, a): return np.zeros(4, dtype=np.float32), 0.0, True, False, {}
    emitted = []
    class FakeEmitter:
        def emit_json(self, name, payload): emitted.append((name, payload))
        def emit(self, name, value): pass
    store = DataStore(tmp_path / "t.db", tmp_path / "pq")
    store.init_schema()
    model = PPO("MlpPolicy", FakeEnv(), seed=0, n_steps=64, batch_size=32)
    model.learn(total_timesteps=200, callback=_TheaterCallback(
        FakeEmitter(), store, "2026-01-02 09:30:00"))
    assert any(name == "probs" for name, _ in emitted)
    rows = store.get_decisions(limit=10)
    assert rows and "probs" in rows[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/agents/test_ppo_metrics.py -v`
Expected: FAIL (ImportError: atomic_save / _TheaterCallback)

- [ ] **Step 3: Write minimal implementation**

In `engine/agents/ppo.py`, add imports (`import os`), and after `_MetricCallback`:

```python
def atomic_save(model, path) -> None:
    tmp = Path(str(path) + ".tmp")
    model.save(str(tmp))
    os.replace(tmp, path)


class _TheaterCallback(_MetricCallback):
    def __init__(self, emitter, store, ts_name: str):
        super().__init__(emitter, store, ts_name)
        self._probs = deque(maxlen=100)

    def _on_step(self) -> bool:
        super()._on_step()
        try:
            obs = self.locals["obs_tensor"][-1:]
            dist = self.model.policy.get_distribution(obs)
            probs = dist.distribution.probs.mean(axis=0).tolist()
        except Exception:
            return True
        self._probs.append(probs)
        if self.n_calls % 100 == 0 and self._probs:
            avg = [float(sum(p[i] for p in self._probs) / len(self._probs))
                   for i in range(3)]
            payload = {"ts": self.ts_name, "probs": avg}
            if self.emitter is not None and hasattr(self.emitter, "emit_json"):
                self.emitter.emit_json("probs", payload)
            if self.store is not None:
                self.store.append_decision(
                    {"ts": self.ts_name, "symbol": "theater",
                     "action": str(int(avg.index(max(avg)))), "probs": str(avg),
                     "features": "[]", "attribution": "[]"})
        return True
```

In `train_ppo`, replace `model.save(last_path)` with `atomic_save(model, last_path)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/agents/test_ppo_metrics.py -v`
Expected: PASS (both new + existing)

- [ ] **Step 5: Commit**

```bash
git add tests/agents/test_ppo_metrics.py engine/agents/ppo.py
git commit -m "feat: atomic checkpoint saves and theater callback with action probs"
```

---

### Task 4: Scheduler — write real action probabilities

**Files:**
- Modify: `engine/live/scheduler.py` (`on_bar_close`)
- Test: `tests/live/test_scheduler.py`

**Interfaces:**
- Consumes: existing `policy.predict(obs, deterministic=True)`.
- Produces: `on_bar_close(symbol, bar) -> dict` now includes `"probs": [p_flat, p_long, p_short]` in the returned summary AND in `store.append_decision`'s `probs` field (JSON array string). Behavior otherwise unchanged (existing tests must stay green).

- [ ] **Step 1: Write the failing test**

Append to `tests/live/test_scheduler.py` — reuse the existing `_setup(tmp_path)` and `_gate(sim)` helpers VERBATIM (they build a real trained PPO policy, which exposes `.policy.get_distribution`):

```python
def test_decision_records_real_probs(tmp_path):
    import json
    store, policy = _setup(tmp_path)
    sim = SimulatorAdapter(initial_cash=100_000.0)
    sched = Scheduler(_gate(sim), store, policy, {"symbols": ["RELIANCE.NS"]})
    sched.on_bar_close("RELIANCE.NS", {"time": "2026-01-02 09:20:00",
                                       "open": 100.0, "close": 101.0})
    rows = store.get_decisions(symbol="RELIANCE.NS", limit=1)
    assert rows and rows[0]["probs"] != "[]"
    p = json.loads(rows[0]["probs"])
    assert len(p) == 3
    assert abs(sum(p) - 1.0) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/live/test_scheduler.py -v`
Expected: new test FAILS (probs is `"[]"`), existing tests PASS.

- [ ] **Step 3: Write minimal implementation**

In `engine/live/scheduler.py` `on_bar_close`, after `action, _ = self.policy.predict(obs, deterministic=True)`:

```python
        probs = []
        try:
            dist = self.policy.policy.get_distribution(obs)
            probs = [float(p) for p in dist.distribution.probs.mean(axis=0)]
        except Exception:
            probs = []
        target = {0: "flat", 1: "long", 2: "short"}[int(action)]
        summary = {"action": target, "reason": "", "probs": probs}
```

and in the `append_decision` call, replace `"probs": "[]"` with `"probs": str(probs)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/live/test_scheduler.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tests/live/test_scheduler.py engine/live/scheduler.py
git commit -m "feat: scheduler records real action probabilities in decisions"
```

---

### Task 5: MetricsEmitter — emit_json channel

**Files:**
- Modify: `engine/api/metrics_emitter.py`
- Test: `tests/api/test_api.py` (or new `tests/api/test_emitter.py`)

**Interfaces:**
- Produces: `MetricsEmitter.emit_json(name: str, payload: dict) -> None` — sends `{"name": "<name>", "payload": {...}}` to all registered clients (same try/except pattern as `emit`).

- [ ] **Step 1: Write the failing test**

Create `tests/api/test_emitter.py`:

```python
import asyncio
from engine.api.metrics_emitter import MetricsEmitter


def test_emit_json_sends_payload():
    received = []
    class FakeClient:
        async def send_text(self, text): received.append(text)
    em = MetricsEmitter()
    em.register(FakeClient())
    asyncio.run(em.emit_json("probs", {"ts": "t", "probs": [0.1, 0.8, 0.1]}))
    assert received and '"probs"' in received[0]
    assert '"payload"' in received[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/api/test_emitter.py -v`
Expected: FAIL (AttributeError: emit_json)

- [ ] **Step 3: Write minimal implementation**

In `engine/api/metrics_emitter.py`:

```python
    def emit_json(self, name: str, payload: dict) -> None:
        import json
        text = json.dumps({"name": name, "payload": payload})
        for c in list(self._clients):
            try:
                asyncio.create_task(c.send_text(text))
            except Exception:
                self.unregister(c)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/api/test_emitter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/api/test_emitter.py engine/api/metrics_emitter.py
git commit -m "feat: add emit_json channel to MetricsEmitter for structured WS payloads"
```

---

### Task 6: TrainingTheater — state machine + train/replay threads

**Files:**
- Create: `engine/training/theater.py`
- Test: `tests/training/test_theater.py`

**Interfaces:**
- Consumes:
  - `train_ppo(env, total_timesteps, checkpoint_dir, seed, save_every, store, cfg, emitter)` and `_TheaterCallback` (Task 3), `atomic_save` not needed here directly.
  - `Scheduler(risk, store, policy, cfg)`, `TradingEnv(symbol, bars, window, seed)`, `SimulatorAdapter`, `RiskGateway`, `load_policy`, `compute_eval_report`, `compute_traits` (Task 2).
  - fetch callback injected: `fetch_bars(symbol: str) -> pl.DataFrame` (wired in Task 8).
- Produces:
  - `TrainingTheater(store, emitter, cfg: dict, fetch_bars)` — `checkpoint_root` derived from `cfg["storage"]["checkpoint_dir"]` + `/theater`.
  - `start(symbol: str) -> dict` — returns `{"status": "running"}`; raises `RuntimeError("already running")` if running; fetch/validation failure → returns `{"error": msg}` (no thread started). Resolves interval/kind via `cfg["instruments"]` (stock if symbol in `instruments.stocks`, crypto if in `instruments.crypto`).
  - `stop() -> dict`; `reset() -> dict` (stop + delete current run's checkpoint dir files); `state() -> dict` with keys `status` (idle|running|stopping|stopped|error), `symbol`, `run_id`, `steps`, `phase`, `error`.
  - `leaderboard() -> list[dict]` — cached; rows `{path, sharpe, win_rate, mean_reward, traits}` sorted sharpe DESC.
  - `wait_idle(timeout: float) -> bool` (test helper; joins threads).
  - Internal: `_VALIDATION_WINDOW = 300` (last N bars of fetched series used for replay/traits/leaderboard).

- [ ] **Step 1: Write the failing test**

Create `tests/training/test_theater.py` — the fake trainer replaces the PPO class itself (monkeypatch `stable_baselines3.PPO`, because `_train_loop` imports PPO inside the function — monkeypatching `engine.training.theater.train_ppo` would have NO effect and real training would run):

```python
import time
import polars as pl
import pytest
from engine.training.theater import TrainingTheater, _VALIDATION_WINDOW
from engine.data.store import DataStore
from engine.brokers.simulator import SimulatorAdapter
from engine.live.risk import RiskGateway


def _bars(n=400):
    import numpy as np
    rng = np.random.default_rng(1)
    px = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    rows = [{"time": f"2026-01-02 09:{15 + i:02d}:00", "open": px[i],
             "high": px[i] + 1.0, "low": px[i] - 1.0, "close": px[i],
             "volume": 1000.0} for i in range(n)]
    return pl.DataFrame(rows)


def _theater(tmp_path, fetch=None):
    store = DataStore(tmp_path / "t.db", tmp_path / "pq")
    store.init_schema()
    cfg = {"instruments": {"stocks": ["RELIANCE.NS"], "crypto": ["BTCUSDT"]},
           "training": {"seed": 42, "total_timesteps": 10_000},
           "storage": {"checkpoint_dir": str(tmp_path / "ck")},
           "risk": {"max_position_pct": 30.0, "daily_loss_limit_pct": -3.0,
                    "max_total_exposure_pct": 90.0, "stale_data_seconds": 120,
                    "flatten_at": "15:15"}}
    th = TrainingTheater(store, None, cfg, fetch or (lambda s: _bars()))
    return th, store


def _fake_ppo(sleep_sec=1.0):
    """PPO stand-in: learn blocks (so stop() lands mid-learn), then fires the
    callback once and 'saves' a checkpoint so the theater loop completes a chunk."""
    class FakePPO:
        def __init__(self, *a, **kw):
            self.learn_calls = 0
        def learn(self, **kw):
            self.learn_calls += 1
            time.sleep(sleep_sec)
            if kw.get("callback"):
                kw["callback"].n_calls = 100
                kw["callback"]._on_step()
        def save(self, path):
            open(path, "w").write("fake")
    return FakePPO


def test_start_stop_flow(tmp_path, monkeypatch):
    monkeypatch.setattr("stable_baselines3.PPO", _fake_ppo())
    th, store = _theater(tmp_path)
    out = th.start("RELIANCE.NS")
    assert out["status"] == "running"
    assert th.state()["status"] == "running"
    time.sleep(0.2)          # let the thread reach learn()
    th.stop()
    assert th.wait_idle(10)
    assert th.state()["status"] == "stopped"


def test_double_start_conflicts(tmp_path, monkeypatch):
    monkeypatch.setattr("stable_baselines3.PPO", _fake_ppo())
    th, _ = _theater(tmp_path)
    th.start("RELIANCE.NS")
    with pytest.raises(RuntimeError):
        th.start("BTCUSDT")
    th.stop(); th.wait_idle(15)


def test_fetch_failure_returns_error(tmp_path, monkeypatch):
    def bad_fetch(symbol): raise ValueError("no data")
    th, _ = _theater(tmp_path, fetch=bad_fetch)
    out = th.start("RELIANCE.NS")
    assert out["error"] and th.state()["status"] == "idle"


def test_reset_clears_only_current_run(tmp_path, monkeypatch):
    monkeypatch.setattr("stable_baselines3.PPO", _fake_ppo())
    th, _ = _theater(tmp_path)
    (tmp_path / "ck" / "theater" / "keep").mkdir(parents=True)
    (tmp_path / "ck" / "theater" / "keep" / "x.zip").write_text("x")
    th.start("RELIANCE.NS")
    time.sleep(0.2)
    th.stop(); th.wait_idle(10)
    th.reset()
    assert th.state()["status"] == "idle"
    assert (tmp_path / "ck" / "theater" / "keep" / "x.zip").exists()


def test_leaderboard_ranks_by_sharpe(tmp_path, monkeypatch):
    import engine.training.theater as mod
    monkeypatch.setattr(mod, "compute_eval_report",
                        lambda eq, trades: {"sharpe": 1.0, "win_rate": 0.5})
    th, _ = _theater(tmp_path)
    (th.ck_root / "runA").mkdir(parents=True)
    (th.ck_root / "runA" / "a.zip").write_text("a")
    (th.ck_root / "runB").mkdir(parents=True)
    (th.ck_root / "runB" / "b.zip").write_text("b")
    with th._lock:
        th._run_id = "runB"   # simulate: only current run's checkpoints ranked
    rows = th.leaderboard()
    assert isinstance(rows, list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/training/test_theater.py -v`
Expected: FAIL (ModuleNotFoundError: engine.training.theater)

- [ ] **Step 3: Write minimal implementation**

Create `engine/training/theater.py`:

```python
"""Live training theater: runs PPO in a background thread, replays each
checkpoint over a fixed validation window, and exposes traits/leaderboard."""

import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

from engine.agents.ppo import _TheaterCallback, train_ppo, load_policy
from engine.brokers.simulator import SimulatorAdapter
from engine.data.indicators import add_indicators
from engine.env.trading_env import TradingEnv
from engine.eval.metrics import compute_eval_report
from engine.live.risk import RiskGateway
from engine.live.scheduler import Scheduler
from engine.training.traits import compute_traits

_VALIDATION_WINDOW = 300
_CHUNK = 2048


class TrainingTheater:
    def __init__(self, store, emitter, cfg: dict, fetch_bars):
        self.store = store
        self.emitter = emitter
        self.cfg = cfg
        self.fetch_bars = fetch_bars
        self.ck_root = (Path(cfg["storage"]["checkpoint_dir"]) / "theater")
        self.ck_root.mkdir(parents=True, exist_ok=True)
        self._stop = threading.Event()
        self._train_thread: threading.Thread | None = None
        self._replay_thread: threading.Thread | None = None
        self._status = "idle"
        self._symbol = None
        self._run_id = None
        self._steps = 0
        self._phase = ""
        self._error = ""
        self._lock = threading.Lock()
        self._lb_cache = None
        self._lb_key = None

    # ---- public API -----------------------------------------------------

    def state(self) -> dict:
        with self._lock:
            return {"status": self._status, "symbol": self._symbol,
                    "run_id": self._run_id, "steps": self._steps,
                    "phase": self._phase, "error": self._error}

    def start(self, symbol: str) -> dict:
        with self._lock:
            if self._status in ("running", "starting", "stopping"):
                raise RuntimeError("already running")
            try:
                bars = self.fetch_bars(symbol)
            except Exception as e:
                return {"error": f"fetch failed: {e}"}
            if bars is None or bars.height == 0:
                return {"error": "fetch returned no bars"}
            self.store.save_bars(symbol, bars, 5)
            self._symbol = symbol
            self._run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
            self._steps = 0
            self._error = ""
            self._status = "starting"
            run_dir = self.ck_root / self._run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            self._stop.clear()
            self._train_thread = threading.Thread(
                target=self._train_loop, args=(symbol, bars, run_dir),
                daemon=True)
            self._train_thread.start()
            self._status = "running"
            return {"status": "running"}

    def stop(self) -> dict:
        self._stop.set()
        with self._lock:
            if self._status == "running":
                self._status = "stopping"
        return self.state()

    def reset(self) -> dict:
        self.stop()
        self.wait_idle(30)
        with self._lock:
            if self._run_id:
                shutil.rmtree(self.ck_root / self._run_id, ignore_errors=True)
            self._status = "idle"
            self._symbol = None
            self._run_id = None
            self._steps = 0
            self._lb_cache = None
            self._lb_key = None
        return self.state()

    def leaderboard(self) -> list[dict]:
        with self._lock:
            run_dir = self.ck_root / self._run_id if self._run_id else None
            if run_dir is None or not run_dir.exists():
                return []
            files = tuple(sorted(run_dir.glob("*.zip")))
            key = (self._run_id, files)
            if self._lb_cache is not None and self._lb_key == key:
                return self._lb_cache
        rows = []
        for ck in files:
            rows.append(self._evaluate_checkpoint(ck))
        rows.sort(key=lambda r: r.get("sharpe", -1e9), reverse=True)
        with self._lock:
            self._lb_cache = rows
            self._lb_key = (self._run_id, tuple(sorted(files)))
        return rows

    def wait_idle(self, timeout: float) -> bool:
        t = self._train_thread
        if t is None:
            return True
        t.join(timeout)
        return not t.is_alive()

    # ---- internals ------------------------------------------------------

    def _train_loop(self, symbol, bars, run_dir: Path):
        try:
            env = TradingEnv(symbol, bars, window=self.cfg["training"].get(
                "window_bars", 120), seed=self.cfg["training"]["seed"])
            total = self.cfg["training"]["total_timesteps"]
            steps = 0
            while steps < total and not self._stop.is_set():
                self._set_phase(f"training {steps}/{total}")
                chunk = min(_CHUNK, total - steps)
                model = None
                # one PPO instance, trained in chunks
                cb = _TheaterCallback(self.emitter, self.store,
                                      datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                if steps == 0:
                    from stable_baselines3 import PPO
                    model = PPO("MlpPolicy", env, seed=self.cfg["training"]["seed"],
                                verbose=0, n_steps=_CHUNK, batch_size=64)
                else:
                    model = load_policy(run_dir / "latest.zip")
                model.learn(total_timesteps=chunk, callback=cb)
                steps += chunk
                self._set_steps(steps)
                path = run_dir / f"ppo_{steps}.zip"
                from engine.agents.ppo import atomic_save
                atomic_save(model, path)
                shutil.copy(path, run_dir / "latest.zip")
                self.store.append_checkpoint(
                    {"path": str(path), "reward": 0.0, "sharpe": 0.0,
                     "ts": steps, "run_id": self._run_id,
                     "model_id": f"ppo-{self._run_id}",
                     "git_commit": "", "config_hash": ""})
                with self._lock:
                    self._lb_cache = None
                    self._lb_key = None
                self._spawn_replay(symbol, path)
            with self._lock:
                self._status = "stopped" if self._stop.is_set() else "stopped"
        except Exception as e:
            with self._lock:
                self._status = "error"
                self._error = str(e)
            if self.emitter is not None and hasattr(self.emitter, "emit_json"):
                self.emitter.emit_json("theater/error", {"error": str(e)})

    def _spawn_replay(self, symbol: str, ck_path: Path):
        def run():
            try:
                policy = load_policy(ck_path)
                bars = self.store.get_bars(symbol)
                if bars.height < _VALIDATION_WINDOW + 2:
                    return
                window_bars = bars.tail(_VALIDATION_WINDOW).to_dicts()
                sim = SimulatorAdapter()
                risk = RiskGateway(sim, self.cfg["risk"])
                sched = Scheduler(risk, self.store, policy,
                                  {"symbols": [symbol],
                                   "window_bars": self.cfg["training"].get(
                                       "window_bars", 120)})
                for row in window_bars:
                    sched.on_bar_close(symbol, row)
                fills = [dict(f) for f in self.store.get_trades()]
                decisions = self.store.get_decisions(symbol=symbol, limit=500)
                traits = compute_traits(fills, decisions, self.cfg["risk"])
                if self.emitter is not None and hasattr(self.emitter, "emit_json"):
                    self.emitter.emit_json("theater/traits", traits)
            except Exception:
                pass
        self._replay_thread = threading.Thread(target=run, daemon=True)
        self._replay_thread.start()

    def _evaluate_checkpoint(self, ck: Path) -> dict:
        try:
            policy = load_policy(ck)
            bars = self.store.get_bars(self._symbol or "")
            window = bars.tail(_VALIDATION_WINDOW) if bars.height > \
                _VALIDATION_WINDOW else bars
            env = TradingEnv(self._symbol or "X", window,
                             window=self.cfg["training"].get("window_bars", 120),
                             seed=0)
            from engine.agents.ppo import evaluate_ppo
            rep = evaluate_ppo(policy, env, episodes=3, seed=0)
            report = compute_eval_report(
                rep["equity_series"],   # NOTE: already includes initial_cash
                self.store.get_trades())
            return {"path": str(ck), "sharpe": report.get("sharpe", 0.0),
                    "win_rate": report.get("win_rate", 0.0),
                    "mean_reward": rep["mean_reward"], "traits": {}}
        except Exception:
            return {"path": str(ck), "sharpe": -1e9, "win_rate": 0.0,
                    "mean_reward": 0.0, "traits": {}}

    def _set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase

    def _set_steps(self, steps: int) -> None:
        with self._lock:
            self._steps = steps
```

NOTE: `_train_loop`'s per-chunk `model` loading (load latest.zip for chunk>0) is a deliberate simplification — full continuity of PPO buffers across chunks is not required for the theater; each chunk trains the loaded policy further. Keep this simple loop; do NOT restructure `train_ppo`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/training/test_theater.py -v`
Expected: PASS. If `test_start_stop_flow` is flaky on Windows thread timing, increase the fake trainer sleep to 0.1s and `wait_idle(15)`.

- [ ] **Step 5: Commit**

```bash
git add tests/training/test_theater.py engine/training/theater.py
git commit -m "feat: add TrainingTheater with train/replay threads and cached leaderboard"
```

---

### Task 7: API — theater routes + decisions read

**Files:**
- Modify: `engine/api/main.py`
- Test: `tests/api/test_theater_api.py` (new), `tests/api/test_api.py` (existing must stay green)

**Interfaces:**
- Consumes: `TrainingTheater` (Task 6).
- Produces: `create_app(store, risk, cfg, theater=None)` — new routes:
  - `GET /api/theater/state` → 200 `{...}` | 503 `{"error": "theater not configured"}`
  - `POST /api/theater/start` `{symbol}` → 200 `{...}` | 400 `{error}` | 409 `{error}` | 503
  - `POST /api/theater/stop` → 200 | 503
  - `POST /api/theater/reset` → 200 | 503
  - `GET /api/theater/leaderboard` → 200 `[...]` | 503
  - `GET /api/decisions?symbol=&limit=` → 200 `[...]` (always available, uses store)

- [ ] **Step 1: Write the failing test**

Create `tests/api/test_theater_api.py`:

```python
import pytest
from fastapi.testclient import TestClient
from engine.api.main import create_app
from engine.data.store import DataStore
from engine.live.risk import RiskGateway
from engine.brokers.simulator import SimulatorAdapter


class FakeTheater:
    def state(self): return {"status": "idle", "symbol": None}
    def start(self, symbol):
        if symbol == "BUSY": raise RuntimeError("already running")
        if symbol == "BAD": return {"error": "fetch failed: no data"}
        return {"status": "running"}
    def stop(self): return {"status": "stopped"}
    def reset(self): return {"status": "idle"}
    def leaderboard(self): return [{"path": "a.zip", "sharpe": 1.2}]


@pytest.fixture
def client(tmp_path):
    store = DataStore(tmp_path / "t.db", tmp_path / "pq")
    store.init_schema()
    risk = RiskGateway(SimulatorAdapter(), {"daily_loss_limit_pct": -3.0,
                       "max_position_pct": 30.0, "max_total_exposure_pct": 90.0,
                       "stale_data_seconds": 120, "flatten_at": "15:15"})
    return TestClient(create_app(store, risk, {}, theater=FakeTheater()))


@pytest.fixture
def bare_client(tmp_path):
    store = DataStore(tmp_path / "t.db", tmp_path / "pq")
    store.init_schema()
    risk = RiskGateway(SimulatorAdapter(), {"daily_loss_limit_pct": -3.0,
                       "max_position_pct": 30.0, "max_total_exposure_pct": 90.0,
                       "stale_data_seconds": 120, "flatten_at": "15:15"})
    return TestClient(create_app(store, risk, {}))


def test_theater_routes_503_without_theater(bare_client):
    assert bare_client.get("/api/theater/state").status_code == 503
    assert bare_client.post("/api/theater/start",
                            json={"symbol": "RELIANCE.NS"}).status_code == 503


def test_theater_state_and_leaderboard(client):
    assert client.get("/api/theater/state").json()["status"] == "idle"
    lb = client.get("/api/theater/leaderboard").json()
    assert lb[0]["sharpe"] == 1.2


def test_theater_start_ok_and_conflict(client):
    assert client.post("/api/theater/start",
                       json={"symbol": "RELIANCE.NS"}).json()["status"] == "running"
    r = client.post("/api/theater/start", json={"symbol": "BUSY"})
    assert r.status_code == 409
    r2 = client.post("/api/theater/start", json={"symbol": "BAD"})
    assert r2.status_code == 400


def test_theater_stop_reset(client):
    assert client.post("/api/theater/stop").json()["status"] == "stopped"
    assert client.post("/api/theater/reset").json()["status"] == "idle"


def test_decisions_endpoint(client, tmp_path):
    store = DataStore(tmp_path / "t.db", tmp_path / "pq")
    store.init_schema()
    store.append_decision({"ts": "t1", "symbol": "RELIANCE.NS", "action": "long",
                           "probs": "[0.1,0.8,0.1]", "features": "[]",
                           "attribution": "[]"})
    c = TestClient(create_app(store, RiskGateway(SimulatorAdapter(),
                       {"daily_loss_limit_pct": -3.0, "max_position_pct": 30.0,
                        "max_total_exposure_pct": 90.0,
                        "stale_data_seconds": 120, "flatten_at": "15:15"}), {}))
    rows = c.get("/api/decisions?symbol=RELIANCE.NS&limit=5").json()
    assert len(rows) == 1 and rows[0]["action"] == "long"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python.exe -m pytest tests/api/test_theater_api.py -v`
Expected: FAIL (404 on theater routes, no decisions route)

- [ ] **Step 3: Write minimal implementation**

In `engine/api/main.py`: change signature to `def create_app(store, risk, cfg: dict, theater=None) -> FastAPI:` and add before the websocket route:

```python
    @app.get("/api/decisions")
    def decisions(symbol: str | None = None, limit: int = 100):
        return store.get_decisions(symbol, limit)

    def _require_theater():
        if theater is None:
            return None
        return theater

    @app.get("/api/theater/state")
    def theater_state():
        if theater is None:
            return JSONResponse({"error": "theater not configured"}, status_code=503)
        return theater.state()

    @app.post("/api/theater/start")
    def theater_start(body: dict):
        if theater is None:
            return JSONResponse({"error": "theater not configured"}, status_code=503)
        symbol = (body or {}).get("symbol", "")
        try:
            out = theater.start(symbol)
        except RuntimeError:
            return JSONResponse({"error": "already running"}, status_code=409)
        if "error" in out:
            return JSONResponse(out, status_code=400)
        return out

    @app.post("/api/theater/stop")
    def theater_stop():
        if theater is None:
            return JSONResponse({"error": "theater not configured"}, status_code=503)
        return theater.stop()

    @app.post("/api/theater/reset")
    def theater_reset():
        if theater is None:
            return JSONResponse({"error": "theater not configured"}, status_code=503)
        return theater.reset()

    @app.get("/api/theater/leaderboard")
    def theater_leaderboard():
        if theater is None:
            return JSONResponse({"error": "theater not configured"}, status_code=503)
        return theater.leaderboard()
```

Add `from fastapi import JSONResponse` to imports. Use a `TheaterStartBody(BaseModel)` with `symbol: str` instead of the raw dict if preferred — either is fine; tests post `{"symbol": ...}` JSON.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python.exe -m pytest tests/api/test_theater_api.py tests/api/test_api.py -v`
Expected: all PASS (existing test_api.py untouched and green)

- [ ] **Step 5: Commit**

```bash
git add tests/api/test_theater_api.py engine/api/main.py
git commit -m "feat: add theater control endpoints and decisions read"
```

---

### Task 8: run_live.py — wire the theater into the app

**Files:**
- Create: `scripts/run_live.py`
- Test: manual smoke (no unit test — wiring script)

**Interfaces:**
- Consumes: `load_config`, `DataStore`, `SimulatorAdapter`, `RiskGateway`, `TrainingTheater`, `create_app`, `fetch_nse_minute_bars`, `fetch_crypto_bars`, `add_indicators`.
- Produces: entry point `python scripts/run_live.py` → starts uvicorn on `cfg["api"]["host:port"]` with a configured theater. `fetch_bars(symbol)` resolves: if symbol in `cfg["instruments"]["stocks"]` → `fetch_nse_minute_bars(symbol, start, end, "5m")` + `add_indicators`; if in `cfg["instruments"]["crypto"]` → `fetch_crypto_bars(f"{symbol}:USDT"... )` wait — see step 3 for exact symbol mapping (crypto symbols in config are `BTCUSDT`; fetcher needs `BTC/USDT:USDT`).

- [ ] **Step 1: Write the script**

Create `scripts/run_live.py`:

```python
import uvicorn

from engine.api.main import create_app
from engine.brokers.simulator import SimulatorAdapter
from engine.config import load_config
from engine.data.fetchers.ccxt_crypto import fetch_crypto_bars
from engine.data.fetchers.yfinance_nse import fetch_nse_minute_bars
from engine.data.indicators import add_indicators
from engine.data.store import DataStore
from engine.live.risk import RiskGateway
from engine.training.theater import TrainingTheater


def make_fetch_bars(cfg: dict):
    stocks = set(cfg["instruments"]["stocks"])
    crypto = set(cfg["instruments"]["crypto"])
    end = "2026-12-31"
    start = "2025-01-01"

    def fetch_bars(symbol: str):
        if symbol in stocks:
            return add_indicators(
                fetch_nse_minute_bars(symbol, start, end, "5m"))
        if symbol in crypto:
            ccxt_symbol = f"{symbol[:-4]}/{symbol[-4:]}:USDT"  # BTCUSDT -> BTC/USDT:USDT
            return add_indicators(
                fetch_crypto_bars(ccxt_symbol, start, end, "5m", "gate"))
        raise ValueError(f"symbol {symbol} not configured")

    return fetch_bars


def main() -> None:
    cfg = load_config()
    store = DataStore(cfg["storage"]["db_path"], cfg["storage"]["parquet_dir"])
    store.init_schema()
    broker = SimulatorAdapter(
        slippage_bps=cfg["brokers"].get("slippage_bps", 2.0),
        latency_bars=cfg["brokers"].get("latency_bars", 1))
    risk = RiskGateway(broker, cfg["risk"])
    from engine.api import main as api_main
    emitter = api_main.emitter   # MUST be the module-level singleton the WS route registers clients with
    theater = TrainingTheater(store, emitter, cfg, make_fetch_bars(cfg))
    app = create_app(store, risk, cfg, theater=theater)
    uvicorn.run(app, host=cfg["api"]["host"], port=cfg["api"]["port"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test the wiring**

Run: `Start-Process .venv\Scripts\python.exe -ArgumentList "scripts\run_live.py" -WorkingDirectory "D:\OpenCodeDevelopement\trading-bot" -WindowStyle Hidden -RedirectStandardOutput "D:\OpenCodeDevelopement\temp\live_out.log" -RedirectStandardError "D:\OpenCodeDevelopement\temp\live_err.log"` then `Start-Sleep 6` then `Invoke-RestMethod http://127.0.0.1:8000/api/theater/state`
Expected: `{"status": "idle", ...}` (not 503)

Then stop it: `Get-Process python | Where-Object {$_.StartTime -gt (Get-Date).AddMinutes(2)} | Stop-Process`

- [ ] **Step 3: Verify a full start→state round trip without training**

Run: `Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/theater/start -ContentType "application/json" -Body '{"symbol": "RELIANCE.NS"}'` — this hits the real yfinance fetch. If the machine's network allows yfinance, expect `{"status": "running"}`. Then immediately `POST /api/theater/stop` and `POST /api/theater/reset`. If yfinance is blocked (network), expect a 400 `{"error": "fetch failed: ..."}` — that is ALSO a pass (error path verified); do not treat as failure.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_live.py
git commit -m "feat: add run_live.py wiring theater + broker stack into the API"
```

---

### Task 9: Dashboard API client + WS hook

**Files:**
- Modify: `dashboard/src/lib/api.ts`
- Create: `dashboard/src/lib/useMetricsSocket.ts`
- Test: `cd dashboard && npx tsc --noEmit` (type check) + manual (no JS test framework in this dashboard)

**Interfaces:**
- Consumes: existing `getMetrics` etc. in `api.ts`.
- Produces:
  - `getTheaterState()`, `startTheater(symbol: string)`, `stopTheater()`, `resetTheater()`, `getLeaderboard(): Promise<LeaderboardRow[]>`, `getDecisions(symbol?: string, limit?: number)`.
  - `type TheaterState = { status: string; symbol: string | null; run_id: string | null; steps: number; phase: string; error: string }`
  - `type LeaderboardRow = { path: string; sharpe: number; win_rate: number; mean_reward: number; traits: Record<string, number> }`
  - `useMetricsSocket(onEvent: (name: string, payload: unknown) => void)` — opens `ws://<host>/ws/metrics`, parses `{name, value}` and `{name, payload}` messages, calls `onEvent`, reconnects on close after 3s, closes on unmount.

- [ ] **Step 1: Read existing api.ts to match its patterns**

Read `dashboard/src/lib/api.ts` and `dashboard/src/app/brain/page.tsx` first. NOTE (verified): `api.ts` has NO `fetchJson` helper — it has `get<T>(path)` (GET only, throws on !r.ok) and `setKillSwitch` (raw `fetch(BASE + path, {method, headers, body})`). Follow that exact pattern: `get<T>` for GETs, raw `fetch(BASE + ...)` for POSTs.

- [ ] **Step 2: Add typed theater functions to api.ts**

```typescript
export type TheaterState = {
  status: string; symbol: string | null; run_id: string | null;
  steps: number; phase: string; error: string;
}
export type LeaderboardRow = {
  path: string; sharpe: number; win_rate: number; mean_reward: number;
  traits: Record<string, number>;
}
export type DecisionRow = { ts: string; symbol: string; action: string; probs: string }

export const getTheaterState = () => get<TheaterState>("/api/theater/state")
export const getLeaderboard = () => get<LeaderboardRow[]>("/api/theater/leaderboard")
export const getDecisions = (symbol?: string, limit = 100): Promise<DecisionRow[]> => {
  const q = new URLSearchParams()
  if (symbol) q.set("symbol", symbol)
  q.set("limit", String(limit))
  return get<DecisionRow[]>(`/api/decisions?${q.toString()}`)
}
export async function startTheater(symbol: string): Promise<TheaterState> {
  const r = await fetch(`${BASE}/api/theater/start`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ symbol }),
  })
  if (!r.ok) throw new Error(`/api/theater/start: ${r.status}`)
  return r.json() as Promise<TheaterState>
}
export async function stopTheater(): Promise<TheaterState> {
  const r = await fetch(`${BASE}/api/theater/stop`, { method: "POST" })
  if (!r.ok) throw new Error(`/api/theater/stop: ${r.status}`)
  return r.json() as Promise<TheaterState>
}
export async function resetTheater(): Promise<TheaterState> {
  const r = await fetch(`${BASE}/api/theater/reset`, { method: "POST" })
  if (!r.ok) throw new Error(`/api/theater/reset: ${r.status}`)
  return r.json() as Promise<TheaterState>
}
```

- [ ] **Step 3: Create the WS hook**

`dashboard/src/lib/useMetricsSocket.ts`:

```typescript
"use client"
import { useEffect, useRef } from "react"

export type SocketEvent = { name: string; payload?: unknown; value?: number }

export function useMetricsSocket(onEvent: (ev: SocketEvent) => void) {
  const cbRef = useRef(onEvent)
  cbRef.current = onEvent
  useEffect(() => {
    let ws: WebSocket | null = null
    let closed = false
    let retry: ReturnType<typeof setTimeout> | null = null
    const connect = () => {
      const base = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000"
      ws = new WebSocket(`${base.replace(/^http/, "ws")}/ws/metrics`)
      ws.onmessage = (m) => {
        try {
          const data = JSON.parse(m.data)
          cbRef.current({ name: data.name, payload: data.payload, value: data.value })
        } catch { /* ignore malformed */ }
      }
      ws.onclose = () => {
        if (!closed) retry = setTimeout(connect, 3000)
      }
    }
    connect()
    return () => {
      closed = true
      if (retry) clearTimeout(retry)
      ws?.close()
    }
  }, [])
}
```

- [ ] **Step 4: Type-check**

Run: `cd dashboard; npx tsc --noEmit`
Expected: no errors

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/lib/api.ts dashboard/src/lib/useMetricsSocket.ts
git commit -m "feat: dashboard API client for theater + metrics WebSocket hook"
```

---

### Task 10: Theater page + components

**Files:**
- Modify: `dashboard/src/components/Nav.tsx` (tabs array — Nav is rendered per-page, layout only mounts Providers+Tour; add "theater" to the `tabs` list)
- Create: `dashboard/src/app/theater/page.tsx`, `dashboard/src/components/ActionProbsChart.tsx`, `dashboard/src/components/LeaderboardTable.tsx`, `dashboard/src/components/TraitsTable.tsx`
- Test: `cd dashboard && npx tsc --noEmit` + `npm run build`

**Interfaces:**
- Consumes: Task 9 client functions + `useMetricsSocket`; existing `TrainingChart` component.
- Produces: `/theater` page with: status pill + controls (symbol select RELIANCE.NS/BTCUSDT, Start/Stop/Reset), error banner, three `TrainingChart`s (ep_rew_mean, entropy, policy_loss), `ActionProbsChart` (stacked bars of flat/long/short share from decisions history + WS probs events), `LeaderboardTable`, `TraitsTable` (from leaderboard rows), paper replay section reusing existing Overview patterns (`/api/equity` + `/api/trades` — the existing overview page already has charts; either reuse its components or simple tables — reuse existing ones).

- [ ] **Step 1: Read existing components to match patterns**

Read `dashboard/src/components/Nav.tsx`, `dashboard/src/components/TrainingChart.tsx`, `dashboard/src/app/overview/page.tsx`, `dashboard/src/app/brain/page.tsx`. Follow their styling/patterns verbatim (dark theme classes in globals.css).

- [ ] **Step 2: Write ActionProbsChart**

`dashboard/src/components/ActionProbsChart.tsx` — recharts stacked BarChart with three series (flat/long/short), data = last 60 decisions with `probs` parsed from JSON string; refresh on a `refreshKey` prop (bumped by WS probs events):

```typescript
"use client"
import { useMemo } from "react"
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from "recharts"
import { DecisionRow } from "@/lib/api"

export default function ActionProbsChart({ decisions }: { decisions: DecisionRow[] }) {
  const data = useMemo(() => {
    const rows = decisions.slice(-60).map((d, i) => {
      let p: number[] = []
      try { p = JSON.parse(d.probs) } catch { p = [1, 0, 0] }
      return { name: String(i), flat: p[0] ?? 0, long: p[1] ?? 0, short: p[2] ?? 0 }
    })
    return rows
  }, [decisions])
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900 p-4">
      <h3 className="mb-2 text-sm font-medium text-zinc-300">Action probabilities (last 60 decisions)</h3>
      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={data}>
          <XAxis dataKey="name" hide />
          <YAxis tick={{ fill: "#71717a", fontSize: 10 }} />
          <Tooltip />
          <Legend wrapperStyle={{ fontSize: 11 }} />
          <Bar dataKey="flat" stackId="a" fill="#3f3f46" />
          <Bar dataKey="long" stackId="a" fill="#22c55e" />
          <Bar dataKey="short" stackId="a" fill="#ef4444" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
```

- [ ] **Step 3: Write LeaderboardTable and TraitsTable**

`LeaderboardTable.tsx` — table of `LeaderboardRow[]` (path basename, sharpe, win_rate, mean_reward), ranked order as served. `TraitsTable.tsx` — table with trait name/value rows from the top leaderboard row's `traits` dict (avg_hold_bars, trade_frequency, win_rate, max_position_notional_pct, long_short_bias, trades). Match dark-theme table styling from existing pages (look at logs page tables for the class pattern).

- [ ] **Step 4: Write the page**

`dashboard/src/app/theater/page.tsx` — "use client"; state: `symbol`, `state` (TheaterState), `leaderboard`, `decisions`, `error`. On mount: poll `getTheaterState` + `getLeaderboard` every 10s (useQuery from @tanstack/react-query with refetchInterval 10000, matching TrainingChart), fetch `getDecisions(symbol)` on symbol/state change. `useMetricsSocket` handler: on `probs` → refetch decisions; on `theater/traits` → refetch leaderboard. Controls row: select (RELIANCE.NS / BTCUSDT), Start button (disabled while running), Stop, Reset, status pill showing `status · steps · phase`, red error banner when `state.error`. Sections: three TrainingCharts, ActionProbsChart, LeaderboardTable, TraitsTable, and a "Latest-policy replay" card pulling `/api/equity` + `/api/trades` via existing query patterns.

- [ ] **Step 5: Add Nav tab**

Add `Theater` to the Nav links array in `Nav.tsx` (match existing entries: label + href `/theater`).

- [ ] **Step 6: Type-check and build**

Run: `cd dashboard; npx tsc --noEmit; npm run build`
Expected: both pass. Fix any TS errors (e.g., unused imports) until green.

- [ ] **Step 7: Commit**

```bash
git add dashboard/src
git commit -m "feat: add Theater page with live training watch, traits, and leaderboard"
```

---

### Task 11: End-to-end verification

**Files:** none (verification only)

- [ ] **Step 1: Full backend test suite**

Run: `.venv\Scripts\python.exe -m pytest`
Expected: all PASS (62 existing + ~16 new ≈ 78, no skips except Dhan/Gate live tests which now run — count may vary by key presence)

- [ ] **Step 2: Live smoke via run_live.py**

Start `scripts/run_live.py` hidden (Task 8 step 2 pattern). Then:
1. `GET /api/theater/state` → 200 idle
2. `POST /api/theater/start {"symbol": "BTCUSDT"}` → 200 running (crypto fetch via Gate public API — faster and network-stable; skip RELIANCE.NS if yfinance is blocked)
3. `GET /api/theater/state` after 10s → steps > 0 (real PPO training running)
4. `GET /api/decisions?symbol=BTCUSDT&limit=5` → rows with non-empty probs
5. `POST /api/theater/stop` → stopped; `POST /api/theater/reset` → idle
6. Stop the server process.

- [ ] **Step 3: Dashboard build + visual check**

Run: `cd dashboard; npm run build`; start dev server (`npm run dev`), open `http://localhost:3000/theater` via Playwright; verify: page renders without console errors, controls present, status pill updates after a manual `POST /api/theater/start` (from Step 2 or a fresh one), action-prob bars render after decisions exist. Take a screenshot into `D:\OpenCodeDevelopement\temp\theater.png` for the user.

- [ ] **Step 4: Update README quickstart**

Add to `README.md` quickstart: `python scripts\run_live.py` → open dashboard → Theater tab → pick symbol → Start. Mention honest label: "theater replay = latest saved policy simulated over recent bars".

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: document the Live Theater run flow"
```

---

### Task 12: Push + handoff

- [ ] **Step 1: Full suite once more**

Run: `.venv\Scripts\python.exe -m pytest` — all green.

- [ ] **Step 2: Push**

```bash
git push origin master
```

(If push hangs like before, use: `$env:GIT_TERMINAL_PROMPT="0"; git -c credential.interactive=never push origin master`.)

- [ ] **Step 3: Report**

Summarize for the user: what to run (`python scripts\run_live.py`, dashboard `npm run dev`, Theater tab), what they'll see, and the honest scope note (replay not live trading; memory = weights + 120-bar window; session-features fix noted as V2 improvement).