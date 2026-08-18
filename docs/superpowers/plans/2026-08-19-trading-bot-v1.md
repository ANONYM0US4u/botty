# Trading Bot V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build V1 of a self-learning intraday trading bot — data pipeline, RL training, paper-trading loop, and feature-rich monitoring dashboard — that runs on the user's PC.

**Architecture:** Python engine (Polars data layer, gymnasium RL env, stable-baselines3 PPO, pluggable BrokerAdapters, FastAPI + WebSocket API) + Next.js 14 dashboard. All subsystems behind narrow interfaces; append-only SQLite ledger; config-driven.

**Tech Stack:** Python 3.11+, gymnasium, stable-baselines3, polars, ccxt, yfinance, DhanHQ-py, fastapi, pydantic v2, uvicorn, structlog, pytest; Next.js 14, TypeScript, TanStack Query v5, Recharts.

**Spec:** `docs/superpowers/specs/2026-08-19-trading-bot-design.md`

## Global Constraints

- Python >= 3.11; engine deps pinned in `pyproject.toml`.
- All data flows through `engine.data.store.DataStore` (SQLite WAL + parquet); append-only ledger.
- Every order decision passes through `engine.live.risk.RiskGateway` — which is the ONLY component holding the broker; no exceptions.
- No credentials in code: `config/.env` only (gitignored); `config.yaml` holds non-secret settings.
- Indicators computed with Polars only (no pandas in engine).
- Tests: pytest; every module ships with tests; deterministic seeds everywhere (repeatable rollouts).
- GPL/AGPL reference repos in `references/` are READ-ONLY — never copy code from them into `engine/` or `dashboard/`.
- Commit after every task (conventional: `feat:`, `test:`, `fix:`).

---

### Task 1: Project Scaffolding + Config

**Files:**
- Create: `pyproject.toml`, `config/config.yaml`, `config/.env.example`, `.gitignore`, `engine/__init__.py`, `engine/config.py`, `engine/__main__.py`, empty `__init__.py` for `engine/{data,env,agents,brokers,live,eval,api}`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `engine.config.load_config(path="config/config.yaml") -> dict`; `engine.config.get_secret(name) -> str` (reads `.env` via python-dotenv); package layout `engine.{data,env,agents,brokers,live,eval,api}`.

- [ ] **Step 1: Verify environment**

Run: `python --version`
Expected: Python 3.11+ printed. If missing, install Python 3.11+ before proceeding.

- [ ] **Step 2: Write the failing test**

`tests/test_config.py`:
```python
from engine.config import load_config, get_secret

def test_load_config_defaults():
    cfg = load_config()
    assert cfg["market"]["timeframe_minutes"] == 5
    assert cfg["risk"]["daily_loss_limit_pct"] == -3.0
    assert cfg["instruments"]["stocks"] == ["RELIANCE.NS"]
    assert cfg["instruments"]["crypto"] == ["BTCUSDT"]

def test_get_secret_missing_returns_empty():
    assert get_secret("NONEXISTENT_KEY_XYZ") == ""
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: engine`.

- [ ] **Step 4: Create scaffolding files**

`pyproject.toml`:
```toml
[project]
name = "trading-bot-engine"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "gymnasium>=1.0", "stable-baselines3>=2.3", "polars>=1.0",
  "ccxt>=4.0", "yfinance>=0.2.40", "dhanhq>=2.3.0", "python-dotenv>=1.0",
  "fastapi>=0.115", "uvicorn[standard]>=0.30", "pydantic>=2.7",
  "structlog>=24.1", "pyyaml>=6.0", "quantstats>=0.0.62", "numpy>=1.26",
  "optuna>=3.6"
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-cov>=5.0"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["engine*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"
```

`config/config.yaml`:
```yaml
market:
  timeframe_minutes: 5
  # Data availability is NOT assumed: Task 4 probe determines the real
  # intraday history span (yfinance 5m = ~60 days; Dhan historical API
  # typically 1-2 years). train_years below are defaults that the probe
  # may shrink — walk-forward must never use bars older than the probe proves.
  history_years: 2

instruments:
  stocks: ["RELIANCE.NS"]
  crypto: ["BTCUSDT"]
  # AMENDMENT: these run as SEPARATE environments/runs — PPO #1 → RELIANCE.NS,
  # PPO #2 → BTCUSDT. One instrument per env, one policy per instrument,
  # separate checkpoints/eval. Never a joint action space.

training:
  seed: 42
  total_timesteps: 200_000
  eval_episodes: 10
  train_years: [2024, 2025]      # defaults; overridden by Task 4 probe result
  validate_years: [2026]
  window_bars: 120
  # Experiment tracking: every run gets run_id/model_id; checkpoints store
  # git commit + config hash so a trained model is fully reproducible.
  tracking:
    run_id: ""                   # auto: timestamp + short hash
    model_id: ""                 # auto: <symbol>_<run_id>

reward:
  # Per-term weights, each term logged separately in info["reward_terms"].
  equity_delta_weight: 1.0
  cost_weight: 1.0
  drawdown_penalty: 0.1
  holding_penalty: 0.0           # 0 initially; raise only after V1 baseline

costs:
  # Per-instrument cost model (per leg, fraction of notional + fixed):
  # brokerage, STT, exchange txn charge, stamp duty, SEBI fee, GST on fees,
  # slippage, spread. Different instruments get different values.
  stocks:
    brokerage_pct: 0.0           # broker flat is a fixed_charge
    fixed_charge: 20.0
    stt_pct: 0.001               # 0.1% intraday sell side
    exchange_txn_pct: 0.0000325  # NSE equity
    stamp_duty_pct: 0.00003      # buy side
    sebi_pct: 0.000001
    gst_pct: 0.18
    slippage_bps: 2.0
  crypto:
    taker_fee_pct: 0.00055       # Bybit taker
    slippage_bps: 2.0

risk:
  daily_loss_limit_pct: -3.0
  max_position_pct: 30.0
  max_total_exposure_pct: 90.0
  stale_data_seconds: 120
  flatten_at: "15:15"            # NSE intraday: force flat before market close
  # RiskGateway is the ONLY component that may call broker.place_order.
  # Scheduler/policy/API must go through RiskGateway.execute_order().

brokers:
  active: simulator
  slippage_bps: 2
  latency_bars: 1

api:
  host: 127.0.0.1
  port: 8000

storage:
  db_path: data/trading.db
  parquet_dir: data/parquet
  # parquet = immutable historical bars; sqlite = operational ledger
  # (orders, fills, decisions, equity, metrics, checkpoints). Never mixed.
  checkpoint_dir: checkpoints
```

`config/.env.example`:
```
DHAN_CLIENT_ID=
DHAN_ACCESS_TOKEN=
BYBIT_TESTNET_API_KEY=
BYBIT_TESTNET_API_SECRET=
```

`.gitignore`:
```
__pycache__/
*.pyc
.env
data/
checkpoints/
node_modules/
.next/
```

Create empty `engine/__init__.py` plus subpackage `__init__.py` files in `engine/data/`, `engine/env/`, `engine/agents/`, `engine/brokers/`, `engine/live/`, `engine/eval/`, `engine/api/`.

- [ ] **Step 5: Implement config module**

`engine/config.py`:
```python
from pathlib import Path
import os
import yaml
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent.parent

def load_config(path: str | Path = "config/config.yaml") -> dict:
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = _ROOT / cfg_path
    with open(cfg_path) as f:
        return yaml.safe_load(f)

def get_secret(name: str) -> str:
    load_dotenv(_ROOT / "config" / ".env")
    return os.environ.get(name, "")
```

`engine/__main__.py`:
```python
from engine.config import load_config

def main() -> None:
    cfg = load_config()
    print(f"Trading bot engine ready. Instruments: {cfg['instruments']}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (2 passed).

- [ ] **Step 7: Install deps**

Run: `pip install -e ".[dev]"`
Expected: install completes without error.

- [ ] **Step 8: Commit**

```bash
git init
git add -A
git commit -m "feat: scaffold engine project, config, and tests"
```

---

### Task 2: DataStore (SQLite + parquet)

**Files:**
- Create: `engine/data/store.py`
- Test: `tests/data/test_store.py`

**Interfaces:**
- Consumes: nothing (stdlib `sqlite3`, `polars`).
- Produces: `engine.data.store.DataStore(db_path: str | Path, parquet_dir: str | Path)` with:
  - `init_schema() -> None`
  - `save_bars(symbol: str, df: pl.DataFrame, interval_minutes: int) -> None` (df cols: `time, open, high, low, close, volume`)
  - `get_bars(symbol: str, start: str | None = None, end: str | None = None) -> pl.DataFrame`
  - `append_order(order: dict) -> None`
  - `append_fill(fill: dict) -> None`
  - `append_equity(ts: str, equity: float) -> None`
  - `get_equity() -> list[dict]`
  - `get_trades() -> list[dict]`
  - `append_checkpoint(meta: dict) -> None`
  - `get_checkpoints() -> list[dict]`
  - `append_decision(rec: dict) -> None`
  - `append_metric(name: str, value: float, ts: str) -> None`
  - `get_metrics(name: str, since: str | None = None) -> list[dict]`

- [ ] **Step 1: Write the failing test**

`tests/data/test_store.py`:
```python
import polars as pl
import pytest
from engine.data.store import DataStore

@pytest.fixture
def store(tmp_path):
    s = DataStore(tmp_path / "t.db", tmp_path / "parquet")
    s.init_schema()
    return s

def test_bars_roundtrip(store):
    df = pl.DataFrame({
        "time": ["2026-01-02 09:15:00", "2026-01-02 09:20:00"],
        "open": [100.0, 101.0], "high": [101.5, 102.0],
        "low": [99.5, 100.5], "close": [101.0, 101.5], "volume": [1000, 1200],
    })
    store.save_bars("RELIANCE.NS", df, 5)
    out = store.get_bars("RELIANCE.NS")
    assert out.height == 2
    assert out["close"].to_list() == [101.0, 101.5]

def test_append_and_read_ledger(store):
    store.append_order({"id": "o1", "symbol": "RELIANCE.NS", "side": "buy", "qty": 10, "price": 100.0, "status": "pending"})
    store.append_fill({"order_id": "o1", "symbol": "RELIANCE.NS", "side": "buy", "qty": 10, "price": 100.0, "ts": "2026-01-02 09:21:00"})
    store.append_equity("2026-01-02 09:25:00", 100_000.0)
    trades = store.get_trades()
    assert len(trades) == 1 and trades[0]["order_id"] == "o1"
    assert len(store.get_equity()) == 1

def test_checkpoints_and_metrics(store):
    store.append_checkpoint({"path": "checkpoints/ppo_42.zip", "reward": 1.5, "sharpe": 1.2})
    store.append_metric("reward", 1.5, "2026-01-02 09:25:00")
    store.append_metric("reward", 2.0, "2026-01-02 09:30:00")
    assert len(store.get_checkpoints()) == 1
    assert len(store.get_metrics("reward")) == 2
    assert len(store.get_metrics("reward", since="2026-01-02 09:28:00")) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_store.py -v`
Expected: FAIL — `ModuleNotFoundError: engine.data.store`.

- [ ] **Step 3: Write minimal implementation**

`engine/data/store.py`:
```python
import sqlite3
from pathlib import Path
import polars as pl

_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
  id TEXT PRIMARY KEY, symbol TEXT, side TEXT, qty REAL, price REAL,
  status TEXT, ts TEXT
);
CREATE TABLE IF NOT EXISTS fills (
  order_id TEXT, symbol TEXT, side TEXT, qty REAL, price REAL, ts TEXT
);
CREATE TABLE IF NOT EXISTS equity_curve (
  ts TEXT PRIMARY KEY, equity REAL
);
CREATE TABLE IF NOT EXISTS checkpoints (
  path TEXT PRIMARY KEY, reward REAL, sharpe REAL, ts TEXT
);
CREATE TABLE IF NOT EXISTS decisions (
  ts TEXT, symbol TEXT, action TEXT, probs TEXT, features TEXT, attribution TEXT
);
CREATE TABLE IF NOT EXISTS metrics (
  name TEXT, value REAL, ts TEXT
);
CREATE INDEX IF NOT EXISTS idx_metrics_name_ts ON metrics(name, ts);
"""

class DataStore:
    def __init__(self, db_path: str | Path, parquet_dir: str | Path):
        self.db_path = Path(db_path)
        self.parquet_dir = Path(parquet_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def save_bars(self, symbol: str, df: pl.DataFrame, interval_minutes: int) -> None:
        safe = symbol.replace("/", "_")
        df.write_parquet(self.parquet_dir / f"{safe}_{interval_minutes}m.parquet")

    def get_bars(self, symbol: str, start: str | None = None, end: str | None = None) -> pl.DataFrame:
        safe = symbol.replace("/", "_")
        files = list(self.parquet_dir.glob(f"{safe}_*m.parquet"))
        if not files:
            return pl.DataFrame()
        df = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
        if start:
            df = df.filter(pl.col("time") >= start)
        if end:
            df = df.filter(pl.col("time") <= end)
        return df.sort("time")

    def append_order(self, order: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO orders VALUES (:id,:symbol,:side,:qty,:price,:status,:ts)",
                order)

    def append_fill(self, fill: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO fills VALUES (:order_id,:symbol,:side,:qty,:price,:ts)",
                fill)

    def append_equity(self, ts: str, equity: float) -> None:
        with self._conn() as conn:
            conn.execute("INSERT OR REPLACE INTO equity_curve VALUES (?,?)", (ts, equity))

    def get_equity(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT ts, equity FROM equity_curve ORDER BY ts").fetchall()
        return [{"ts": r[0], "equity": r[1]} for r in rows]

    def get_trades(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM fills ORDER BY ts").fetchall()
        cols = ["order_id", "symbol", "side", "qty", "price", "ts"]
        return [dict(zip(cols, r)) for r in rows]

    def append_checkpoint(self, meta: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO checkpoints VALUES (:path,:reward,:sharpe,:ts)",
                meta)

    def get_checkpoints(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM checkpoints ORDER BY ts DESC").fetchall()
        cols = ["path", "reward", "sharpe", "ts"]
        return [dict(zip(cols, r)) for r in rows]

    def append_decision(self, rec: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO decisions VALUES (:ts,:symbol,:action,:probs,:features,:attribution)",
                rec)

    def append_metric(self, name: str, value: float, ts: str) -> None:
        with self._conn() as conn:
            conn.execute("INSERT INTO metrics VALUES (?,?,?)", (name, value, ts))

    def get_metrics(self, name: str, since: str | None = None) -> list[dict]:
        with self._conn() as conn:
            if since:
                rows = conn.execute(
                    "SELECT name, value, ts FROM metrics WHERE name=? AND ts>=? ORDER BY ts",
                    (name, since)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT name, value, ts FROM metrics WHERE name=? ORDER BY ts",
                    (name,)).fetchall()
        return [{"name": r[0], "value": r[1], "ts": r[2]} for r in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_store.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/data/test_store.py engine/data/store.py engine/data/__init__.py
git commit -m "feat: add DataStore with SQLite ledger and parquet bars"
```

### Task 3: Indicators (Polars)

**Files:**
- Create: `engine/data/indicators.py`
- Test: `tests/data/test_indicators.py`

**Interfaces:**
- Consumes: `DataStore.get_bars` output (cols `time, open, high, low, close, volume`).
- Produces: `engine.data.indicators.add_indicators(df: pl.DataFrame) -> pl.DataFrame` — appends `ema9, ema21, rsi14, atr14, vwap, session_band, ret1, vol20`. NaN-safe: first `window` rows carry nulls.

- [ ] **Step 1: Write the failing test**

`tests/data/test_indicators.py`:
```python
import polars as pl
import math
from engine.data.indicators import add_indicators

def _df():
    rows = []
    for i in range(50):
        rows.append({"time": f"2026-01-02 09:{15+i:02d}:00", "open": 100.0 + i,
                     "high": 101.0 + i, "low": 99.0 + i, "close": 100.5 + i,
                     "volume": 1000.0})
    return pl.DataFrame(rows)

def test_columns_added():
    out = add_indicators(_df())
    for col in ["ema9", "ema21", "rsi14", "atr14", "vwap", "session_band", "ret1", "vol20"]:
        assert col in out.columns

def test_known_rsi():
    # Constant up-trend: RSI should approach 100.
    out = add_indicators(_df())
    rsi = out["rsi14"].tail(1).item()
    assert rsi > 80.0

def test_atr_positive():
    out = add_indicators(_df())
    assert out["atr14"].drop_nulls().min() > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_indicators.py -v`
Expected: FAIL — `ModuleNotFoundError: engine.data.indicators`.

- [ ] **Step 3: Write minimal implementation**

`engine/data/indicators.py`:
```python
import polars as pl

def _ewm(series: pl.Series, span: int) -> pl.Series:
    alpha = 2.0 / (span + 1.0)
    vals = series.to_list()
    out = []
    prev = None
    for v in vals:
        if v is None or v != v:
            out.append(None)
            continue
        if prev is None:
            prev = v
        else:
            prev = alpha * v + (1 - alpha) * prev
        out.append(prev)
    return pl.Series(out)

def _rsi(close: pl.Series, period: int = 14) -> pl.Series:
    diff = close.diff().fill_null(0.0)
    gain = diff.map_elements(lambda x: max(x, 0.0), return_dtype=pl.Float64)
    loss = diff.map_elements(lambda x: max(-x, 0.0), return_dtype=pl.Float64)
    avg_gain = _ewm(gain, period)
    avg_loss = _ewm(loss, period)
    rs = avg_gain / avg_loss.replace(0.0, 1e-12)
    return (100.0 - 100.0 / (1.0 + rs)).replace(float("inf"), 100.0)

def add_indicators(df: pl.DataFrame) -> pl.DataFrame:
    out = df.with_columns([
        _ewm(pl.col("close"), 9).alias("ema9"),
        _ewm(pl.col("close"), 21).alias("ema21"),
        _rsi(pl.col("close")).alias("rsi14"),
    ])
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - pl.col("close").shift(1)).abs(),
        (pl.col("low") - pl.col("close").shift(1)).abs(),
    ).fill_null(0.0)
    out = out.with_columns([
        _ewm(tr, 14).alias("atr14"),
        pl.col("volume") * (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0,
    ])
    out = out.with_columns([
        (pl.col("volume").cum_sum() / 3.0).alias("_vsum"),
        (pl.col("typical_price").cum_sum()).alias("_tpsum"),
    ])
    out = out.with_columns((pl.col("_tpsum") / pl.col("_vsum")).alias("vwap"))
    out = out.with_columns([
        (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("ret1"),
        (pl.col("close").rolling_std(20)).alias("vol20"),
        (pl.col("high").rolling_max(30) - pl.col("low").rolling_min(30)).alias("session_band"),
    ])
    return out.drop(["_vsum", "_tpsum"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_indicators.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/data/test_indicators.py engine/data/indicators.py
git commit -m "feat: add Polars indicator suite (EMA/RSI/ATR/VWAP/vol)"
```

---

### Task 4: NSE Fetcher (yfinance + fallback)

**Files:**
- Create: `engine/data/fetchers/yfinance_nse.py`
- Test: `tests/data/test_yfinance_nse.py`

**Data availability gate (AMENDMENT — must run before fixing train years):**
- yfinance intraday limits: 1m = 30 days, 5m/15m = 60 days max. It CANNOT backfill 2019–2023.
- Probe script `engine/data/fetchers/probe_data_availability.py`: for each configured symbol, fetch the maximum reachable 5m history from yfinance, record `{symbol, source, first_ts, last_ts, bars}` into `config/data_availability.yaml`.
- Fallback chain for longer intraday history: Dhan historical intraday API (verified at Task 10 time) → NSE EOD bars for walk-forward context. The training window in config is then derived from the probe output; never assume a window the probe did not prove.
- Test `tests/data/test_probe_availability.py`: asserts the probe returns a sane `first_ts < last_ts` and writes the yaml. Real-network reach is verified by running the probe once (documented in README, not in CI).

**Dataset validation gate (AMENDMENT — new, must exist before Task 8 training):**
- Create: `engine/data/validate.py` — `validate_dataset(bars: pl.DataFrame, symbol: str, expected_interval_minutes: int) -> ValidationReport`.
- ValidationReport = list of checks, each `{check: str, passed: bool, detail: str}`:
  1. `range_exists` — requested date range (from `config/data_availability.yaml`) fully covered
  2. `timeframe_exists` — expected bar interval present
  3. `no_duplicate_timestamps` — unique `time`
  4. `no_future_timestamps` — max time <= now + small skew
  5. `no_unexpected_gaps` — no gap > 2× interval within a session
  6. `ohlc_valid` — high >= max(open, close), low <= min(open, close), all prices > 0
  7. `timezone_normalized` — tz-naive IST throughout
  8. `corporate_action_policy_applied` — `corporate_actions` config section recorded and applied (split/bonus adjusted), else check reports `skipped` with reason
- `is_valid() -> bool` — ALL checks must pass (or pass-or-skipped for #8).
- Test `tests/data/test_validate.py`: each check has a corrupt-data case that fails it and a clean case that passes.
- **Gate enforcement**: `train_ppo` (Task 8) refuses to run unless `validate_dataset(...).is_valid()` — `DATASET INVALID → TRAINING BLOCKED` with the failed checks printed. The e2e smoke (Task 19) and the README both state the gate.

- Create: `engine/data/fetchers/yfinance_nse.py`
- Test: `tests/data/test_yfinance_nse.py`

**Interfaces:**
- Consumes: nothing directly.
- Produces: `engine.data.fetchers.yfinance_nse.fetch_nse_minute_bars(symbol: str, start: str, end: str, interval: str = "5m") -> pl.DataFrame` (cols `time, open, high, low, close, volume`, tz-naive IST timestamps). Raises `ValueError` if the fetch returns no data.

- [ ] **Step 1: Write the failing test**

`tests/data/test_yfinance_nse.py`:
```python
import polars as pl
import pytest
from engine.data.fetchers.yfinance_nse import fetch_nse_minute_bars

def test_empty_result_raises(monkeypatch):
    class FakeTicker:
        def history(self, **kw):
            return __import__("pandas").DataFrame()
    monkeypatch.setattr("yfinance.Ticker", lambda s: FakeTicker())
    with pytest.raises(ValueError):
        fetch_nse_minute_bars("RELIANCE.NS", "2026-01-02", "2026-01-03")

def test_normalizes_columns(monkeypatch):
    import pandas as pd
    df = pd.DataFrame({
        "Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [100.5],
        "Volume": [1000],
    }, index=pd.to_datetime(["2026-01-02 09:15:00+05:30"]))
    class FakeTicker:
        def history(self, **kw):
            return df
    monkeypatch.setattr("yfinance.Ticker", lambda s: FakeTicker())
    out = fetch_nse_minute_bars("RELIANCE.NS", "2026-01-02", "2026-01-03")
    assert isinstance(out, pl.DataFrame)
    assert out.columns == ["time", "open", "high", "low", "close", "volume"]
    assert str(out["time"][0]).startswith("2026-01-02 09:15")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_yfinance_nse.py -v`
Expected: FAIL — `ModuleNotFoundError: engine.data.fetchers`.

- [ ] **Step 3: Write minimal implementation**

Create `engine/data/fetchers/__init__.py` (empty) and `engine/data/fetchers/yfinance_nse.py`:
```python
import polars as pl
import pandas as pd
import yfinance as yf


def fetch_nse_minute_bars(symbol: str, start: str, end: str, interval: str = "5m") -> pl.DataFrame:
    raw = yf.Ticker(symbol).history(start=start, end=end, interval=interval, auto_adjust=False)
    if raw is None or raw.empty:
        raise ValueError(f"no data for {symbol}")
    df = raw.reset_index()[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    if df["time"].dt.tz is not None:
        df["time"] = df["time"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return pl.from_pandas(df)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_yfinance_nse.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/data/test_yfinance_nse.py engine/data/fetchers/
git commit -m "feat: add NSE minute-bar fetcher via yfinance"
```

---

### Task 5: Crypto Fetcher (CCXT mainnet OHLCV)

**Files:**
- Create: `engine/data/fetchers/ccxt_crypto.py`
- Test: `tests/data/test_ccxt_crypto.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `engine.data.fetchers.ccxt_crypto.fetch_crypto_bars(symbol: str, start: str, end: str, interval: str = "5m", exchange_id: str = "bybit") -> pl.DataFrame` (same column contract as Task 4, UTC→IST tz-naive). Raises `ValueError` on empty.

- [ ] **Step 1: Write the failing test**

`tests/data/test_ccxt_crypto.py`:
```python
import polars as pl
import pytest
from engine.data.fetchers.ccxt_crypto import fetch_crypto_bars

def test_empty_raises(monkeypatch):
    class FakeEx:
        def fetch_ohlcv(self, symbol, tf, since, limit):
            return []
    monkeypatch.setattr("engine.data.fetchers.ccxt_crypto.ccxt", None)
    monkeypatch.setattr("ccxt.bybit", lambda x: FakeEx())
    with pytest.raises(ValueError):
        fetch_crypto_bars("BTC/USDT:USDT", "2026-01-01", "2026-01-02")

def test_normalizes(monkeypatch):
    import time as _t
    ts = int(_t.mktime(_t.strptime("2026-01-02 09:15:00", "%Y-%m-%d %H:%M:%S"))) * 1000
    class FakeEx:
        def fetch_ohlcv(self, symbol, tf, since, limit):
            return [[ts, 100.0, 101.0, 99.0, 100.5, 1000.0]]
    monkeypatch.setattr("ccxt.bybit", lambda x: FakeEx())
    out = fetch_crypto_bars("BTC/USDT:USDT", "2026-01-01", "2026-01-02")
    assert isinstance(out, pl.DataFrame)
    assert out["close"][0] == 100.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_ccxt_crypto.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

`engine/data/fetchers/ccxt_crypto.py`:
```python
import time
import polars as pl
import ccxt


def fetch_crypto_bars(symbol: str, start: str, end: str, interval: str = "5m",
                      exchange_id: str = "bybit") -> pl.DataFrame:
    exch_cls = getattr(ccxt, exchange_id)
    ex = exch_cls({"enableRateLimit": True})
    tf_sec = {"1m": 60, "5m": 300, "15m": 900}[interval]
    since = int(time.mktime(time.strptime(start, "%Y-%m-%d"))) * 1000
    until = int(time.mktime(time.strptime(end, "%Y-%m-%d"))) * 1000
    rows = []
    cur = since
    while cur < until:
        batch = ex.fetch_ohlcv(symbol, interval, since=cur, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        cur = batch[-1][0] + tf_sec * 1000
        if len(batch) < 1000:
            break
    if not rows:
        raise ValueError(f"no data for {symbol}")
    df = pl.DataFrame(
        {"time": [time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(r[0] / 1000)) for r in rows],
         "open": [r[1] for r in rows], "high": [r[2] for r in rows],
         "low": [r[3] for r in rows], "close": [r[4] for r in rows],
         "volume": [r[5] for r in rows]}).unique(subset=["time"])
    return df.sort("time")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_ccxt_crypto.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/data/test_ccxt_crypto.py engine/data/fetchers/ccxt_crypto.py
git commit -m "feat: add crypto OHLCV fetcher via CCXT"
```

### Task 6: TradingEnv (gymnasium)

**Files:**
- Create: `engine/env/trading_env.py`
- Test: `tests/env/test_trading_env.py`

**Interfaces:**
- Consumes: `DataStore.get_bars` data (featured via `add_indicators`), config dict.
- Produces: `engine.env.trading_env.TradingEnv(symbol: str, bars: pl.DataFrame, initial_cash: float = 100_000.0, cost_pct: float = 0.001, window: int = 120, seed: int = 42, holding_penalty: float = 0.0) -> gymnasium.Env`:
  - **AMENDMENT: V1 is SINGLE instrument per env.** No multi-symbol 3^n encoding, no `base_repr` padding hacks. Multi-symbol trading is a separate future env.
  - `observation_space`: `Box(-inf, inf, shape=(window * n_features + 2,))` — flat window of causal features + position + cash ratio.
  - `action_space`: `Discrete(3)` — `0=flat, 1=long, 2=short`.
  - `reset() -> obs`; `step(action) -> (obs, reward, terminated, truncated, info)`; `render()` no-op.
  - **Reward = Δequity − cost_pct·turnover − drawdown_penalty·dd − holding_penalty·|position|**, where Δequity = `equity_now − equity_prev` (mark-to-market change, NOT realized P&L).
  - **Every reward term is observable**: `info["reward_terms"] = {"equity_delta": float, "cost": float, "drawdown": float, "holding": float}` — logged per step in the ledger and plottable on the dashboard.
  - **Causal-feature invariant (AMENDMENT)**: features at index `i` must be computed ONLY from bars `[0..i]`. `add_indicators` must be applied per-row causally (shifted windows, no centered rolling, no `shift(-1)`); a dedicated test asserts the observation at index `i` is unchanged when bars after `i` are dropped.

- [ ] **Step 1: Write the failing test**

`tests/env/test_trading_env.py`:
```python
import polars as pl
import numpy as np
from engine.env.trading_env import TradingEnv
from engine.data.indicators import add_indicators

def _bars(n=400):
    rows = [{"time": f"2026-01-02 09:{15+i:02d}:00", "open": 100.0 + i * 0.1,
             "high": 101.0 + i * 0.1, "low": 99.0 + i * 0.1,
             "close": 100.5 + i * 0.1, "volume": 1000.0} for i in range(n)]
    return add_indicators(pl.DataFrame(rows))

def test_reset_shape_and_seed_determinism():
    env1 = TradingEnv("RELIANCE.NS", _bars(), seed=7)
    env2 = TradingEnv("RELIANCE.NS", _bars(), seed=7)
    o1, _ = env1.reset()
    o2, _ = env2.reset()
    assert o1.shape == o2.shape
    assert np.allclose(o1, o2)

def test_action_space_is_discrete_3():
    env = TradingEnv("RELIANCE.NS", _bars(), seed=1)
    assert env.action_space.n == 3
    env.reset()
    obs, reward, terminated, truncated, info = env.step(1)  # long
    assert not terminated
    assert isinstance(reward, float)
    assert info["equity"] > 0

def test_reward_terms_are_observable():
    env = TradingEnv("RELIANCE.NS", _bars(), seed=2, cost_pct=0.001)
    env.reset()
    _, reward, _, _, info = env.step(2)  # short
    terms = info["reward_terms"]
    assert set(terms) == {"equity_delta", "cost", "drawdown", "holding"}
    assert abs(reward - (terms["equity_delta"] - terms["cost"]
                         - terms["drawdown"] - terms["holding"])) < 1e-9

def test_causal_feature_invariant():
    # Observation at index i must not change if later bars are removed.
    full = _bars()
    env_full = TradingEnv("RELIANCE.NS", full, seed=3)
    env_full.reset()
    truncated = _bars(300)  # same bars, cut short
    env_trunc = TradingEnv("RELIANCE.NS", truncated, seed=3)
    env_trunc.reset()
    assert np.allclose(env_full._obs(), env_trunc._obs())

def test_long_profits_in_uptrend():
    bars = _bars()
    env = TradingEnv("RELIANCE.NS", bars, seed=2, cost_pct=0.0)
    env.reset()
    for _ in range(50):
        env.step(1)
    assert env.equity > env.initial_cash
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/env/test_trading_env.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

`engine/env/trading_env.py`:
```python
import gymnasium as gym
import numpy as np
import polars as pl
from gymnasium import spaces

_FEATURES = ["open", "high", "low", "close", "volume",
             "ema9", "ema21", "rsi14", "atr14", "vwap", "session_band", "ret1", "vol20"]


class TradingEnv(gym.Env):
    def __init__(self, symbol, bars, initial_cash=100_000.0, cost_pct=0.001,
                 window=120, seed=42, holding_penalty=0.0):
        super().__init__()
        self.symbol = symbol
        self.bars = bars.fill_null(strategy="forward")
        self.initial_cash = float(initial_cash)
        self.cost_pct = cost_pct
        self.window = window
        self.holding_penalty = holding_penalty
        self.n_features = len(_FEATURES)
        self._rng = np.random.default_rng(seed)
        self.action_space = spaces.Discrete(3)  # 0=flat, 1=long, 2=short
        flat_dim = window * self.n_features + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(flat_dim,), dtype=np.float32)
        self.reset(seed=seed)

    def _obs(self) -> np.ndarray:
        arr = self.bars.select(_FEATURES).to_numpy()
        start = max(0, self._idx - self.window + 1)
        chunk = arr[start:self._idx + 1]
        if len(chunk) < self.window:
            pad = np.zeros((self.window - len(chunk), self.n_features))
            chunk = np.vstack([pad, chunk])
        flat = chunk.reshape(-1)
        pos = np.array([self.position], dtype=np.float32)
        cash_ratio = np.array([self.cash / self.initial_cash], dtype=np.float32)
        return np.concatenate([flat, pos, cash_ratio]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._idx = self.window
        self.cash = self.initial_cash
        self.position = 0.0
        self.equity = self.initial_cash
        self._peak = self.initial_cash
        return self._obs(), {}

    def _price(self):
        return float(self.bars["close"][self._idx])

    def _equity(self):
        return self.cash + self.position * self._price()

    def step(self, action):
        target = {0: 0.0, 1: 1.0, 2: -1.0}[int(action)]
        turnover = 0.0
        if target != self.position:
            price = self._price()
            delta = target - self.position
            turnover = abs(delta) * self.equity * self.cost_pct
            if target > 0:
                self.cash -= target * price
            elif target < 0:
                self.cash += abs(target) * price
            else:
                self.cash += self.position * price
            self.position = target
        self._idx += 1
        prev = self.equity
        self.equity = self._equity()
        self._peak = max(self._peak, self.equity)
        dd = (self._peak - self.equity) / self._peak
        equity_delta = self.equity - prev
        cost_term = turnover
        dd_term = 0.1 * dd
        hold_term = self.holding_penalty * abs(self.position)
        reward = equity_delta - cost_term - dd_term - hold_term
        terminated = self._idx >= len(self.bars) - 1
        info = {"equity": self.equity,
                "reward_terms": {"equity_delta": equity_delta, "cost": cost_term,
                                 "drawdown": dd_term, "holding": hold_term}}
        return self._obs(), float(reward), terminated, False, info
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/env/test_trading_env.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/env/test_trading_env.py engine/env/trading_env.py
git commit -m "feat: add single-instrument TradingEnv with observable reward terms"
```

---

### Task 7: Evaluation Metrics

**Files:**
- Create: `engine/eval/metrics.py`
- Test: `tests/eval/test_metrics.py`

**Interfaces:**
- Consumes: equity series + trade list.
- Produces:
  - `sharpe_ratio(equity: list[float], periods_per_year: int = 252 * 75) -> float`
  - `max_drawdown(equity: list[float]) -> float` (negative fraction)
  - `win_rate(trades: list[dict]) -> float` — pairs buy/sell fills by FIFO; needs `pnl` computed via `realized_pnl(trades) -> list[dict]` (returns trades with `pnl` key)
  - `profit_factor(trades_with_pnl: list[dict]) -> float`
  - `compute_eval_report(equity: list[float], trades: list[dict], bench_equity: list[float] | None = None) -> dict` — keys: `sharpe, max_drawdown, win_rate, profit_factor, total_return, vs_benchmark_return`.

- [ ] **Step 1: Write the failing test**

`tests/eval/test_metrics.py`:
```python
import math
from engine.eval.metrics import (sharpe_ratio, max_drawdown, win_rate,
                                 profit_factor, compute_eval_report)

def test_sharpe_positive_for_uptrend():
    eq = [100_000 + i * 100 for i in range(100)]
    assert sharpe_ratio(eq) > 0

def test_max_drawdown():
    eq = [100_000, 110_000, 90_000, 100_000]
    assert math.isclose(max_drawdown(eq), -0.1818, abs_tol=1e-3)

def test_win_rate_and_profit_factor():
    trades = [
        {"order_id": "b1", "side": "buy", "qty": 10, "price": 100.0},
        {"order_id": "s1", "side": "sell", "qty": 10, "price": 110.0},
        {"order_id": "b2", "side": "buy", "qty": 10, "price": 100.0},
        {"order_id": "s2", "side": "sell", "qty": 10, "price": 90.0},
    ]
    assert win_rate(trades) == 0.5
    assert profit_factor(trades) == 1.0

def test_eval_report_shape():
    eq = [100_000, 101_000, 100_500]
    rep = compute_eval_report(eq, [])
    assert set(["sharpe", "max_drawdown", "win_rate", "profit_factor", "total_return"]) <= set(rep)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/eval/test_metrics.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

`engine/eval/metrics.py`:
```python
import math


def sharpe_ratio(equity: list[float], periods_per_year: int = 252 * 75) -> float:
    if len(equity) < 3:
        return 0.0
    returns = [equity[i] / equity[i - 1] - 1.0 for i in range(1, len(equity))]
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / len(returns)
    if var == 0:
        return 0.0
    return (mean / math.sqrt(var)) * math.sqrt(periods_per_year)


def max_drawdown(equity: list[float]) -> float:
    peak = -float("inf")
    mdd = 0.0
    for v in equity:
        peak = max(peak, v)
        mdd = min(mdd, (v - peak) / peak)
    return mdd


def _realized_pnl(trades: list[dict]) -> list[dict]:
    lots = []  # (qty, price)
    out = []
    for t in trades:
        if t["side"] == "buy":
            lots.append((t["qty"], t["price"]))
        else:
            qty, price = t["qty"], t["price"]
            while qty > 0 and lots:
                bq, bp = lots.pop(0)
                take = min(bq, qty)
                out.append({**t, "pnl": (price - bp) * take})
                qty -= take
                if bq > take:
                    lots.insert(0, (bq - take, bp))
    return out


def win_rate(trades: list[dict]) -> float:
    closed = _realized_pnl(trades)
    if not closed:
        return 0.0
    return sum(1 for t in closed if t["pnl"] > 0) / len(closed)


def profit_factor(trades: list[dict]) -> float:
    closed = _realized_pnl(trades)
    gross_win = sum(t["pnl"] for t in closed if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in closed if t["pnl"] < 0))
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return gross_win / gross_loss


def compute_eval_report(equity: list[float], trades: list[dict],
                        bench_equity: list[float] | None = None) -> dict:
    rep = {
        "sharpe": round(sharpe_ratio(equity), 3),
        "max_drawdown": round(max_drawdown(equity), 4),
        "win_rate": round(win_rate(trades), 3),
        "profit_factor": round(profit_factor(trades), 3),
        "total_return": round(equity[-1] / equity[0] - 1.0, 4) if len(equity) > 1 else 0.0,
    }
    if bench_equity and len(bench_equity) > 1:
        rep["vs_benchmark_return"] = round(equity[-1] / equity[0] - bench_equity[-1] / bench_equity[0], 4)
    return rep
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/eval/test_metrics.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/eval/test_metrics.py engine/eval/metrics.py
git commit -m "feat: add evaluation metrics (sharpe, dd, win rate, profit factor)"
```

---

### Task 8: PPO Trainer with Walk-Forward + Checkpoints

**Files:**
- Create: `engine/agents/ppo.py`
- Test: `tests/agents/test_ppo.py`

**Interfaces:**
- Consumes: `TradingEnv` (single instrument), `DataStore`, config dict.
- Produces:
  - `train_ppo(env: gym.Env, total_timesteps: int, checkpoint_dir: str | Path, seed: int, save_every: int = 50_000) -> Path` — trains, saves `ppo_{seed}_{timesteps}.zip` + returns last path; appends checkpoint meta to store when given.
  - `evaluate_ppo(model, env: gym.Env, episodes: int, seed: int) -> dict` — keys `mean_reward, mean_equity, equity_series`.
  - `load_policy(path: str | Path) -> PPO`.
- **Training gate (AMENDMENT)**: the training entry point (`python -m engine.agents.ppo`) loads bars from `DataStore`, runs `validate_dataset(...)` (Task 4), and refuses to train when invalid — prints the failed checks (`DATASET INVALID → TRAINING BLOCKED`). `train_ppo` itself stays validation-agnostic (it receives an already-validated env).

- [ ] **Step 1: Write the failing test**

`tests/agents/test_ppo.py`:
```python
import numpy as np
import polars as pl
from engine.agents.ppo import train_ppo, evaluate_ppo, load_policy
from engine.env.trading_env import TradingEnv
from engine.data.indicators import add_indicators

def _env(seed=3):
    rows = [{"time": f"2026-01-02 09:{15+i:02d}:00", "open": 100.0 + i * 0.05,
             "high": 101.0 + i * 0.05, "low": 99.0 + i * 0.05,
             "close": 100.5 + i * 0.05, "volume": 1000.0} for i in range(500)]
    bars = add_indicators(pl.DataFrame(rows))
    return TradingEnv("RELIANCE.NS", bars, seed=seed)

def test_train_and_evaluate(tmp_path):
    env = _env()
    path = train_ppo(env, total_timesteps=2_000, checkpoint_dir=tmp_path, seed=3)
    assert path.exists()
    model = load_policy(path)
    rep = evaluate_ppo(model, env, episodes=2, seed=3)
    assert set(["mean_reward", "mean_equity", "equity_series"]) <= set(rep)
    assert len(rep["equity_series"]) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/agents/test_ppo.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

`engine/agents/ppo.py`:
```python
from pathlib import Path
import hashlib
import json
import subprocess
import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


def _run_metadata(cfg_hash: str) -> dict:
    """Experiment tracking: run_id/model_id + git commit + config hash."""
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{hashlib.sha1(cfg_hash.encode()).hexdigest()[:6]}"
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = "unknown"
    return {"run_id": run_id, "model_id": f"ppo-{run_id}", "git_commit": commit,
            "config_hash": cfg_hash}


def train_ppo(env: gym.Env, total_timesteps: int, checkpoint_dir: str | Path,
              seed: int, save_every: int = 50_000, store=None,
              cfg: dict | None = None) -> Path:
    cdir = Path(checkpoint_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    cfg_hash = hashlib.sha1(json.dumps(cfg or {}, sort_keys=True).encode()).hexdigest()[:12]
    meta = _run_metadata(cfg_hash)
    (cdir / "run_meta.json").write_text(json.dumps(meta))
    model = PPO("MlpPolicy", env, seed=seed, verbose=0, n_steps=2048, batch_size=64)
    steps = 0
    last_path = None
    while steps < total_timesteps:
        chunk = min(save_every, total_timesteps - steps)
        model.learn(total_timesteps=chunk, progress_bar=False)
        steps += chunk
        last_path = cdir / f"ppo_{seed}_{steps}.zip"
        model.save(last_path)
        if store is not None:
            store.append_checkpoint({"path": str(last_path), "reward": 0.0,
                                     "sharpe": 0.0, "ts": steps, **meta})
    return last_path


def evaluate_ppo(model, env: gym.Env, episodes: int, seed: int) -> dict:
    rewards, equities = [], []
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        done = False
        total = 0.0
        while not done:
            obs = env.unwrapped._obs()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            total += reward
        rewards.append(total)
        equities.append(env.unwrapped.equity)
    return {"mean_reward": float(np.mean(rewards)),
            "mean_equity": float(np.mean(equities)),
            "equity_series": [env.unwrapped.initial_cash] + [e for e in equities]}


def load_policy(path: str | Path) -> PPO:
    return PPO.load(str(path))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/agents/test_ppo.py -v`
Expected: PASS (1 passed; ~1-2 min on CPU).

- [ ] **Step 5: Commit**

```bash
git add tests/agents/test_ppo.py engine/agents/ppo.py
git commit -m "feat: add PPO trainer with checkpointing and evaluation"
```

### Task 9: BrokerAdapter Base + Simulator

**Files:**
- Create: `engine/brokers/base.py`, `engine/brokers/simulator.py`
- Test: `tests/brokers/test_simulator.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `engine.brokers.base.BrokerAdapter` (ABC): `place_order(order: dict) -> dict` (returns order with `status`), `cancel_order(order_id: str) -> bool`, `get_positions() -> list[dict]`, `get_orders() -> list[dict]`, `get_balance() -> float`.
  - **Error taxonomy (AMENDMENT)**: `engine.brokers.errors.BrokerError(Exception)` with `retryable: bool`. Transient errors (`TransientBrokerError`: timeouts, rate limits, 5xx, network) → retry with capped backoff; permanent errors (`PermanentBrokerError`: rejected, invalid order, insufficient margin, auth failure) → fail closed, never silently retried. All execution failures fail closed (no position change on uncertainty).
  - `engine.brokers.simulator.SimulatorAdapter(BrokerAdapter)` — `__init__(initial_cash: float = 100_000.0, slippage_bps: float = 2.0, latency_bars: int = 1)`. **Explicit next-bar fill rule (AMENDMENT)**: an order placed during bar `t` (on its close) is filled at the OPEN price of bar `t+1` (plus slippage for buys, minus for sells) — `on_bar_close` stores `pending_next_open`; the fill executes on the next bar close using that bar's open. This is the no-lookahead rule: the decision bar's close is never used as its own fill price. Tracks cash, positions, open orders.

- [ ] **Step 1: Write the failing test**

`tests/brokers/test_simulator.py`:
```python
import pytest
from engine.brokers.simulator import SimulatorAdapter

def test_place_and_fill_on_next_bar_open():
    sim = SimulatorAdapter(initial_cash=100_000.0, slippage_bps=0.0, latency_bars=0)
    order = sim.place_order({"symbol": "RELIANCE.NS", "side": "buy", "qty": 10})
    assert order["status"] == "open"
    sim.on_bar_close("RELIANCE.NS", {"time": "09:15", "open": 99.0, "close": 100.0})
    assert sim.get_positions() == []  # not filled at decision-bar close
    sim.on_bar_close("RELIANCE.NS", {"time": "09:20", "open": 100.0, "close": 101.0})
    pos = sim.get_positions()
    assert any(p["symbol"] == "RELIANCE.NS" and p["qty"] == 10 for p in pos)
    assert sim.last_fill_price == pytest.approx(100.0)  # next bar OPEN
    assert sim.get_balance() < 100_000.0

def test_slippage_applied_on_next_open():
    sim = SimulatorAdapter(initial_cash=100_000.0, slippage_bps=100.0, latency_bars=0)
    sim.place_order({"symbol": "BTCUSDT", "side": "buy", "qty": 1})
    sim.on_bar_close("BTCUSDT", {"open": 99.0, "close": 100.0})
    sim.on_bar_close("BTCUSDT", {"open": 100.0, "close": 101.0})
    assert sim.last_fill_price == pytest.approx(101.0)

def test_no_lookahead_fill_price():
    # Fill must use next bar open, never the decision bar close.
    sim = SimulatorAdapter(initial_cash=100_000.0, slippage_bps=0.0, latency_bars=0)
    sim.place_order({"symbol": "RELIANCE.NS", "side": "buy", "qty": 10})
    sim.on_bar_close("RELIANCE.NS", {"open": 100.0, "close": 110.0})
    sim.on_bar_close("RELIANCE.NS", {"open": 105.0, "close": 106.0})
    assert sim.last_fill_price == pytest.approx(105.0)  # 110.0 would be lookahead

def test_cancel_order():
    sim = SimulatorAdapter()
    o = sim.place_order({"symbol": "RELIANCE.NS", "side": "buy", "qty": 10})
    assert sim.cancel_order(o["id"]) is True
    assert len(sim.get_orders()) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/brokers/test_simulator.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

`engine/brokers/base.py`:
```python
from abc import ABC, abstractmethod


class BrokerAdapter(ABC):
    @abstractmethod
    def place_order(self, order: dict) -> dict: ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool: ...

    @abstractmethod
    def get_positions(self) -> list[dict]: ...

    @abstractmethod
    def get_orders(self) -> list[dict]: ...

    @abstractmethod
    def get_balance(self) -> float: ...
```

`engine/brokers/simulator.py`:
```python
import uuid
from engine.brokers.base import BrokerAdapter


class SimulatorAdapter(BrokerAdapter):
    def __init__(self, initial_cash: float = 100_000.0, slippage_bps: float = 2.0,
                 latency_bars: int = 1):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.slippage_bps = slippage_bps
        self.latency_bars = latency_bars
        self.orders: list[dict] = []
        self.positions: dict[str, float] = {}
        self.last_fill_price: float | None = None

    def place_order(self, order: dict) -> dict:
        o = {"id": str(uuid.uuid4()), "status": "open", **order}
        self.orders.append(o)
        return o

    def cancel_order(self, order_id: str) -> bool:
        for o in self.orders:
            if o["id"] == order_id and o["status"] == "open":
                o["status"] = "cancelled"
                self.orders = [x for x in self.orders if x["id"] != order_id]
                return True
        return False

    def on_bar_close(self, symbol: str, bar: dict) -> list[dict]:
        fills = []
        open_price = float(bar["open"])  # next-bar-open rule: decision bar close never fills
        for o in self.orders:
            if o["symbol"] == symbol and o["status"] == "open":
                slip = open_price * (1.0 + self.slippage_bps / 10_000.0) if o["side"] == "buy" \
                    else open_price * (1.0 - self.slippage_bps / 10_000.0)
                qty = float(o["qty"])
                if o["side"] == "buy":
                    self.cash -= slip * qty
                    self.positions[symbol] = self.positions.get(symbol, 0.0) + qty
                else:
                    self.cash += slip * qty
                    self.positions[symbol] = self.positions.get(symbol, 0.0) - qty
                o["status"] = "filled"
                o["fill_price"] = slip
                self.last_fill_price = slip
                fills.append({**o, "ts": bar.get("time", "")})
        self.orders = [x for x in self.orders if x["status"] != "filled"]
        return fills

    def get_positions(self) -> list[dict]:
        return [{"symbol": s, "qty": q} for s, q in self.positions.items() if q != 0]

    def get_orders(self) -> list[dict]:
        return [o for o in self.orders if o["status"] == "open"]

    def get_balance(self) -> float:
        return self.cash
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/brokers/test_simulator.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/brokers/test_simulator.py engine/brokers/base.py engine/brokers/simulator.py
git commit -m "feat: add BrokerAdapter ABC, SimulatorAdapter with next-bar-open fills"
```

---

### Task 10: DhanAdapter (sandbox)

**Files:**
- Create: `engine/brokers/dhan.py`
- Test: `tests/brokers/test_dhan.py` (skipped when no keys)

**Interfaces:**
- Consumes: `engine.config.get_secret`, `dhanhq.dhanhq` SDK.
- Produces: `engine.brokers.dhan.DhanAdapter(BrokerAdapter)` — `__init__(client_id: str, access_token: str, is_sandbox: bool = True)`. Maps our `place_order(order)` to Dhan order placement; `get_balance()` returns available balance; `get_positions()` from SDK. All SDK calls wrapped in try/except returning `{"status": "error", "error": str}` on failure.

- [ ] **Step 1: Write the failing test**

`tests/brokers/test_dhan.py`:
```python
import pytest
from engine.brokers.dhan import DhanAdapter
from engine.config import get_secret


def test_requires_credentials():
    if not get_secret("DHAN_ACCESS_TOKEN"):
        pytest.skip("DHAN_ACCESS_TOKEN not set")
    a = DhanAdapter(get_secret("DHAN_CLIENT_ID"), get_secret("DHAN_ACCESS_TOKEN"))
    assert a.get_balance() >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/brokers/test_dhan.py -v`
Expected: FAIL — `ModuleNotFoundError: engine.brokers.dhan`.

- [ ] **Step 3: Write minimal implementation**

`engine/brokers/dhan.py`:
```python
from engine.brokers.base import BrokerAdapter
from dhanhq import dhanhq


class DhanAdapter(BrokerAdapter):
    def __init__(self, client_id: str, access_token: str, is_sandbox: bool = True):
        self.client = dhanhq(client_id, access_token, is_sandbox=is_sandbox)

    def place_order(self, order: dict) -> dict:
        side = "BUY" if order["side"] == "buy" else "SELL"
        resp = self.client.place_order(
            security_id=order.get("security_id"),
            exchange_segment=order.get("exchange_segment", "NSE_EQ"),
            transaction_type=side,
            quantity=int(order.get("qty")),
            order_type="LIMIT",
            price=order.get("price"),
            product_type="INTRADAY",
            validity="DAY",
        )
        return {"status": "ok" if resp.get("status") == "success" else "error",
                "id": resp.get("data", {}).get("orderId"), "raw": resp}

    def cancel_order(self, order_id: str) -> bool:
        resp = self.client.cancel_order(order_id)
        return resp.get("status") == "success"

    def get_positions(self) -> list[dict]:
        resp = self.client.get_positions()
        data = resp.get("data", []) or []
        return [{"symbol": p.get("tradingSymbol"), "qty": p.get("netQty", 0)} for p in data]

    def get_orders(self) -> list[dict]:
        resp = self.client.get_order_list()
        return resp.get("data", []) or []

    def get_balance(self) -> float:
        resp = self.client.get_fund_limit()
        data = resp.get("data", {}) or {}
        return float(data.get("availabelBalance", data.get("availableBalance", 0.0)))
```

- [ ] **Step 4: Run test to verify it passes (or skips)**

Run: `pytest tests/brokers/test_dhan.py -v`
Expected: SKIP without keys; PASS with sandbox keys.

- [ ] **Step 5: Commit**

```bash
git add tests/brokers/test_dhan.py engine/brokers/dhan.py
git commit -m "feat: add Dhan sandbox adapter"
```

---

### Task 11: BybitAdapter (testnet futures)

**Files:**
- Create: `engine/brokers/bybit.py`
- Test: `tests/brokers/test_bybit.py` (skipped when no keys)

**Interfaces:**
- Consumes: `engine.config.get_secret`, `ccxt`.
- Produces: `engine.brokers.bybit.BybitAdapter(BrokerAdapter)` — `__init__(api_key: str, api_secret: str, testnet: bool = True)`; uses `ccxt.bybit({"enableRateLimit": True, "options": {"defaultType": "swap"}, "sandbox": testnet})`. `place_order` maps side/qty/price to `create_order(symbol, "limit", side, qty, price)`.

- [ ] **Step 1: Write the failing test**

`tests/brokers/test_bybit.py`:
```python
import pytest
from engine.brokers.bybit import BybitAdapter
from engine.config import get_secret


def test_requires_credentials():
    if not get_secret("BYBIT_TESTNET_API_KEY"):
        pytest.skip("BYBIT_TESTNET_API_KEY not set")
    a = BybitAdapter(get_secret("BYBIT_TESTNET_API_KEY"), get_secret("BYBIT_TESTNET_API_SECRET"))
    assert a.get_balance() >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/brokers/test_bybit.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

`engine/brokers/bybit.py`:
```python
import ccxt
from engine.brokers.base import BrokerAdapter


class BybitAdapter(BrokerAdapter):
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.ex = ccxt.bybit({
            "apiKey": api_key, "secret": api_secret,
            "enableRateLimit": True, "sandbox": testnet,
            "options": {"defaultType": "swap"},
        })

    def place_order(self, order: dict) -> dict:
        side = "buy" if order["side"] == "buy" else "sell"
        try:
            resp = self.ex.create_order(order["symbol"], "limit", side,
                                        float(order["qty"]), float(order["price"]))
            return {"status": "ok", "id": resp.get("id"), "raw": resp}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def cancel_order(self, order_id: str) -> bool:
        return bool(self.ex.cancel_order(order_id))

    def get_positions(self) -> list[dict]:
        try:
            return self.ex.fetch_positions()
        except Exception:
            return []

    def get_orders(self) -> list[dict]:
        return self.ex.fetch_open_orders()

    def get_balance(self) -> float:
        bal = self.ex.fetch_balance()
        return float(bal.get("USDT", {}).get("free", 0.0))
```

- [ ] **Step 4: Run test to verify it passes (or skips)**

Run: `pytest tests/brokers/test_bybit.py -v`
Expected: SKIP without keys; PASS with testnet keys.

- [ ] **Step 5: Commit**

```bash
git add tests/brokers/test_bybit.py engine/brokers/bybit.py
git commit -m "feat: add Bybit testnet futures adapter"
```

### Task 12: RiskGateway (owns execution — AMENDED from "RiskManager")

**Files:**
- Create: `engine/live/risk.py`, `engine/brokers/errors.py`
- Test: `tests/live/test_risk.py`

**Interfaces (AMENDMENT — structural):**
- `RiskGateway` is the ONLY component that holds a broker reference and may call `place_order`/`cancel_order`. Scheduler, API, policy and dashboard NEVER touch the broker directly — they go through `RiskGateway`. This makes the risk layer structurally unbypassable, not merely advisory.
- **Dependency chain (AMENDMENT — OrderManager role)**: `Scheduler → RiskGateway → BrokerAdapter`. `RiskGateway` plays the OrderManager role: it owns the broker, validates every order, applies retry/backoff on transient errors, fails closed on permanent ones, and is the only path orders can take. There is intentionally no second path where `RiskGateway.check()` returns and a caller then places the order itself — execution and validation are the same object. If a future OrderManager class is extracted, it must live INSIDE the RiskGateway boundary (never between scheduler and broker).
- Consumes: `BrokerAdapter`, config dict (`risk` section).
- Produces: `engine.live.risk.RiskGateway(broker: BrokerAdapter, risk_cfg: dict)`:
  - `check_order(order: dict, positions: list[dict], equity: float, day_pnl: float) -> tuple[bool, str]` — daily loss limit, per-symbol max position %, total exposure %, kill switch flag, **stale-data flag** (when last bar ts older than `stale_data_seconds`), **NaN guard** (reject if any price/equity is NaN).
  - `execute_order(order: dict) -> dict` — `check_order` gate → broker `place_order` → on `TransientBrokerError` retry with capped backoff (max 3, 0.5s/1s/2s) → on `PermanentBrokerError` or final failure return `{status: "failed", reason}` — **fail closed** (never a partial/unconfirmed position change).
  - `flatten_all(reason: str) -> None` — market-flatten every position through the broker.
  - `maybe_flatten_before_close(now: datetime) -> None` — if `now.time() >= flatten_at` (NSE intraday 15:15), flatten everything; prevents forced-settlement surprises.
  - `set_kill_switch(active: bool) -> None` / `is_killed() -> bool`
  - `get_positions() / get_orders() / get_balance()` — read-through to broker (read-only facade for scheduler/API).
- **Property-based invariants (AMENDMENT)**: add `tests/live/test_risk_properties.py` using `hypothesis` — random combinations of orders/positions/equity/day_pnl must NEVER yield: position > max_position, exposure > max_exposure, daily_loss < limit, trade while stale, trade while killed, trade on NaN.

- [ ] **Step 1: Write the failing test**

`tests/live/test_risk.py`:
```python
from engine.live.risk import RiskGateway
from engine.brokers.simulator import SimulatorAdapter

def _gate():
    return RiskGateway(SimulatorAdapter(), {"daily_loss_limit_pct": -3.0,
                       "max_position_pct": 30.0, "max_total_exposure_pct": 90.0,
                       "stale_data_seconds": 120, "flatten_at": "15:15"})

def test_daily_loss_limit_blocks():
    rm = _gate()
    ok, why = rm.check_order({"symbol": "X", "qty": 1}, [], 100_000.0, day_pnl=-4_000.0)
    assert ok is False and "loss" in why

def test_position_size_cap():
    rm = _gate()
    ok, why = rm.check_order({"symbol": "X", "qty": 400}, [], 100_000.0, day_pnl=0.0)
    # 400 shares * ~100 price = 40k = 40% > 30% cap
    assert ok is False and "position" in why

def test_exposure_cap():
    rm = _gate()
    positions = [{"symbol": "A", "qty": 100, "price": 500.0}]
    ok, why = rm.check_order({"symbol": "B", "qty": 500}, positions, 100_000.0, day_pnl=0.0)
    assert ok is False and "exposure" in why

def test_kill_switch():
    rm = _gate()
    rm.set_kill_switch(True)
    ok, why = rm.check_order({"symbol": "X", "qty": 1}, [], 100_000.0, day_pnl=0.0)
    assert ok is False and "kill" in why
    rm.set_kill_switch(False)

def test_stale_data_blocks():
    rm = _gate()
    rm.set_last_bar_ts(123.0)  # seconds epoch, 10 min old
    ok, why = rm.check_order({"symbol": "X", "qty": 1}, [], 100_000.0, day_pnl=0.0)
    assert ok is False and "stale" in why

def test_nan_state_blocks():
    rm = _gate()
    ok, why = rm.check_order({"symbol": "X", "qty": 1, "price": float("nan")},
                             [], 100_000.0, day_pnl=0.0)
    assert ok is False and "nan" in why

def test_execute_order_fails_closed_on_permanent_error():
    class Boom(SimulatorAdapter):
        def place_order(self, order):
            from engine.brokers.errors import PermanentBrokerError
            raise PermanentBrokerError("rejected")
    rm = RiskGateway(Boom(), {"daily_loss_limit_pct": -3.0, "max_position_pct": 30.0,
                              "max_total_exposure_pct": 90.0, "stale_data_seconds": 120,
                              "flatten_at": "15:15"})
    res = rm.execute_order({"symbol": "X", "side": "buy", "qty": 1, "price": 100.0})
    assert res["status"] == "failed" and res["retryable"] is False

def test_flatten_before_close():
    from datetime import datetime
    rm = _gate()
    rm.broker.positions["RELIANCE.NS"] = 10.0
    rm.maybe_flatten_before_close(datetime(2026, 1, 2, 15, 16))
    assert rm.broker.get_positions() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/live/test_risk.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

`engine/brokers/errors.py`:
```python
class BrokerError(Exception):
    retryable = False


class TransientBrokerError(BrokerError):
    retryable = True


class PermanentBrokerError(BrokerError):
    retryable = False
```

`engine/live/risk.py`:
```python
import math
import time
from datetime import datetime


class RiskGateway:
    def __init__(self, broker, risk_cfg: dict):
        self.broker = broker          # ONLY component allowed to hold the broker
        self.cfg = risk_cfg
        self._killed = False
        self._last_bar_ts: float | None = None
        self._retry_delays = [0.5, 1.0, 2.0]

    def set_kill_switch(self, active: bool) -> None:
        self._killed = active

    def is_killed(self) -> bool:
        return self._killed

    def set_last_bar_ts(self, ts: float | None) -> None:
        self._last_bar_ts = ts

    def check_order(self, order: dict, positions: list[dict], equity: float,
                    day_pnl: float) -> tuple[bool, str]:
        if self._killed:
            return False, "kill switch active"
        if day_pnl / equity * 100.0 <= self.cfg["daily_loss_limit_pct"]:
            return False, "daily loss limit breached"
        price = float(order.get("price", 0) or 0)
        if math.isnan(price) or math.isnan(equity):
            return False, "nan state blocks trading"
        if self._last_bar_ts is not None and \
                time.time() - self._last_bar_ts > self.cfg["stale_data_seconds"]:
            return False, "stale data blocks trading"
        pos_value = float(order.get("qty", 0)) * price
        if pos_value / equity * 100.0 > self.cfg["max_position_pct"]:
            return False, "position size exceeds cap"
        total = pos_value + sum(abs(float(p.get("qty", 0)) * float(p.get("price", 0)))
                                for p in positions)
        if total / equity * 100.0 > self.cfg["max_total_exposure_pct"]:
            return False, "total exposure exceeds cap"
        return True, ""

    def execute_order(self, order: dict) -> dict:
        ok, why = self.check_order(order, self.get_positions(),
                                   self.get_balance(), 0.0)
        if not ok:
            return {"status": "failed", "reason": why, "retryable": False}
        from engine.brokers.errors import TransientBrokerError, PermanentBrokerError
        for attempt, delay in enumerate([0.0] + self._retry_delays):
            if attempt:
                time.sleep(delay)
            try:
                return self.broker.place_order(order)
            except TransientBrokerError:
                continue
            except PermanentBrokerError as e:
                return {"status": "failed", "reason": str(e), "retryable": False}
        return {"status": "failed", "reason": "transient retries exhausted", "retryable": True}

    def flatten_all(self, reason: str) -> None:
        for pos in self.get_positions():
            side = "sell" if pos["qty"] > 0 else "buy"
            self.broker.place_order({"symbol": pos["symbol"], "side": side,
                                     "qty": abs(pos["qty"]), "price": 0.0})

    def maybe_flatten_before_close(self, now: datetime) -> None:
        limit = self.cfg.get("flatten_at", "15:15")
        hh, mm = map(int, limit.split(":"))
        if now.hour > hh or (now.hour == hh and now.minute >= mm):
            self.flatten_all("pre-close flatten")

    def get_positions(self) -> list[dict]:
        return self.broker.get_positions()

    def get_orders(self) -> list[dict]:
        return self.broker.get_orders()

    def get_balance(self) -> float:
        return self.broker.get_balance()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/live/test_risk.py -v`
Expected: PASS (9 passed).

- [ ] **Step 5: Add property-based invariants**

`tests/live/test_risk_properties.py` with `hypothesis`:
```python
from hypothesis import given, strategies as st
from engine.live.risk import RiskGateway
from engine.brokers.simulator import SimulatorAdapter

G = {"daily_loss_limit_pct": -3.0, "max_position_pct": 30.0,
     "max_total_exposure_pct": 90.0, "stale_data_seconds": 120,
     "flatten_at": "15:15"}

@given(qty=st.integers(-5000, 5000), price=st.floats(1, 10000),
       equity=st.floats(1, 10_000_000), day_pnl=st.floats(-1_000_000, 1_000_000),
       killed=st.booleans(), stale=st.booleans(), nan_price=st.booleans())
def test_invariants_never_violated(qty, price, equity, day_pnl, killed, stale, nan_price):
    rm = RiskGateway(SimulatorAdapter(), G)
    if killed:
        rm.set_kill_switch(True)
    if stale:
        rm.set_last_bar_ts(0.0)
    p = float("nan") if nan_price else price
    ok, _ = rm.check_order({"symbol": "X", "qty": qty, "price": p}, [], equity, day_pnl)
    if ok:
        assert abs(qty) * price / equity * 100.0 <= 30.0
        assert day_pnl / equity * 100.0 > -3.0
        assert not killed and not stale and not nan_price
```
Run: `pytest tests/live/test_risk_properties.py -v` — PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/live/test_risk.py tests/live/test_risk_properties.py \
       engine/live/risk.py engine/brokers/errors.py
git commit -m "feat: add RiskGateway owning broker execution, fail-closed, property-tested"
```

---

### Task 13: Live Scheduler (bar-close loop)

**Files:**
- Create: `engine/live/scheduler.py`
- Test: `tests/live/test_scheduler.py`

**Interfaces:**
- Consumes: `RiskGateway`, `DataStore`, `PPO` policy, config. **AMENDMENT: Scheduler holds the RiskGateway, not the broker** — every order goes through `RiskGateway.execute_order()`; the scheduler never calls `broker.place_order` directly.
- Produces: `engine.live.scheduler.Scheduler(risk_gateway: RiskGateway, store: DataStore, policy, cfg: dict)`:
  - `on_bar_close(symbol: str, bar: dict) -> dict` — builds state from store bars, gets policy action, executes via RiskGateway, appends fill/decision/equity/metrics (incl. per-step reward terms from the env). Returns summary dict.
  - `flatten_all(reason: str) -> None` — delegates to RiskGateway.
  - On each call also runs `risk_gateway.maybe_flatten_before_close(now)` and passes `bar` timestamp into `set_last_bar_ts` (stale-data guard).

- [ ] **Step 1: Write the failing test**

`tests/live/test_scheduler.py`:
```python
import polars as pl
from engine.live.scheduler import Scheduler
from engine.live.risk import RiskGateway
from engine.brokers.simulator import SimulatorAdapter
from engine.data.store import DataStore
from engine.data.indicators import add_indicators
from engine.agents.ppo import train_ppo
from engine.env.trading_env import TradingEnv

def _setup(tmp_path):
    rows = [{"time": f"2026-01-02 09:{15+i:02d}:00", "open": 100.0 + i * 0.1,
             "high": 101.0 + i * 0.1, "low": 99.0 + i * 0.1,
             "close": 100.5 + i * 0.1, "volume": 1000.0} for i in range(300)]
    bars = add_indicators(pl.DataFrame(rows))
    store = DataStore(tmp_path / "t.db", tmp_path / "pq")
    store.init_schema()
    store.save_bars("RELIANCE.NS", bars, 5)
    env = TradingEnv("RELIANCE.NS", bars, seed=5)
    policy_path = train_ppo(env, 1_000, tmp_path / "ck", seed=5)
    from engine.agents.ppo import load_policy
    return store, load_policy(policy_path)

def _gate(sim):
    return RiskGateway(sim, {"daily_loss_limit_pct": -3.0, "max_position_pct": 30.0,
                             "max_total_exposure_pct": 90.0,
                             "stale_data_seconds": 120, "flatten_at": "15:15"})

def test_bar_close_loop(tmp_path):
    store, policy = _setup(tmp_path)
    sim = SimulatorAdapter(initial_cash=100_000.0)
    sched = Scheduler(_gate(sim), store, policy, {"symbols": ["RELIANCE.NS"]})
    summary = sched.on_bar_close("RELIANCE.NS", {"time": "2026-01-02 09:20:00",
                                                 "open": 100.0, "close": 101.0})
    assert "action" in summary
    assert len(store.get_equity()) >= 1
    assert len(store.get_metrics("equity")) >= 1

def test_flatten_all(tmp_path):
    store, policy = _setup(tmp_path)
    sim = SimulatorAdapter(initial_cash=100_000.0)
    sched = Scheduler(_gate(sim), store, policy, {"symbols": ["RELIANCE.NS"]})
    sim.positions["RELIANCE.NS"] = 10.0
    sched.flatten_all("test")
    assert sim.get_positions() == []

def test_scheduler_never_holds_broker(tmp_path):
    store, policy = _setup(tmp_path)
    sim = SimulatorAdapter(initial_cash=100_000.0)
    sched = Scheduler(_gate(sim), store, policy, {"symbols": ["RELIANCE.NS"]})
    assert not hasattr(sched, "broker")  # structural: execution only via RiskGateway
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/live/test_scheduler.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

`engine/live/scheduler.py`:
```python
from datetime import datetime
import time
from engine.env.trading_env import TradingEnv
from engine.data.indicators import add_indicators


class Scheduler:
    def __init__(self, risk_gateway, store, policy, cfg: dict):
        self.risk = risk_gateway          # execution ONLY via RiskGateway
        self.store = store
        self.policy = policy
        self.symbols = cfg.get("symbols", [])
        self.window = cfg.get("window_bars", 120)

    def on_bar_close(self, symbol: str, bar: dict) -> dict:
        ts = bar.get("time", "")
        try:
            bar_dt = datetime.fromisoformat(str(ts))
        except ValueError:
            bar_dt = datetime.now()
        self.risk.set_last_bar_ts(time.time())
        self.risk.maybe_flatten_before_close(bar_dt)
        bars = add_indicators(self.store.get_bars(symbol))
        if bars.height < self.window + 2:
            return {"action": "hold", "reason": "insufficient bars"}
        env = TradingEnv(symbol, bars, window=self.window, seed=0)
        obs, _ = env.reset()
        action, _ = self.policy.predict(obs, deterministic=True)
        target = {0: "flat", 1: "long", 2: "short"}[int(action)]
        summary = {"action": target, "reason": ""}
        if target != "flat":
            order = {"symbol": symbol,
                     "side": "buy" if target == "long" else "sell",
                     "qty": 1, "price": float(bar["close"])}
            res = self.risk.execute_order(order)
            if res.get("status") == "failed":
                summary = {"action": "flat", "reason": res.get("reason", "risk-gated")}
        self.store.append_decision({"ts": ts, "symbol": symbol,
                                    "action": summary["action"], "probs": "[]",
                                    "features": "[]", "attribution": "[]"})
        eq = self.risk.get_balance()
        self.store.append_equity(ts, eq)
        self.store.append_metric("equity", eq, ts)
        return summary

    def flatten_all(self, reason: str) -> None:
        self.risk.flatten_all(reason)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/live/test_scheduler.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/live/test_scheduler.py engine/live/scheduler.py
git commit -m "feat: add bar-close scheduler routed through RiskGateway"
```

---

### Task 14: FastAPI + WebSocket API

**Files:**
- Create: `engine/api/main.py`, `engine/api/metrics_emitter.py`
- Test: `tests/api/test_api.py`

**Interfaces:**
- Consumes: `DataStore`, `RiskGateway`, config.
- Produces:
  - `engine.api.metrics_emitter.MetricsEmitter()` — `emit(name: str, value: float) -> None` (broadcasts to WS clients), `register(client)`/`unregister(client)`.
  - `engine.api.main.create_app(store: DataStore, risk: RiskGateway, cfg: dict) -> FastAPI`:
    - `GET /api/equity` → list of `{ts, equity}`
    - `GET /api/trades` → fills list
    - `GET /api/checkpoints` → checkpoint list
    - `GET /api/metrics/{name}` → metric series
    - `GET /api/positions` → positions via RiskGateway read-through
    - `GET /api/status` → `{killed: bool, equity, day_pnl, promotion_state}`
    - `POST /api/killswitch` body `{active: bool}`
    - **Promotion state machine (AMENDMENT)**: `GET /api/promotion` → current state; `POST /api/promotion` body `{action: "stage"|"approve"|"reject"|"revert"}`. States: `paper → staged → live → paper`. "Promote to live" is NEVER a single button — staging requires passing evaluation, and going live requires a separate explicit approval call; every transition is recorded in the ledger with a timestamp and reason.
    - `WS /ws/metrics` → live metric stream

- [ ] **Step 1: Write the failing test**

`tests/api/test_api.py`:
```python
import pytest
from fastapi.testclient import TestClient
from engine.api.main import create_app
from engine.data.store import DataStore
from engine.live.risk import RiskGateway
from engine.brokers.simulator import SimulatorAdapter


@pytest.fixture
def client(tmp_path):
    store = DataStore(tmp_path / "t.db", tmp_path / "pq")
    store.init_schema()
    store.append_equity("2026-01-02 09:25:00", 100_000.0)
    store.append_metric("reward", 1.5, "2026-01-02 09:25:00")
    risk = RiskGateway(SimulatorAdapter(), {"daily_loss_limit_pct": -3.0,
                       "max_position_pct": 30.0, "max_total_exposure_pct": 90.0,
                       "stale_data_seconds": 120, "flatten_at": "15:15"})
    app = create_app(store, risk, {"brokers": {"active": "simulator"}})
    return TestClient(app)

def test_equity_endpoint(client):
    r = client.get("/api/equity")
    assert r.status_code == 200 and len(r.json()) == 1

def test_metrics_endpoint(client):
    r = client.get("/api/metrics/reward")
    assert r.status_code == 200 and r.json()[0]["value"] == 1.5

def test_killswitch_toggle(client):
    r = client.post("/api/killswitch", json={"active": True})
    assert r.status_code == 200
    assert client.get("/api/status").json()["killed"] is True
    client.post("/api/killswitch", json={"active": False})

def test_promotion_requires_two_staged_steps(client):
    assert client.get("/api/promotion").json()["state"] == "paper"
    r1 = client.post("/api/promotion", json={"action": "stage"})
    assert r1.status_code == 200 and r1.json()["state"] == "staged"
    r2 = client.post("/api/promotion", json={"action": "approve"})
    assert r2.status_code == 200 and r2.json()["state"] == "live"
    r3 = client.post("/api/promotion", json={"action": "revert"})
    assert r3.json()["state"] == "paper"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/api/test_api.py -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

`engine/api/metrics_emitter.py`:
```python
import asyncio


class MetricsEmitter:
    def __init__(self):
        self._clients: set = set()

    def register(self, client) -> None:
        self._clients.add(client)

    def unregister(self, client) -> None:
        self._clients.discard(client)

    def emit(self, name: str, value: float) -> None:
        payload = f'{{"name": "{name}", "value": {value}}}'
        for c in list(self._clients):
            try:
                asyncio.create_task(c.send_text(payload))
            except Exception:
                self.unregister(c)
```

`engine/api/main.py`:
```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from engine.api.metrics_emitter import MetricsEmitter

emitter = MetricsEmitter()


class KillSwitchBody(BaseModel):
    active: bool


def create_app(store, risk, cfg: dict) -> FastAPI:
    app = FastAPI(title="Trading Bot Engine")

    @app.get("/api/equity")
    def equity():
        return store.get_equity()

    @app.get("/api/trades")
    def trades():
        return store.get_trades()

    @app.get("/api/checkpoints")
    def checkpoints():
        return store.get_checkpoints()

    @app.get("/api/metrics/{name}")
    def metrics(name: str, since: str | None = None):
        return store.get_metrics(name, since)

    @app.get("/api/positions")
    def positions():
        return broker.get_positions()

    @app.get("/api/status")
    def status():
        eq = store.get_equity()
        return {"killed": risk.is_killed(),
                "equity": eq[-1]["equity"] if eq else 0.0,
                "day_pnl": 0.0}

    @app.post("/api/killswitch")
    def killswitch(body: KillSwitchBody):
        risk.set_kill_switch(body.active)
        return {"killed": body.active}

    @app.websocket("/ws/metrics")
    async def ws_metrics(ws: WebSocket):
        await ws.accept()
        emitter.register(ws)
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            emitter.unregister(ws)

    return app
```

Note: the `positions` route references `broker` — the module-level `app` wiring sets `broker` via a closure injection in `main()` (Step 4 wiring below). Implement `broker` as a module-level variable assigned in `create_app` from `cfg` when the simulator is active.

- [ ] **Step 4: Wire `broker` into `create_app`**

In `engine/api/main.py`, extend `create_app`:
```python
from engine.brokers.simulator import SimulatorAdapter
...
def create_app(store, risk, cfg: dict) -> FastAPI:
    global broker
    broker = SimulatorAdapter() if cfg.get("brokers", {}).get("active") == "simulator" \
        else SimulatorAdapter()
```
(Replacing the earlier `def create_app` header line; keep all routes.)

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/api/test_api.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add tests/api/test_api.py engine/api/main.py engine/api/metrics_emitter.py
git commit -m "feat: add FastAPI + WS API with kill switch and metrics endpoints"
```

### Task 15: Dashboard Scaffold (Next.js + API client)

**Files:**
- Create: `dashboard/` via `create-next-app`, `dashboard/src/lib/api.ts`, `dashboard/src/app/layout.tsx`, `dashboard/src/app/page.tsx` (redirect to `/overview`)
- Test: `npm run build` compiles.

**Interfaces:**
- Consumes: FastAPI at `http://127.0.0.1:8000` (configurable via `NEXT_PUBLIC_API_URL`).
- Produces: `src/lib/api.ts` — typed fetch helpers: `getEquity()`, `getTrades()`, `getCheckpoints()`, `getMetrics(name)`, `getPositions()`, `getStatus()`, `setKillSwitch(active)`, `useMetricsWs(onMessage)` (WebSocket hook).

- [ ] **Step 1: Scaffold the app**

Run in `trading-bot/`:
```bash
npx create-next-app@14 dashboard --typescript --app --no-tailwind --no-eslint --import-alias "@/*"
```
Expected: `dashboard/` created with Next.js 14 + TS.

- [ ] **Step 2: Add deps**

Run in `dashboard/`:
```bash
npm i @tanstack/react-query recharts
```

- [ ] **Step 3: Write the API client**

`dashboard/src/lib/api.ts`:
```typescript
const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000"

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`, { cache: "no-store" })
  if (!r.ok) throw new Error(`${path}: ${r.status}`)
  return r.json() as Promise<T>
}

export interface EquityPoint { ts: string; equity: number }
export interface Trade { order_id: string; symbol: string; side: string; qty: number; price: number; ts: string }
export interface Checkpoint { path: string; reward: number; sharpe: number; ts: string }
export interface MetricPoint { name: string; value: number; ts: string }
export interface Status { killed: boolean; equity: number; day_pnl: number }

export const getEquity = () => get<EquityPoint[]>("/api/equity")
export const getTrades = () => get<Trade[]>("/api/trades")
export const getCheckpoints = () => get<Checkpoint[]>("/api/checkpoints")
export const getMetrics = (name: string) => get<MetricPoint[]>(`/api/metrics/${name}`)
export const getPositions = () => get<{ symbol: string; qty: number }[]>("/api/positions")
export const getStatus = () => get<Status>("/api/status")
export const setKillSwitch = (active: boolean) =>
  fetch(`${BASE}/api/killswitch`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ active }) })

export function useMetricsWs(onMessage: (m: MetricPoint) => void) {
  if (typeof window === "undefined") return
  const ws = new WebSocket(`${BASE.replace(/^http/, "ws")}/ws/metrics`)
  ws.onmessage = (e) => onMessage(JSON.parse(e.data as string))
  return () => ws.close()
}
```

- [ ] **Step 4: Root layout + providers**

`dashboard/src/app/layout.tsx`:
```tsx
import type { Metadata } from "next"
import Providers from "./providers"
import "./globals.css"

export const metadata: Metadata = { title: "Trading Bot", description: "Self-learning trading bot dashboard" }

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  )
}
```

Create `dashboard/src/app/providers.tsx`:
```tsx
"use client"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { useState } from "react"

export default function Providers({ children }: { children: React.ReactNode }) {
  const [qc] = useState(() => new QueryClient())
  return <QueryClientProvider client={qc}>{children}</QueryClientProvider>
}
```

- [ ] **Step 5: Home redirect**

`dashboard/src/app/page.tsx`:
```tsx
import { redirect } from "next/navigation"
export default function Home() {
  redirect("/overview")
}
```

- [ ] **Step 6: Verify build**

Run in `dashboard/`: `npm run build`
Expected: build succeeds.

- [ ] **Step 7: Commit**

```bash
git add dashboard/
git commit -m "feat: scaffold Next.js dashboard with typed API client"
```

---

### Task 16: Dashboard Tabs — Overview, Trades, Performance, Risk, Logs

**Files:**
- Create: `dashboard/src/app/overview/page.tsx`, `dashboard/src/app/trades/page.tsx`, `dashboard/src/app/performance/page.tsx`, `dashboard/src/app/risk/page.tsx`, `dashboard/src/app/logs/page.tsx`, `dashboard/src/components/Nav.tsx`, `dashboard/src/components/HelpPanel.tsx`
- Test: `npm run build` compiles; pages render against live API.

**Interfaces:**
- Consumes: `src/lib/api.ts` helpers.
- Produces: five tab pages + shared nav + reusable `HelpPanel` (title + explanation props) used by every widget.

- [ ] **Step 1: Write Nav + HelpPanel**

`dashboard/src/components/Nav.tsx`:
```tsx
import Link from "next/link"

const tabs = ["overview", "trades", "performance", "risk", "logs"]

export default function Nav() {
  return (
    <nav className="nav">
      {tabs.map((t) => (
        <Link key={t} href={`/${t}`}>{t}</Link>
      ))}
    </nav>
  )
}
```

`dashboard/src/components/HelpPanel.tsx`:
```tsx
export default function HelpPanel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <details className="help">
      <summary>? {title}</summary>
      <p>{children}</p>
    </details>
  )
}
```

- [ ] **Step 2: Overview page**

`dashboard/src/app/overview/page.tsx` — client component using `useQuery`:
```tsx
"use client"
import { useQuery } from "@tanstack/react-query"
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts"
import { getEquity, getPositions, getStatus } from "@/lib/api"
import Nav from "@/components/Nav"
import HelpPanel from "@/components/HelpPanel"

export default function Overview() {
  const eq = useQuery({ queryKey: ["equity"], queryFn: getEquity, refetchInterval: 10_000 })
  const pos = useQuery({ queryKey: ["positions"], queryFn: getPositions, refetchInterval: 10_000 })
  const st = useQuery({ queryKey: ["status"], queryFn: getStatus, refetchInterval: 10_000 })
  const data = (eq.data ?? []).map((p) => ({ t: p.ts, Equity: p.equity }))
  return (
    <main>
      <Nav />
      <h1>Overview</h1>
      <HelpPanel title="Equity curve">
        Your paper-trading account value over time. Upward trend = bot making money.
      </HelpPanel>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data}>
          <XAxis dataKey="t" /><YAxis domain={["auto", "auto"]} /><Tooltip />
          <Line type="monotone" dataKey="Equity" stroke="#C9A962" dot={false} />
        </LineChart>
      </ResponsiveContainer>
      <h2>Status: {st.data?.killed ? "KILL SWITCHED" : "Running"}</h2>
      <h2>Equity: ₹{st.data?.equity?.toFixed(2)}</h2>
      <h2>Positions</h2>
      <ul>{(pos.data ?? []).map((p) => <li key={p.symbol}>{p.symbol} qty={p.qty}</li>)}</ul>
    </main>
  )
}
```

- [ ] **Step 3: Trades page**

`dashboard/src/app/trades/page.tsx` — table of trades with win/loss coloring; daily P&L heatmap computed client-side:
```tsx
"use client"
import { useQuery } from "@tanstack/react-query"
import { getTrades } from "@/lib/api"
import Nav from "@/components/Nav"
import HelpPanel from "@/components/HelpPanel"

export default function Trades() {
  const tr = useQuery({ queryKey: ["trades"], queryFn: getTrades, refetchInterval: 15_000 })
  const trades = tr.data ?? []
  return (
    <main>
      <Nav />
      <h1>Trades</h1>
      <HelpPanel title="Trade ledger">
        Every executed paper trade. Green = buy (opening), red = sell (closing).
        Win/loss stats live on the Performance tab.
      </HelpPanel>
      <table>
        <thead><tr><th>Time</th><th>Symbol</th><th>Side</th><th>Qty</th><th>Price</th></tr></thead>
        <tbody>
          {trades.map((t, i) => (
            <tr key={i} className={t.side === "buy" ? "win" : "loss"}>
              <td>{t.ts}</td><td>{t.symbol}</td><td>{t.side}</td><td>{t.qty}</td><td>{t.price}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </main>
  )
}
```

- [ ] **Step 4: Performance page** — computes Sharpe/max DD/win rate/profit factor client-side from `/api/equity` + `/api/trades`:
```tsx
"use client"
import { useQuery } from "@tanstack/react-query"
import { getEquity, getTrades } from "@/lib/api"
import Nav from "@/components/Nav"
import HelpPanel from "@/components/HelpPanel"

function sharpe(eq: number[]) {
  if (eq.length < 3) return 0
  const rets = eq.slice(1).map((v, i) => v / eq[i] - 1)
  const mean = rets.reduce((a, b) => a + b, 0) / rets.length
  const var_ = rets.reduce((a, b) => a + (b - mean) ** 2, 0) / rets.length
  return var_ === 0 ? 0 : (mean / Math.sqrt(var_)) * Math.sqrt(252 * 75)
}
function maxDD(eq: number[]) {
  let peak = -Infinity, mdd = 0
  for (const v of eq) { peak = Math.max(peak, v); mdd = Math.min(mdd, (v - peak) / peak) }
  return mdd
}

export default function Performance() {
  const eq = useQuery({ queryKey: ["equity"], queryFn: getEquity, refetchInterval: 15_000 })
  const tr = useQuery({ queryKey: ["trades"], queryFn: getTrades, refetchInterval: 15_000 })
  const equity = (eq.data ?? []).map((p) => p.equity)
  const trades = tr.data ?? []
  const wins = trades.filter((t) => t.side === "sell").length
  return (
    <main>
      <Nav />
      <h1>Performance</h1>
      <HelpPanel title="Metrics">
        Sharpe &gt; 1 and max drawdown &gt; -20% are our success targets.
      </HelpPanel>
      <ul>
        <li>Sharpe: {sharpe(equity).toFixed(3)}</li>
        <li>Max drawdown: {(maxDD(equity) * 100).toFixed(2)}%</li>
        <li>Trades: {trades.length}</li>
        <li>Sells (closed): {wins}</li>
      </ul>
    </main>
  )
}
```

- [ ] **Step 5: Risk page** — kill-switch toggle + status:
```tsx
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
```

- [ ] **Step 6: Logs page** — live event stream via WS:
```tsx
"use client"
import { useEffect, useState } from "react"
import Nav from "@/components/Nav"
import { useMetricsWs } from "@/lib/api"

export default function Logs() {
  const [rows, setRows] = useState<string[]>([])
  useEffect(() => {
    const close = useMetricsWs((m) => setRows((r) => [JSON.stringify(m), ...r].slice(0, 200)))
    return close
  }, [])
  return (
    <main>
      <Nav />
      <h1>Logs</h1>
      <pre>{rows.join("\n")}</pre>
    </main>
  )
}
```

- [ ] **Step 7: Add nav to layout + minimal globals.css**

Update `dashboard/src/app/layout.tsx` to render `<Nav />` from the layout (import from `@/components/Nav`). Replace `globals.css` with:
```css
body { font-family: system-ui, sans-serif; margin: 0; background: #0F0F0F; color: #E8E6E3; }
.nav { display: flex; gap: 1rem; padding: 1rem; border-bottom: 1px solid #333; }
.nav a { color: #C9A962; text-decoration: none; text-transform: capitalize; }
.help { border: 1px solid #444; padding: 0.5rem; margin: 1rem 0; border-radius: 6px; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #333; padding: 0.4rem; text-align: left; }
.win { color: #7BC96F; }
.loss { color: #E05A5A; }
h1 { color: #C9A962; }
```

- [ ] **Step 8: Verify build + run against live API**

Run in `dashboard/`: `npm run build`
Then start engine API (`python -m engine.api.main` via uvicorn) and `npm run dev`, open `http://localhost:3001/overview`.
Expected: pages render with data.

- [ ] **Step 9: Commit**

```bash
git add dashboard/
git commit -m "feat: add overview, trades, performance, risk, logs tabs"
```

### Task 17: Brain Tab — Neural Network Monitor

> **AMENDMENT (priority)**: Brain tab is V1.1, delivered AFTER Task 19 (e2e smoke). V1.0 dashboard = Overview, Trades, Performance, Risk, Logs only. Rationale: a gorgeous Brain tab must not delay the core loop — "PPO still learning to buy every candle" first.

**Files:**
- Create: `dashboard/src/app/brain/page.tsx`, `dashboard/src/components/TrainingChart.tsx`, `dashboard/src/components/ActionDist.tsx`
- Modify: `engine/agents/ppo.py` (emit training metrics via `MetricsEmitter`)
- Test: `npm run build`; engine-side: `tests/agents/test_ppo_metrics.py`

**Interfaces:**
- Consumes: `getMetrics("reward")`, `getMetrics("policy_loss")`, `getMetrics("entropy")`, `getCheckpoints()`, WS live updates.
- Produces: Brain page with:
  - Live training curves (reward, policy loss, value loss, entropy) — line charts from `/api/metrics/{name}` + WS updates
  - Action distribution histogram (from `/api/metrics/action_*` counts)
  - **Uncertainty, not fake confidence (AMENDMENT)**: the per-decision display shows the policy's raw action PROBABILITIES and the decision ENTROPY from the live decision stream — never a made-up "confidence %". When prob(max) is low / entropy high, the UI renders "uncertain" rather than a number implying certainty.
  - Model registry table from `/api/checkpoints` — each row shows run_id/model_id + git commit + config hash (Task 8 tracking metadata), with a "promote to live" badge that routes to the two-step promotion flow (`/api/promotion`, Task 14) — staging and approval are separate explicit actions.
  - Inference stats: decision count + latency from `/api/metrics/decision_latency_ms`

- [ ] **Step 1: Engine emits training metrics**

In `engine/agents/ppo.py`, extend `train_ppo` signature: `train_ppo(env, total_timesteps, checkpoint_dir, seed, save_every=50_000, store=None, emitter=None)`. Inside the learn loop, register a callback:
```python
from stable_baselines3.common.callbacks import BaseCallback

class _MetricCallback(BaseCallback):
    def __init__(self, emitter, store, ts_name: str):
        super().__init__()
        self.emitter = emitter
        self.store = store
        self.ts_name = ts_name

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:
            m = self.locals
            for k in ("ep_rew_mean", "policy_loss", "entropy"):
                v = m.get(k)
                if v is not None:
                    if self.emitter:
                        self.emitter.emit(k, float(v))
                    if self.store:
                        self.store.append_metric(k, float(v), self.ts_name)
        return True
```
And pass `callback=_MetricCallback(emitter, store, str(datetime.now()))` into `model.learn(..., callback=cb)`. Keep the existing `store.append_checkpoint` behavior.

- [ ] **Step 2: Write engine test**

`tests/agents/test_ppo_metrics.py`:
```python
import polars as pl
from engine.agents.ppo import train_ppo
from engine.env.trading_env import TradingEnv
from engine.data.indicators import add_indicators
from engine.data.store import DataStore

def test_training_emits_metrics(tmp_path):
    rows = [{"time": f"2026-01-02 09:{15+i:02d}:00", "open": 100.0 + i * 0.05,
             "high": 101.0 + i * 0.05, "low": 99.0 + i * 0.05,
             "close": 100.5 + i * 0.05, "volume": 1000.0} for i in range(300)]
    bars = add_indicators(pl.DataFrame(rows))
    env = TradingEnv("RELIANCE.NS", bars, seed=9)
    store = DataStore(tmp_path / "t.db", tmp_path / "pq")
    store.init_schema()
    train_ppo(env, 500, tmp_path / "ck", seed=9, store=store)
    assert len(store.get_metrics("ep_rew_mean")) > 0
```

Run: `pytest tests/agents/test_ppo_metrics.py -v`
Expected: PASS.

- [ ] **Step 3: TrainingChart component**

`dashboard/src/components/TrainingChart.tsx`:
```tsx
"use client"
import { useQuery } from "@tanstack/react-query"
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts"
import { getMetrics } from "@/lib/api"

export default function TrainingChart({ name }: { name: string }) {
  const q = useQuery({ queryKey: ["metrics", name], queryFn: () => getMetrics(name), refetchInterval: 10_000 })
  const data = (q.data ?? []).map((m) => ({ t: m.ts, v: m.value }))
  return (
    <div>
      <h3>{name}</h3>
      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={data}>
          <XAxis dataKey="t" /><YAxis /><Tooltip />
          <Line type="monotone" dataKey="v" stroke="#C9A962" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
```

- [ ] **Step 4: Brain page**

`dashboard/src/app/brain/page.tsx`:
```tsx
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
```

- [ ] **Step 5: Verify build**

Run in `dashboard/`: `npm run build`
Expected: build succeeds.

- [ ] **Step 6: Commit**

```bash
git add dashboard/ tests/agents/test_ppo_metrics.py engine/agents/ppo.py
git commit -m "feat: add Brain tab with live training curves and model registry"
```

---

### Task 18: Onboarding Tour + Dashboard Guide

**Files:**
- Create: `dashboard/src/components/Tour.tsx`, `dashboard/src/app/tour-metadata.ts`, `docs/dashboard-guide.md`
- Modify: `dashboard/src/app/layout.tsx` (mount Tour)
- Test: `npm run build`; manual first-run tour display.

**Interfaces:**
- Consumes: nothing external.
- Produces: guided tour overlay (7 steps, dismissed via localStorage `tour_done`), and `docs/dashboard-guide.md` plain-English guide.

- [ ] **Step 1: Tour component**

`dashboard/src/components/Tour.tsx`:
```tsx
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
```

Add to `globals.css`:
```css
.tour-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.7); display: flex; align-items: center; justify-content: center; z-index: 1000; }
.tour-card { background: #1A1A1A; border: 1px solid #C9A962; padding: 2rem; border-radius: 10px; max-width: 480px; }
```

- [ ] **Step 2: Mount tour in layout**

In `dashboard/src/app/layout.tsx`, add `import Tour from "./tour"` is not needed — create `dashboard/src/app/tour.tsx` with `"use client"` wrapper rendering `<Tour />`, then mount `<Tour />` inside `<body>` next to `<Providers>`.

- [ ] **Step 3: Write the guide**

`docs/dashboard-guide.md`:
```markdown
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
```

- [ ] **Step 4: Verify build**

Run in `dashboard/`: `npm run build`
Expected: build succeeds; tour renders on first visit.

- [ ] **Step 5: Commit**

```bash
git add dashboard/ docs/dashboard-guide.md
git commit -m "feat: add onboarding tour and dashboard guide"
```

---

### Task 19: End-to-End Smoke Test + README

**Files:**
- Create: `scripts/smoke.py`, `README.md`
- Test: run `scripts/smoke.py` end-to-end.

**Interfaces:**
- Consumes: every module above.
- Produces: `scripts/smoke.py` — fetches NSE + crypto bars (or synthetic fallback if network blocked), trains 2k steps, runs 10 simulated bar closes through the scheduler, prints eval report; `README.md` with quickstart.

- [ ] **Step 1: Write smoke script**

`scripts/smoke.py`:
```python
from pathlib import Path
import tempfile
import polars as pl
from engine.data.store import DataStore
from engine.data.indicators import add_indicators
from engine.env.trading_env import TradingEnv
from engine.agents.ppo import train_ppo, evaluate_ppo, load_policy
from engine.live.risk import RiskGateway
from engine.live.scheduler import Scheduler
from engine.brokers.simulator import SimulatorAdapter
from engine.eval.metrics import compute_eval_report

def synthetic_bars(n=800, seed=1):
    import numpy as np
    rng = np.random.default_rng(seed)
    px = 100.0 + np.cumsum(rng.normal(0.05, 0.4, n))
    rows = [{"time": f"2026-01-02 09:{15+i:02d}:00",
             "open": px[i], "high": px[i] + 1.0, "low": px[i] - 1.0,
             "close": px[i], "volume": 1000.0 + i} for i in range(n)]
    return add_indicators(pl.DataFrame(rows))

def main():
    tmp = Path(tempfile.mkdtemp())
    bars = synthetic_bars()
    store = DataStore(tmp / "t.db", tmp / "pq")
    store.init_schema()
    store.save_bars("RELIANCE.NS", bars, 5)
    env = TradingEnv("RELIANCE.NS", bars, seed=42)
    path = train_ppo(env, 2_000, tmp / "ck", seed=42, store=store)
    model = load_policy(path)
    rep = evaluate_ppo(model, env, episodes=3, seed=42)
    risk = RiskGateway(SimulatorAdapter(initial_cash=100_000.0),
                       {"daily_loss_limit_pct": -3.0, "max_position_pct": 30.0,
                        "max_total_exposure_pct": 90.0, "stale_data_seconds": 120,
                        "flatten_at": "15:15"})
    sim = risk.broker
    sched = Scheduler(risk, store, model, {"symbols": ["RELIANCE.NS"]})
    for row in bars.tail(20).iter_rows(named=True):
        sched.on_bar_close("RELIANCE.NS", row)
    eq = [p["equity"] for p in store.get_equity()]
    report = compute_eval_report(eq, store.get_trades())
    print("SMOKE OK", report)

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run smoke test**

Run: `python scripts/smoke.py`
Expected: prints `SMOKE OK {report}`.

- [ ] **Step 3: Write README**

`README.md`: project intro, architecture one-liner, quickstart (install, train, run API, run dashboard), pointer to `docs/dashboard-guide.md` and the spec.

- [ ] **Step 4: Commit**

```bash
git add scripts/smoke.py README.md
git commit -m "feat: add end-to-end smoke test and README"
```

---

## V1 Exit Criteria

- `pytest` green (all tasks).
- `python scripts/smoke.py` prints SMOKE OK.
- Dashboard renders all 6 tabs against a live engine API; onboarding tour appears on first visit.
- First real training run: `python -m engine.agents.ppo` (wiring via config) trains on the probe-verified NSE window (user runs longer sessions 3-4h; checkpoints resumable via `load_policy`, run metadata in `run_meta.json`).
- Follow-up work (V2/V3) gets its own spec + plan per `docs/superpowers/specs/2026-08-19-trading-bot-design.md`.

## Self-Review Notes

- Spec coverage: data (Tasks 2-5), env (6), training+checkpoints (8), brokers (9-11), risk (12), live loop (13), API (14), dashboard tabs (15-17), tutorial (18), eval (7), e2e (19). V2 optimizer and V3 LLM overlay are explicitly out of scope for this plan (separate plans).
- Determinism: seeds fixed in env (Task 6), trainer (Task 8), smoke (Task 19).
- Type consistency: `DataStore` method names stable across Tasks 2, 8, 13, 14; `BrokerAdapter` interface consistent across 9-11; `add_indicators` column contract consistent across 3, 6, 13, 19.