import polars as pl
from engine.live.scheduler import Scheduler
from engine.live.risk import RiskGateway
from engine.brokers.simulator import SimulatorAdapter
from engine.data.store import DataStore
from engine.data.indicators import add_indicators
from engine.agents.ppo import train_ppo, load_policy
from engine.env.trading_env import TradingEnv


def _setup(tmp_path):
    rows = [{"time": f"2026-01-02 09:{15 + i:02d}:00", "open": 100.0 + i * 0.1,
             "high": 101.0 + i * 0.1, "low": 99.0 + i * 0.1,
             "close": 100.5 + i * 0.1, "volume": 1000.0} for i in range(300)]
    bars = add_indicators(pl.DataFrame(rows))
    store = DataStore(tmp_path / "t.db", tmp_path / "pq")
    store.init_schema()
    store.save_bars("RELIANCE.NS", bars, 5)
    env = TradingEnv("RELIANCE.NS", bars, seed=5)
    policy_path = train_ppo(env, 1_000, tmp_path / "ck", seed=5)
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
    gate = _gate(sim)
    sched = Scheduler(gate, store, policy, {"symbols": ["RELIANCE.NS"]})
    sim.positions["RELIANCE.NS"] = 10.0
    gate.set_last_price("RELIANCE.NS", 2500.0)
    sched.flatten_all("test")
    assert sim.get_positions() == []
    assert sim.cash == 100_000.0 + 10.0 * 2500.0


def test_scheduler_never_holds_broker(tmp_path):
    store, policy = _setup(tmp_path)
    sim = SimulatorAdapter(initial_cash=100_000.0)
    sched = Scheduler(_gate(sim), store, policy, {"symbols": ["RELIANCE.NS"]})
    assert not hasattr(sched, "broker")  # structural: execution only via RiskGateway


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


def test_persist_false_writes_nothing(tmp_path):
    # M2: replay sandbox must never touch the live ledger.
    store, policy = _setup(tmp_path)
    sim = SimulatorAdapter(initial_cash=100_000.0)
    sched = Scheduler(_gate(sim), store, policy,
                      {"symbols": ["RELIANCE.NS"]}, persist=False)
    sched.on_bar_close("RELIANCE.NS", {"time": "2026-01-02 09:20:00",
                                       "open": 100.0, "close": 101.0})
    assert store.get_decisions(symbol="RELIANCE.NS", limit=1) == []
    assert store.get_equity() == []
    assert store.get_metrics("equity") == []


def test_qty_scales_with_equity(tmp_path):
    store, policy = _setup(tmp_path)
    sim = SimulatorAdapter(initial_cash=100_000.0)
    risk = RiskGateway(sim, {"daily_loss_limit_pct": -3.0, "max_position_pct": 30.0,
                             "max_total_exposure_pct": 90.0,
                             "stale_data_seconds": 120, "flatten_at": "15:15"},
                       store=store)
    sched = Scheduler(risk, store, policy, {"symbols": ["RELIANCE.NS"]})

    class AlwaysLong:
        def predict(self, obs, deterministic=True):
            import numpy as np
            return (np.array(1), None)
        device = "cpu"
    sched.policy = AlwaysLong()
    sched.on_bar_close("RELIANCE.NS", {"time": "2026-01-02 09:20:00",
                                       "open": 100.0, "close": 100.0})
    sched.on_bar_close("RELIANCE.NS", {"time": "2026-01-02 09:25:00",
                                       "open": 100.5, "close": 100.5})
    sched.on_bar_close("RELIANCE.NS", {"time": "2026-01-02 09:30:00",
                                       "open": 101.0, "close": 101.0})
    fills = store.get_trades()
    assert fills and fills[0]["side"] == "buy"
    # 30% of 100k at price 100 => 300 units, not 1
    assert fills[0]["qty"] == 300
    assert sim.get_positions()[0]["qty"] == 300.0


def test_obs_advances_with_bar_index(tmp_path):
    # H2 regression: two different bar times must produce different
    # observations (the old env anchored every decision at bars[120]).
    store, policy = _setup(tmp_path)
    sim = SimulatorAdapter(initial_cash=100_000.0)
    sched = Scheduler(_gate(sim), store, policy, {"symbols": ["RELIANCE.NS"]})

    seen = {}
    class ObsSniff:
        device = "cpu"
        def predict(self, obs, deterministic=True):
            seen.setdefault("obs", []).append(obs.copy())
            import numpy as np
            return (np.array(0), None)
    sched.policy = ObsSniff()
    sched.on_bar_close("RELIANCE.NS", {"time": "2026-01-02 09:20:00",
                                       "open": 100.0, "close": 101.0})
    sched.on_bar_close("RELIANCE.NS", {"time": "2026-01-02 09:25:00",
                                       "open": 100.5, "close": 101.5})
    import numpy as np
    assert len(seen["obs"]) == 2
    assert not np.allclose(seen["obs"][0], seen["obs"][1])