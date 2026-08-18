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
    sched = Scheduler(_gate(sim), store, policy, {"symbols": ["RELIANCE.NS"]})
    sim.positions["RELIANCE.NS"] = 10.0
    sched.flatten_all("test")
    assert sim.get_positions() == []


def test_scheduler_never_holds_broker(tmp_path):
    store, policy = _setup(tmp_path)
    sim = SimulatorAdapter(initial_cash=100_000.0)
    sched = Scheduler(_gate(sim), store, policy, {"symbols": ["RELIANCE.NS"]})
    assert not hasattr(sched, "broker")  # structural: execution only via RiskGateway