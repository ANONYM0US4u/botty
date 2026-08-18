import polars as pl
from engine.agents.ppo import train_ppo
from engine.env.trading_env import TradingEnv
from engine.data.indicators import add_indicators
from engine.data.store import DataStore


def test_training_emits_metrics(tmp_path):
    rows = [{"time": f"2026-01-02 09:{15 + i:02d}:00", "open": 100.0 + i * 0.05,
             "high": 101.0 + i * 0.05, "low": 99.0 + i * 0.05,
             "close": 100.5 + i * 0.05, "volume": 1000.0} for i in range(300)]
    bars = add_indicators(pl.DataFrame(rows))
    env = TradingEnv("RELIANCE.NS", bars, seed=9)
    store = DataStore(tmp_path / "t.db", tmp_path / "pq")
    store.init_schema()
    train_ppo(env, 500, tmp_path / "ck", seed=9, store=store)
    assert len(store.get_metrics("ep_rew_mean")) > 0