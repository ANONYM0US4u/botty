import numpy as np
import polars as pl
from engine.agents.ppo import train_ppo, evaluate_ppo, load_policy
from engine.env.trading_env import TradingEnv
from engine.data.indicators import add_indicators


def _env(seed=3):
    rows = [{"time": f"2026-01-02 09:{15 + i:02d}:00", "open": 100.0 + i * 0.05,
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