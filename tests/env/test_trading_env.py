import polars as pl
import numpy as np
from engine.env.trading_env import TradingEnv
from engine.data.indicators import add_indicators


def _bars(n=400):
    rows = [{"time": f"2026-01-02 09:{15 + i:02d}:00", "open": 100.0 + i * 0.1,
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
    env_full.reset(options={"start_idx": 200})
    truncated = _bars(300)  # same bars, cut short
    env_trunc = TradingEnv("RELIANCE.NS", truncated, seed=3)
    env_trunc.reset(options={"start_idx": 200})
    assert np.allclose(env_full._obs(), env_trunc._obs())


def test_long_profits_in_uptrend():
    bars = _bars()
    env = TradingEnv("RELIANCE.NS", bars, seed=2, cost_pct=0.0)
    env.reset()
    for _ in range(50):
        env.step(1)
    assert env.equity > env.initial_cash


def test_start_idx_anchors_observation_at_that_bar():
    # H2 regression: reset(options={"start_idx": i}) must produce the obs
    # whose LAST window row is bar i (no lookahead, no staleness).
    bars = _bars(400)
    env = TradingEnv("RELIANCE.NS", bars, seed=1)
    env.reset(options={"start_idx": 200})
    assert env._idx == 200
    arr = bars.select(["open", "high", "low", "close", "volume", "ema9"]).to_numpy()
    obs = env._obs()
    nf = env.n_features
    tail = obs[-nf - 2:-2]
    assert np.allclose(tail[:6], arr[200][:6])
    obs_200 = env._obs()
    env.reset(options={"start_idx": 250})
    assert not np.allclose(obs_200, env._obs())  # obs advances with the bar


def test_reset_clamps_invalid_start_idx():
    bars = _bars(400)
    env = TradingEnv("RELIANCE.NS", bars, seed=1)
    env.reset(options={"start_idx": 5})        # below window -> clamp
    assert env._idx == env.window
    env.reset(options={"start_idx": 9999})     # beyond bars -> clamp
    assert env._idx == env.window


def test_missing_indicators_auto_added():
    # M11: raw OHLCV bars must not crash the env with ColumnNotFoundError.
    rows = [{"time": f"2026-01-02 09:{15 + i:02d}:00", "open": 100.0 + i,
             "high": 101.0 + i, "low": 99.0 + i, "close": 100.5 + i,
             "volume": 1000.0} for i in range(200)]
    env = TradingEnv("RELIANCE.NS", pl.DataFrame(rows), seed=1)
    assert env._obs().shape[0] == 120 * env.n_features + 2
    assert not np.isnan(env._obs()).any()


def test_observation_never_contains_inf_or_nan():
    rows = [{"time": f"2026-01-02 09:{15 + i:02d}:00", "open": 0.0, "high": 0.0,
             "low": 0.0, "close": 0.0, "volume": 0.0} for i in range(200)]
    env = TradingEnv("RELIANCE.NS", pl.DataFrame(rows), seed=1)
    obs = env._obs()
    assert np.isfinite(obs).all()