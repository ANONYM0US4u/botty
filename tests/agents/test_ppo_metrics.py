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


def test_atomic_save_renames(tmp_path):
    from pathlib import Path
    import gymnasium as gym
    from stable_baselines3 import PPO
    from gymnasium import spaces
    import numpy as np
    from engine.agents.ppo import atomic_save

    class FakeEnv(gym.Env):
        def __init__(self):
            self.action_space = spaces.Discrete(3)
            self.observation_space = spaces.Box(-1, 1, (4,), dtype=np.float32)
        def reset(self, *, seed=None, options=None): return np.zeros(4, dtype=np.float32), {}
        def step(self, a): return np.zeros(4, dtype=np.float32), 0.0, False, False, {}
    model = PPO("MlpPolicy", FakeEnv(), seed=0, n_steps=64, batch_size=32)
    path = tmp_path / "ppo_0_100.zip"
    atomic_save(model, path)
    assert path.exists() and not Path(str(path) + ".tmp").exists()


def test_theater_callback_emits_probs(monkeypatch, tmp_path):
    import gymnasium as gym
    import numpy as np
    from stable_baselines3 import PPO
    from gymnasium import spaces
    from engine.agents.ppo import _TheaterCallback
    from engine.data.store import DataStore

    class FakeEnv(gym.Env):
        def __init__(self):
            self.action_space = spaces.Discrete(3)
            self.observation_space = spaces.Box(-1, 1, (4,), dtype=np.float32)
        def reset(self, *, seed=None, options=None): return np.zeros(4, dtype=np.float32), {}
        def step(self, a): return np.zeros(4, dtype=np.float32), 0.0, False, False, {}
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