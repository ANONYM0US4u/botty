from pathlib import Path
import hashlib
import json
import subprocess
import time
from collections import deque
from datetime import datetime
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class _MetricCallback(BaseCallback):
    def __init__(self, emitter, store, ts_name: str):
        super().__init__()
        self.emitter = emitter
        self.store = store
        self.ts_name = ts_name
        self._rewards = deque(maxlen=100)

    def _on_step(self) -> bool:
        self._rewards.extend(float(r) for r in self.locals["rewards"])
        if self.n_calls % 100 == 0:
            self._emit("ep_rew_mean",
                       float(np.mean(self._rewards)) if self._rewards else 0.0)
            self._emit("entropy", float(self._policy_entropy()))
            for k in ("policy_loss", "value_loss"):
                v = self.model.logger.name_to_value.get(f"train/{k}")
                if v is None:
                    v = self.model.logger.name_to_value.get(k)
                if v is not None:
                    self._emit(k, float(v))
        return True

    def _policy_entropy(self) -> float:
        try:
            obs = self.locals["obs_tensor"][-1:]
            dist = self.model.policy.get_distribution(obs)
            return float(dist.distribution.entropy().mean().item())
        except Exception:
            return 0.0

    def _emit(self, name: str, value: float) -> None:
        if self.emitter:
            self.emitter.emit(name, value)
        if self.store:
            self.store.append_metric(name, value, self.ts_name)


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
              cfg: dict | None = None, emitter=None) -> Path:
    cdir = Path(checkpoint_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    cfg_hash = hashlib.sha1(json.dumps(cfg or {}, sort_keys=True).encode()).hexdigest()[:12]
    meta = _run_metadata(cfg_hash)
    (cdir / "run_meta.json").write_text(json.dumps(meta))
    model = PPO("MlpPolicy", env, seed=seed, verbose=0, n_steps=2048, batch_size=64)
    cb = _MetricCallback(emitter, store,
                         datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    steps = 0
    last_path = None
    while steps < total_timesteps:
        chunk = min(save_every, total_timesteps - steps)
        model.learn(total_timesteps=chunk, progress_bar=False, callback=cb)
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