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