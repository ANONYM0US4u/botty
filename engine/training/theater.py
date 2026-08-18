"""Live training theater: runs PPO in a background thread, replays each
checkpoint over a fixed validation window, and exposes traits/leaderboard."""

import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

from engine.agents.ppo import _TheaterCallback, load_policy, train_ppo
from engine.brokers.simulator import SimulatorAdapter
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
                self._status = "stopped"
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
                rep["equity_series"],
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