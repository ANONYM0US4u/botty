"""Live training theater: runs PPO in a background thread, replays each
checkpoint over a fixed validation window, and exposes traits/leaderboard."""

import shutil
import threading
import time
import os
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
        self.markets = {"crypto": set(cfg["instruments"]["crypto"]),
                        "nse": set(cfg["instruments"]["stocks"])}
        self._market_of = {s: m for m, syms in self.markets.items()
                           for s in syms}
        self._max_runs = int(cfg.get("theater", {}).get("max_runs_kept", 3))
        self._stop = threading.Event()
        self._train_thread: threading.Thread | None = None
        self._train_threads: list[threading.Thread] = []
        self._active_children = 0
        self._replay_thread: threading.Thread | None = None
        self._status = "idle"
        self._symbol = None
        self._market = None
        self._run_id = None
        self._steps = 0
        self._phase = ""
        self._error = ""
        self._lock = threading.Lock()
        self._lb_cache = None
        self._lb_ready = True
        self._lb_thread: threading.Thread | None = None

    # ---- public API -----------------------------------------------------

    def state(self) -> dict:
        with self._lock:
            return {"status": self._status, "symbol": self._symbol,
                    "run_id": self._run_id, "steps": self._steps,
                    "phase": self._phase, "error": self._error}

    def start(self, symbol: str) -> dict:
        market = self._market_of.get(symbol)
        if market is None:
            return {"error": f"symbol {symbol} not configured"}
        with self._lock:
            if self._status in ("running", "starting", "stopping"):
                raise RuntimeError("already running")
        try:
            bars = self.fetch_bars(symbol)
        except Exception as e:
            return {"error": f"fetch failed: {e}"}
        if bars is None or bars.height == 0:
            return {"error": "fetch returned no bars"}
        with self._lock:
            if self._status in ("running", "starting", "stopping"):
                raise RuntimeError("already running")
            self.store.save_bars(symbol, bars, 5)
            self._symbol = symbol
            self._market = market
            self._run_id = f"{symbol}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            self._steps = 0
            self._error = ""
            self._status = "starting"
            run_dir = self.ck_root / market / self._run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            self._stop.clear()
            self._lb_cache = None
            self._lb_ready = False
            self._prune(market)
            self.store.clear_metrics()
        evo = self.cfg.get("training", {}).get("evolution", {})
        pop = max(1, int(evo.get("population_size", 1)))
        if evo.get("enabled"):
            import torch
            torch.set_num_threads(int(evo.get("torch_threads", 2)))
        parent = self._find_best_parent(market, symbol) if evo.get("enabled") else None
        with self._lock:
            self._active_children = pop
            self._train_threads = []
        for child_id in range(pop):
            t = threading.Thread(
                target=self._child_loop,
                args=(child_id, symbol, bars, run_dir, self._run_id, parent),
                daemon=True)
            self._train_threads.append(t)
            t.start()
        self._train_thread = self._train_threads[0]
        with self._lock:
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
        if not self.wait_idle(30):
            return {"error": "training still active; try again shortly"}
        with self._lock:
            if self._run_id and self._market:
                shutil.rmtree(self.ck_root / self._market / self._run_id,
                              ignore_errors=True)
            self._status = "idle"
            self._symbol = None
            self._market = None
            self._run_id = None
            self._steps = 0
            self._phase = ""
            self._error = ""
            self._lb_cache = None
            self._lb_ready = True
            self._train_thread = None
            self._train_threads = []
            self._active_children = 0
        return self.state()

    def leaderboard(self) -> list[dict]:
        with self._lock:
            if self._lb_ready:
                return self._lb_cache or []
            if self._lb_thread is not None and self._lb_thread.is_alive():
                return self._lb_cache or []
            self._lb_thread = threading.Thread(target=self._lb_worker,
                                               daemon=True)
            self._lb_thread.start()
            return self._lb_cache or []

    def wait_idle(self, timeout: float) -> bool:
        ts = self._train_threads or ([self._train_thread] if self._train_thread else [])
        if not ts:
            return True
        deadline = time.time() + timeout
        for t in ts:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            t.join(remaining)
        return all(not t.is_alive() for t in ts)

    # ---- internals ------------------------------------------------------

    def _child_loop(self, child_id: int, symbol, bars, run_dir: Path, run_id: str,
                    parent: Path | None):
        cost_pct = (self.cfg.get("brokers", {}).get("slippage_bps", 2.0) / 10_000.0)
        seed = self.cfg["training"]["seed"] + child_id
        evo = self.cfg.get("training", {}).get("evolution", {})
        mutation_std = float(evo.get("mutation_std", 0.0))
        child_dir = run_dir / f"child_{child_id}"
        child_dir.mkdir(parents=True, exist_ok=True)
        try:
            env = TradingEnv(symbol, bars, window=self.cfg["training"].get(
                "window_bars", 120), seed=seed,
                cost_pct=cost_pct,
                position_pct=self.cfg["training"].get("position_pct", 0.30),
                dd_penalty=self.cfg.get("reward", {}).get(
                    "drawdown_penalty", 0.1))
            total = self.cfg["training"]["total_timesteps"]
            steps = 0
            while steps < total and not self._stop.is_set():
                self._set_phase(f"child {child_id} training {steps}/{total}")
                chunk = min(_CHUNK, total - steps)
                cb = _TheaterCallback(self.emitter, self.store,
                                      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                      stop_check=self._stop.is_set,
                                      tag=f"c{child_id}" if child_id else "")
                if steps == 0:
                    from stable_baselines3 import PPO
                    model = None
                    if parent is not None:
                        try:
                            model = load_policy(parent, env=env)
                            self._mutate(model, mutation_std)
                        except Exception:
                            model = None
                    if model is None:
                        model = PPO("MlpPolicy", env, seed=seed,
                                    verbose=0, n_steps=_CHUNK, batch_size=64)
                else:
                    model = load_policy(child_dir / "latest.zip", env=env)
                model.learn(total_timesteps=chunk, callback=cb)
                if self._stop.is_set():
                    break  # abort without saving/replaying a half-stop checkpoint
                steps += chunk
                self._set_steps(steps)
                path = child_dir / f"ppo_{steps}.zip"
                from engine.agents.ppo import atomic_save
                atomic_save(model, path)
                shutil.copy(path, child_dir / "latest.zip")
                ev = self._evaluate_checkpoint(path)
                self.store.append_checkpoint(
                    {"path": str(path), "reward": ev.get("mean_reward", 0.0),
                     "sharpe": ev.get("sharpe", 0.0), "ts": steps,
                     "run_id": self._run_id,
                     "model_id": f"ppo-{self._run_id}",
                     "git_commit": "", "config_hash": ""})
                with self._lock:
                    self._lb_cache = None
                    self._lb_ready = False
                self._spawn_replay(symbol, path)
            with self._lock:
                self._active_children -= 1
                done = self._active_children <= 0
            if done and not self._stop.is_set():
                self._set_phase("promoting best checkpoint...")
                self._promote_best(run_dir)
                self._set_phase("completed")
            with self._lock:
                if self._status != "error":
                    self._status = "stopped"
        except Exception as e:
            with self._lock:
                if self._run_id == run_id:
                    self._status = "error"
                    self._error = str(e)
            if self.emitter is not None and hasattr(self.emitter, "emit_json"):
                self.emitter.emit_json("theater/error", {"error": str(e)})

    def _find_best_parent(self, market: str, symbol: str) -> Path | None:
        root = self.ck_root / market
        if not root.exists():
            return None
        best_fit = -1e9
        best_path = None
        for d in sorted(root.iterdir()):
            if not d.is_dir() or not d.name.startswith(symbol + "-"):
                continue
            p = d / "latest.zip"
            if not p.exists():
                continue
            row = self._evaluate_checkpoint(p)
            f = row.get("fitness", row.get("sharpe", -1e9))
            if f > best_fit and f > -1e8:
                best_fit = f
                best_path = p
        return best_path

    @staticmethod
    def _mutate(model, std: float) -> None:
        if std <= 0:
            return
        import torch
        with torch.no_grad():
            for p in model.policy.parameters():
                p.add_(torch.randn_like(p) * std)

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
                                       "window_bars", 120)},
                                  persist=False)
                decisions = []
                for row in window_bars:
                    summary = sched.on_bar_close(symbol, row)
                    decisions.append({"ts": row["time"], "symbol": symbol,
                                      "action": summary["action"],
                                      "probs": str(summary.get("probs", []))})
                fills = risk.get_fills()
                traits = compute_traits(fills, decisions, self.cfg["risk"])
                self.store.append_traits(traits)
                if self.emitter is not None and hasattr(self.emitter, "emit_json"):
                    self.emitter.emit_json("theater/traits", traits)
            except Exception:
                pass
        self._replay_thread = threading.Thread(target=run, daemon=True)
        self._replay_thread.start()

    def _all_checkpoints(self, run_dir: Path) -> list[Path]:
        return sorted(run_dir.rglob("ppo_*.zip"))

    def _lb_files(self, root: Path) -> list[Path]:
        files = []
        for d in sorted(root.iterdir()):
            if d.is_dir() and d.name.startswith("child_"):
                cks = sorted(d.glob("ppo_*.zip"),
                             key=lambda p: int(p.stem.split("_")[1]))
                if cks:
                    files.append(cks[-1])
            elif d.is_file() and d.name.startswith("ppo_"):
                files.append(d)
        return sorted(files)

    def _promote_best(self, run_dir: Path) -> None:
        try:
            evo = self.cfg.get("training", {}).get("evolution", {})
            min_pct = float(evo.get("promotion_min_pct", 0.0))
            total = int(self.cfg["training"]["total_timesteps"])
            min_steps = int(total * min_pct)
            best = -1e9
            best_path = None
            for f in self._all_checkpoints(run_dir):
                if int(f.stem.split("_")[1]) < min_steps:
                    continue
                row = self._evaluate_checkpoint(f)
                s = row.get("fitness", row.get("sharpe", -1e9))
                if s > best and s > -1e8:
                    best = s
                    best_path = f
            if best_path is not None:
                tmp = run_dir / "latest.zip.tmp"
                shutil.copy(best_path, tmp)
                os.replace(tmp, run_dir / "latest.zip")
        except Exception:
            pass

    def _lb_worker(self) -> None:
        while True:
            with self._lock:
                run_id = self._run_id
                market = self._market
            root = self.ck_root / market / run_id if (run_id and market) else None
            if root is None or not root.exists():
                with self._lock:
                    self._lb_ready = True
                return
            files = self._lb_files(root)
            rows = [self._evaluate_checkpoint(f) for f in files]
            rows.sort(key=lambda r: r.get("fitness", r.get("sharpe", -1e9)),
                      reverse=True)
            with self._lock:
                if (self._run_id != run_id or self._market != market
                        or not root.exists()):
                    self._lb_ready = True
                    return
                if self._lb_files(root) != files:
                    continue
                self._lb_cache = rows
                self._lb_ready = True
                return

    def _evaluate_checkpoint(self, ck: Path) -> dict:
        try:
            policy = load_policy(ck)
            bars = self.store.get_bars(self._symbol or "")
            window = bars.tail(_VALIDATION_WINDOW) if bars.height > \
                _VALIDATION_WINDOW else bars
            env = TradingEnv(self._symbol or "X", window,
                             window=self.cfg["training"].get("window_bars", 120),
                             seed=0,
                             position_pct=self.cfg["training"].get(
                                 "position_pct", 0.30),
                             cost_pct=(self.cfg.get("brokers", {}).get(
                                 "slippage_bps", 2.0) / 10_000.0))
            obs, _ = env.reset(options={"start_idx": env.window})
            curve = [env.unwrapped.initial_cash]
            trades = []
            prev_units = 0.0
            done = False
            while not done:
                price = env.unwrapped._price()
                action, _ = policy.predict(env.unwrapped._obs(), deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(int(action))
                done = terminated or truncated
                units = env.unwrapped.position
                if abs(units - prev_units) > 1e-9:
                    side = "buy" if units > prev_units else "sell"
                    trades.append({"order_id": f"eval-{len(trades)}", "side": side,
                                   "qty": abs(units - prev_units), "price": price,
                                   "ts": ""})
                    prev_units = units
                curve.append(env.unwrapped.equity)
            report = compute_eval_report(curve, trades)
            fw = self.cfg.get("training", {}).get("fitness_weights", {})
            sharpe = report.get("sharpe", 0.0)
            win_rate = report.get("win_rate", 0.0)
            fitness = (sharpe * float(fw.get("sharpe", 1.0))
                       + win_rate * float(fw.get("win_rate", 0.0)))
            return {"path": str(ck), "sharpe": sharpe, "win_rate": win_rate,
                    "mean_reward": float(curve[-1] - curve[0]), "traits": {},
                    "fitness": fitness}
        except Exception:
            return {"path": str(ck), "sharpe": -1e9, "win_rate": 0.0,
                    "mean_reward": 0.0, "traits": {}, "fitness": -1e9}

    def _prune(self, market: str) -> None:
        root = self.ck_root / market
        if not root.exists():
            return
        dirs = sorted([d for d in root.iterdir() if d.is_dir()],
                      key=lambda d: d.name)
        for old in dirs[:-self._max_runs]:
            shutil.rmtree(old, ignore_errors=True)

    def _set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase

    def _set_steps(self, steps: int) -> None:
        with self._lock:
            self._steps = steps