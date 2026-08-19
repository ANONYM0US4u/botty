import threading
import time
from datetime import datetime
from pathlib import Path

import polars as pl

from engine.agents.ppo import load_policy
from engine.brokers.simulator import SimulatorAdapter
from engine.live.risk import RiskGateway
from engine.live.scheduler import Scheduler

_POLL_SECONDS = 60


class BotMode:
    """Unified market x mode state machine.

    market: crypto | nse (scoped checkpoint dirs, never mixed)
    mode:   idle | train (theater) | trade (live paper-trading loop)
    """

    def __init__(self, store, emitter, cfg: dict, theater, fetch_bars):
        self.store = store
        self.emitter = emitter
        self.cfg = cfg
        self.theater = theater
        self.fetch_bars = fetch_bars
        self.markets = {"crypto": list(cfg["instruments"]["crypto"]),
                        "nse": list(cfg["instruments"]["stocks"])}
        self._market_of = {s: m for m, syms in self.markets.items()
                           for s in syms}
        self._risk = RiskGateway(
            SimulatorAdapter(
                slippage_bps=cfg["brokers"].get("slippage_bps", 2.0),
                latency_bars=cfg["brokers"].get("latency_bars", 1)),
            cfg["risk"])
        self._lock = threading.RLock()
        self._market = "crypto"
        self._mode = "idle"
        self._switching = False
        self._trade_stop = threading.Event()
        self._trade_thread = None
        self._trade_error = ""
        self._last_poll = None
        self._last_skip = {}
        self._last_bar_seen = {}
        self._policy_cache = {}

    # ---------------- state ----------------

    def state(self) -> dict:
        with self._lock:
            return {"market": self._market, "mode": self._mode,
                    "switching": self._switching,
                    "markets": list(self.markets.keys()),
                    "trade": self._trade_state(),
                    "train": self.theater.state()}

    def _trade_state(self) -> dict:
        running = (self._trade_thread is not None
                   and self._trade_thread.is_alive())
        return {"running": running, "error": self._trade_error,
                "last_poll": self._last_poll,
                "skips": dict(self._last_skip)}

    def can_train(self) -> bool:
        with self._lock:
            return self._mode != "trade"

    # ---------------- switching ----------------

    def set_market(self, market: str) -> dict:
        with self._lock:
            if market not in self.markets:
                return {"error": f"unknown market {market}"}
            if market == self._market and self._trade_thread is None:
                return self.state()
            self._stop_trade_locked()
            self._market = market
            if self._mode == "trade":
                self._start_trade_locked()
            return self.state()

    def set_mode(self, mode: str) -> dict:
        with self._lock:
            if mode not in ("idle", "train", "trade"):
                return {"error": f"unknown mode {mode}"}
            if mode == self._mode:
                return self.state()
            self._stop_trade_locked()
            if mode == "trade":
                self.theater.stop()
                self.theater.wait_idle(30)
                self._start_trade_locked()
            self._mode = mode
            return self.state()

    def _stop_trade_locked(self) -> None:
        if self._trade_thread is not None:
            self._trade_stop.set()
            self._trade_thread.join(timeout=70)
            self._trade_thread = None
        self._trade_stop.clear()
        self._policy_cache.clear()

    def _start_trade_locked(self) -> None:
        self._trade_error = ""
        self._last_bar_seen.clear()
        self._trade_thread = threading.Thread(
            target=self._trade_loop, args=(self._market,), daemon=True)
        self._trade_thread.start()

    # ---------------- trade loop ----------------

    def _trade_loop(self, market: str) -> None:
        symbols = list(self.markets[market])
        while not self._trade_stop.is_set():
            try:
                self._poll(market, symbols)
            except Exception as e:
                self._trade_error = str(e)
                if self.emitter is not None:
                    self.emitter.emit_json(
                        "trade/error", {"error": str(e)})
            self._trade_stop.wait(_POLL_SECONDS)

    def _poll(self, market: str, symbols: list[str]) -> None:
        for symbol in symbols:
            if self._trade_stop.is_set():
                return
            try:
                bars = self.fetch_bars(symbol)
            except Exception:
                self._last_skip[symbol] = "fetch failed (network down, no cache)"
                continue
            if bars is None or bars.height < 2:
                self._last_skip[symbol] = "no bars"
                continue
            ck = self._latest_policy(symbol)
            if ck is None:
                self._last_skip[symbol] = "no trained policy yet"
                continue
            self._last_skip.pop(symbol, None)
            self.store.save_bars(symbol, bars, 5)
            policy = self._policy_for(symbol, ck)
            sched = Scheduler(
                self._risk, self.store, policy,
                {"symbols": [symbol],
                 "window_bars": self.cfg["training"].get("window_bars", 120)})
            last = self._last_bar_seen.get(symbol)
            new_bars = (bars.filter(pl.col("time") > last)
                        if last else bars.tail(400))
            for row in new_bars.to_dicts():
                if self._trade_stop.is_set():
                    return
                sched.on_bar_close(symbol, row)
                self._last_bar_seen[symbol] = row["time"]
            if self.emitter is not None:
                dec = self.store.get_decisions(symbol=symbol, limit=1)
                if dec:
                    self.emitter.emit_json("trade/decision", dec[0])
        self._last_poll = datetime.now().isoformat(timespec="seconds")
        if self.emitter is not None:
            self.emitter.emit_json("trade/state", {
                "market": market,
                "last_poll": self._last_poll,
                "fills": len(self.store.get_trades()),
                "skips": dict(self._last_skip)})

    def _latest_policy(self, symbol: str) -> Path | None:
        market = self._market_of.get(symbol)
        if market is None:
            return None
        root = self.theater.ck_root / market
        if not root.exists():
            return None
        runs = [d for d in root.iterdir()
                if d.is_dir() and d.name.startswith(symbol + "-")]
        if not runs:
            return None
        newest = max(runs, key=lambda d: d.name)
        p = newest / "latest.zip"
        return p if p.exists() else None

    def _policy_for(self, symbol: str, ck: Path):
        mtime = ck.stat().st_mtime
        hit = self._policy_cache.get(symbol)
        if hit is not None and hit[0] == str(ck) and hit[1] == mtime:
            return hit[2]
        policy = load_policy(ck)
        self._policy_cache[symbol] = (str(ck), mtime, policy)
        return policy