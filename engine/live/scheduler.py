from datetime import datetime
import math
import time
import polars as pl
from stable_baselines3.common.utils import obs_as_tensor
from engine.env.trading_env import TradingEnv
from engine.data.indicators import add_indicators


class Scheduler:
    def __init__(self, risk_gateway, store, policy, cfg: dict, persist: bool = True):
        self.risk = risk_gateway          # execution ONLY via RiskGateway
        self.store = store
        self.policy = policy
        self.symbols = cfg.get("symbols", [])
        self.window = cfg.get("window_bars", 120)
        self.qty_pct = float(cfg.get("qty_pct", 30.0)) / 100.0
        self.persist = persist            # False = replay sandbox (no ledger writes)

    def on_bar_close(self, symbol: str, bar: dict) -> dict:
        ts = bar.get("time", "")
        try:
            bar_dt = datetime.fromisoformat(str(ts))
        except ValueError:
            bar_dt = datetime.now()
        self.risk.set_last_bar_ts(time.time())
        self.risk.set_last_price(symbol, float(bar.get("close", 0.0) or 0.0))
        self.risk.maybe_flatten_before_close(bar_dt, symbol)
        self.risk.on_bar_close(symbol, bar)  # settle queued orders at this bar
        bars = add_indicators(self.store.get_bars(symbol))
        if bars.height < self.window + 2:
            return {"action": "hold", "reason": "insufficient bars"}
        env = TradingEnv(symbol, bars, window=self.window, seed=0)
        idx = None
        try:
            hit = bars.with_row_index().filter(pl.col("time") == ts)
            if hit.height == 1:
                idx = int(hit["index"][0])
        except Exception:
            idx = None
        obs, _ = env.reset(options={"start_idx": idx})
        action, _ = self.policy.predict(obs, deterministic=True)
        probs = []
        try:
            obs_t = obs_as_tensor(obs, self.policy.device).unsqueeze(0)
            dist = self.policy.policy.get_distribution(obs_t)
            probs = [float(p) for p in
                     dist.distribution.probs.detach().mean(axis=0)]
        except Exception:
            probs = []
        target = {0: "flat", 1: "long", 2: "short"}[int(action)]
        summary = {"action": target, "reason": "", "probs": probs}
        price = float(bar.get("close", 0.0) or 0.0)
        pos_units = 0.0
        for p in self.risk.get_positions():
            if p.get("symbol") == symbol:
                pos_units = float(p.get("qty", 0.0) or 0.0)
        equity = self.risk.get_balance() + pos_units * price
        target_units = self.qty_pct * equity / price if price > 0 else 0.0
        delta = target_units - pos_units
        if abs(delta) * price / equity <= 0.005:
            summary = {"action": "flat", "reason": "position at target", "probs": probs}
        elif delta > 0:
            qty = math.floor(delta * 1e6) / 1e6
            if qty <= 0:
                summary = {"action": "flat", "reason": "position too small", "probs": probs}
            else:
                order = {"symbol": symbol, "side": "buy", "qty": qty, "price": price}
                res = self.risk.execute_order(order)
                if res.get("status") == "failed":
                    summary = {"action": "flat", "reason": res.get("reason", "risk-gated")}
        else:
            qty = math.floor(-delta * 1e6) / 1e6
            if qty <= 0:
                summary = {"action": "flat", "reason": "position too small", "probs": probs}
            else:
                order = {"symbol": symbol, "side": "sell", "qty": qty, "price": price}
                res = self.risk.execute_order(order)
                if res.get("status") == "failed":
                    summary = {"action": "flat", "reason": res.get("reason", "risk-gated")}
        if self.persist:
            self.store.append_decision({"ts": ts, "symbol": symbol,
                                        "action": summary["action"], "probs": str(probs),
                                        "features": "[]", "attribution": "[]"})
            eq = self.risk.get_balance()
            self.store.append_equity(ts, eq)
            self.store.append_metric("equity", eq, ts)
        return summary

    def flatten_all(self, reason: str) -> None:
        self.risk.flatten_all(reason)