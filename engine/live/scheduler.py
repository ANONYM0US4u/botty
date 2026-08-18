from datetime import datetime
import time
from engine.env.trading_env import TradingEnv
from engine.data.indicators import add_indicators


class Scheduler:
    def __init__(self, risk_gateway, store, policy, cfg: dict):
        self.risk = risk_gateway          # execution ONLY via RiskGateway
        self.store = store
        self.policy = policy
        self.symbols = cfg.get("symbols", [])
        self.window = cfg.get("window_bars", 120)

    def on_bar_close(self, symbol: str, bar: dict) -> dict:
        ts = bar.get("time", "")
        try:
            bar_dt = datetime.fromisoformat(str(ts))
        except ValueError:
            bar_dt = datetime.now()
        self.risk.set_last_bar_ts(time.time())
        self.risk.maybe_flatten_before_close(bar_dt)
        bars = add_indicators(self.store.get_bars(symbol))
        if bars.height < self.window + 2:
            return {"action": "hold", "reason": "insufficient bars"}
        env = TradingEnv(symbol, bars, window=self.window, seed=0)
        obs, _ = env.reset()
        action, _ = self.policy.predict(obs, deterministic=True)
        target = {0: "flat", 1: "long", 2: "short"}[int(action)]
        summary = {"action": target, "reason": ""}
        if target != "flat":
            order = {"symbol": symbol,
                     "side": "buy" if target == "long" else "sell",
                     "qty": 1, "price": float(bar["close"])}
            res = self.risk.execute_order(order)
            if res.get("status") == "failed":
                summary = {"action": "flat", "reason": res.get("reason", "risk-gated")}
        self.store.append_decision({"ts": ts, "symbol": symbol,
                                    "action": summary["action"], "probs": "[]",
                                    "features": "[]", "attribution": "[]"})
        eq = self.risk.get_balance()
        self.store.append_equity(ts, eq)
        self.store.append_metric("equity", eq, ts)
        return summary

    def flatten_all(self, reason: str) -> None:
        self.risk.flatten_all(reason)