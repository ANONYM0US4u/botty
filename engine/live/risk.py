import math
import time
from datetime import datetime


class RiskGateway:
    def __init__(self, broker, risk_cfg: dict):
        self.broker = broker          # ONLY component allowed to hold the broker
        self.cfg = risk_cfg
        self._killed = False
        self._last_bar_ts: float | None = None
        self._retry_delays = [0.5, 1.0, 2.0]

    def set_kill_switch(self, active: bool) -> None:
        self._killed = active

    def is_killed(self) -> bool:
        return self._killed

    def set_last_bar_ts(self, ts: float | None) -> None:
        self._last_bar_ts = ts

    def check_order(self, order: dict, positions: list[dict], equity: float,
                    day_pnl: float) -> tuple[bool, str]:
        if self._killed:
            return False, "kill switch active"
        if day_pnl / equity * 100.0 <= self.cfg["daily_loss_limit_pct"]:
            return False, "daily loss limit breached"
        price = float(order.get("price", 0) or 0)
        if math.isnan(price) or math.isnan(equity):
            return False, "nan state blocks trading"
        if self._last_bar_ts is not None and \
                time.time() - self._last_bar_ts > self.cfg["stale_data_seconds"]:
            return False, "stale data blocks trading"
        if price <= 0:
            return False, "position size cannot be sized without a price"
        pos_value = abs(float(order.get("qty", 0))) * price
        if pos_value / equity * 100.0 > self.cfg["max_position_pct"]:
            return False, "position size exceeds cap"
        total = pos_value + sum(abs(float(p.get("qty", 0)) * float(p.get("price", 0)))
                                for p in positions)
        if total / equity * 100.0 > self.cfg["max_total_exposure_pct"]:
            return False, "total exposure exceeds cap"
        return True, ""

    def execute_order(self, order: dict) -> dict:
        ok, why = self.check_order(order, self.get_positions(),
                                   self.get_balance(), 0.0)
        if not ok:
            return {"status": "failed", "reason": why, "retryable": False}
        from engine.brokers.errors import TransientBrokerError, PermanentBrokerError
        for attempt, delay in enumerate([0.0] + self._retry_delays):
            if attempt:
                time.sleep(delay)
            try:
                return self.broker.place_order(order)
            except TransientBrokerError:
                continue
            except PermanentBrokerError as e:
                return {"status": "failed", "reason": str(e), "retryable": False}
        return {"status": "failed", "reason": "transient retries exhausted", "retryable": True}

    def flatten_all(self, reason: str) -> None:
        for pos in self.get_positions():
            side = "sell" if pos["qty"] > 0 else "buy"
            self.broker.place_order({"symbol": pos["symbol"], "side": side,
                                     "qty": abs(pos["qty"]), "price": 0.0,
                                     "market": True})

    def maybe_flatten_before_close(self, now: datetime) -> None:
        limit = self.cfg.get("flatten_at", "15:15")
        hh, mm = map(int, limit.split(":"))
        if now.hour > hh or (now.hour == hh and now.minute >= mm):
            self.flatten_all("pre-close flatten")

    def get_positions(self) -> list[dict]:
        return self.broker.get_positions()

    def get_orders(self) -> list[dict]:
        return self.broker.get_orders()

    def get_balance(self) -> float:
        return self.broker.get_balance()