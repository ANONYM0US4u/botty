import math
import time
from datetime import datetime


class RiskGateway:
    def __init__(self, broker, risk_cfg: dict, store=None,
                 flatten_symbols: set[str] | None = None):
        self.broker = broker          # ONLY component allowed to hold the broker
        self.cfg = risk_cfg
        self.store = store            # optional: persist fills/orders on execution
        self.flatten_symbols = flatten_symbols  # None = flatten everything
        self._killed = False
        self._last_bar_ts: float | None = None
        self._last_prices: dict[str, float] = {}
        self._retry_delays = [0.5, 1.0, 2.0]
        self._fills: list[dict] = []

    def get_fills(self) -> list[dict]:
        return list(self._fills)

    def set_kill_switch(self, active: bool) -> None:
        self._killed = active

    def is_killed(self) -> bool:
        return self._killed

    def set_last_bar_ts(self, ts: float | None) -> None:
        self._last_bar_ts = ts

    def set_last_price(self, symbol: str, price: float) -> None:
        self._last_prices[symbol] = float(price)

    def _marked_equity(self) -> float:
        total = self.broker.get_balance()
        for p in self.broker.get_positions():
            price = self._last_prices.get(p["symbol"]) \
                or float(p.get("price", 0) or 0)
            if price > 0:
                total += float(p.get("qty", 0) or 0) * price
        return total

    def _pos_value(self, p: dict) -> float:
        price = self._last_prices.get(p["symbol"]) \
            or float(p.get("price", 0) or 0)
        return abs(float(p.get("qty", 0) or 0)) * price

    def check_order(self, order: dict, positions: list[dict],
                    equity: float | None = None, day_pnl: float = 0.0) -> tuple[bool, str]:
        if self._killed:
            return False, "kill switch active"
        if equity is None:
            equity = self._marked_equity()
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
        qty = float(order.get("qty", 0) or 0)
        side = order.get("side", "buy")
        sym = order.get("symbol", "")
        cur_qty = 0.0
        for p in positions:
            if p.get("symbol") == sym:
                cur_qty = float(p.get("qty", 0) or 0)
                break
        post_qty = cur_qty + qty if side == "buy" else cur_qty - qty
        post_value = abs(post_qty) * price
        if post_value / equity * 100.0 > self.cfg["max_position_pct"]:
            return False, "position size exceeds cap"
        other = sum(self._pos_value(p)
                    for p in positions if p.get("symbol") != sym)
        if (post_value + other) / equity * 100.0 > self.cfg["max_total_exposure_pct"]:
            return False, "total exposure exceeds cap"
        return True, ""

    def execute_order(self, order: dict) -> dict:
        ok, why = self.check_order(order, self.get_positions())
        if not ok:
            return {"status": "failed", "reason": why, "retryable": False}
        from engine.brokers.errors import TransientBrokerError, PermanentBrokerError
        for attempt, delay in enumerate([0.0] + self._retry_delays):
            if attempt:
                time.sleep(delay)
            try:
                res = self.broker.place_order(order)
                self._record_fill(res)
                return res
            except TransientBrokerError:
                continue
            except PermanentBrokerError as e:
                return {"status": "failed", "reason": str(e), "retryable": False}
        return {"status": "failed", "reason": "transient retries exhausted", "retryable": True}

    def _record_fill(self, res: dict) -> None:
        """Persist or buffer a broker result that came back as a fill."""
        if res.get("status") != "filled":
            return
        fill = {"order_id": res.get("id", ""), "symbol": res.get("symbol", ""),
                "side": res.get("side", ""), "qty": res.get("qty", 0.0),
                "price": res.get("fill_price", res.get("price", 0.0)),
                "ts": res.get("ts", "")}
        if self.store is not None:
            self.store.append_order(res)
            self.store.append_fill(fill)
        else:
            self._fills.append(fill)

    def on_bar_close(self, symbol: str, bar: dict) -> list[dict]:
        """Let the broker settle queued orders at this bar's open; record fills."""
        fills = self.broker.on_bar_close(symbol, bar)
        for f in fills:
            self._record_fill(f)
        return fills

    def flatten_all(self, reason: str) -> None:
        for pos in self.get_positions():
            if self.flatten_symbols is not None \
                    and pos["symbol"] not in self.flatten_symbols:
                continue  # per-market flatten scope
            price = self._last_prices.get(pos["symbol"], 0.0)
            if price <= 0:
                continue  # cannot size a flatten without a price
            side = "sell" if pos["qty"] > 0 else "buy"
            res = self.broker.place_order({"symbol": pos["symbol"], "side": side,
                                           "qty": abs(pos["qty"]), "price": price,
                                           "market": True})
            self._record_fill(res)

    def maybe_flatten_before_close(self, now: datetime, symbol: str | None = None) -> None:
        if symbol is not None and self.flatten_symbols is not None \
                and symbol not in self.flatten_symbols:
            return  # 24/7 markets do not flatten at the NSE close
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