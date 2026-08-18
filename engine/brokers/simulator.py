import uuid
from engine.brokers.base import BrokerAdapter


class SimulatorAdapter(BrokerAdapter):
    def __init__(self, initial_cash: float = 100_000.0, slippage_bps: float = 2.0,
                 latency_bars: int = 1):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.slippage_bps = slippage_bps
        self.latency_bars = latency_bars
        self.orders: list[dict] = []
        self.positions: dict[str, float] = {}
        self.last_fill_price: float | None = None

    def place_order(self, order: dict) -> dict:
        o = {"id": str(uuid.uuid4()), "status": "open", **order}
        if o.get("market"):
            self._fill_now(o)
        else:
            self.orders.append(o)
        return o

    def _fill_now(self, o: dict) -> None:
        fill_price = float(o.get("price", 0.0) or 0.0)
        qty = float(o["qty"])
        if o["side"] == "buy":
            self.cash -= fill_price * qty
            self.positions[o["symbol"]] = self.positions.get(o["symbol"], 0.0) + qty
        else:
            self.cash += fill_price * qty
            self.positions[o["symbol"]] = self.positions.get(o["symbol"], 0.0) - qty
        o["status"] = "filled"
        o["fill_price"] = fill_price
        self.last_fill_price = fill_price

    def cancel_order(self, order_id: str) -> bool:
        for o in self.orders:
            if o["id"] == order_id and o["status"] in ("open", "pending"):
                o["status"] = "cancelled"
                self.orders = [x for x in self.orders if x["id"] != order_id]
                return True
        return False

    def on_bar_close(self, symbol: str, bar: dict) -> list[dict]:
        fills = []
        open_price = float(bar["open"])  # next-bar-open rule: decision bar close never fills
        for o in self.orders:
            if o["symbol"] == symbol and o["status"] == "pending":
                slip = open_price * (1.0 + self.slippage_bps / 10_000.0) if o["side"] == "buy" \
                    else open_price * (1.0 - self.slippage_bps / 10_000.0)
                qty = float(o["qty"])
                if o["side"] == "buy":
                    self.cash -= slip * qty
                    self.positions[symbol] = self.positions.get(symbol, 0.0) + qty
                else:
                    self.cash += slip * qty
                    self.positions[symbol] = self.positions.get(symbol, 0.0) - qty
                o["status"] = "filled"
                o["fill_price"] = slip
                self.last_fill_price = slip
                fills.append({**o, "ts": bar.get("time", "")})
        self.orders = [x for x in self.orders if x["status"] != "filled"]
        for o in self.orders:
            if o["symbol"] == symbol and o["status"] == "open":
                o["status"] = "pending"
        return fills

    def get_positions(self) -> list[dict]:
        return [{"symbol": s, "qty": q} for s, q in self.positions.items() if q != 0]

    def get_orders(self) -> list[dict]:
        return [o for o in self.orders if o["status"] == "open"]

    def get_balance(self) -> float:
        return self.cash