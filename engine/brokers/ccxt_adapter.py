import time

import ccxt
from engine.brokers.base import BrokerAdapter


class CcxtAdapter(BrokerAdapter):
    def __init__(self, exchange_id: str, api_key: str, api_secret: str,
                 testnet: bool = True):
        self.exchange_id = exchange_id
        cls = getattr(ccxt, exchange_id, None)
        if cls is None:
            raise ValueError(f"exchange {exchange_id} unavailable")
        self.ex = cls({
            "apiKey": api_key, "secret": api_secret,
            "enableRateLimit": True, "sandbox": testnet,
            "options": {"defaultType": "swap", "timeDifference": self._measure_skew()},
        })

    def _measure_skew(self) -> int:
        try:
            return self._server_time_ms() - int(time.time() * 1000)
        except Exception:
            return 0

    def _server_time_ms(self) -> int:
        cls = getattr(ccxt, self.exchange_id, ccxt.gate)
        ex = cls({"enableRateLimit": False, "sandbox": True})
        return int(ex.fetch_time())

    def place_order(self, order: dict) -> dict:
        side = "buy" if order["side"] == "buy" else "sell"
        try:
            resp = self.ex.create_order(order["symbol"], "limit", side,
                                        float(order["qty"]), float(order["price"]))
            return {"status": "ok", "id": resp.get("id"), "raw": resp}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def cancel_order(self, order_id: str) -> bool:
        return bool(self.ex.cancel_order(order_id))

    def get_positions(self) -> list[dict]:
        try:
            return self.ex.fetch_positions()
        except Exception:
            return []

    def get_orders(self) -> list[dict]:
        return self.ex.fetch_open_orders()

    def get_balance(self) -> float:
        bal = self.ex.fetch_balance()
        return float(bal.get("USDT", {}).get("free", 0.0))