import ccxt
from engine.brokers.base import BrokerAdapter


class BybitAdapter(BrokerAdapter):
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.ex = ccxt.bybit({
            "apiKey": api_key, "secret": api_secret,
            "enableRateLimit": True, "sandbox": testnet,
            "options": {"defaultType": "swap"},
        })

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