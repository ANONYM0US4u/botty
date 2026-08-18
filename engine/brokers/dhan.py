from engine.brokers.base import BrokerAdapter
from dhanhq import DhanContext
from dhanhq.dhanhq import dhanhq

_SANDBOX_BASE_URL = "https://sandbox.dhan.co/v2"


class DhanAdapter(BrokerAdapter):
    def __init__(self, client_id: str, access_token: str, is_sandbox: bool = True):
        ctx = DhanContext(client_id, access_token)
        if is_sandbox:
            ctx.dhan_http.base_url = _SANDBOX_BASE_URL
        self.client = dhanhq(ctx)

    def place_order(self, order: dict) -> dict:
        side = "BUY" if order["side"] == "buy" else "SELL"
        resp = self.client.place_order(
            security_id=order.get("security_id"),
            exchange_segment=order.get("exchange_segment", "NSE_EQ"),
            transaction_type=side,
            quantity=int(order.get("qty")),
            order_type="LIMIT",
            price=order.get("price"),
            product_type="INTRADAY",
            validity="DAY",
        )
        return {"status": "ok" if resp.get("status") == "success" else "error",
                "id": resp.get("data", {}).get("orderId"), "raw": resp}

    def cancel_order(self, order_id: str) -> bool:
        resp = self.client.cancel_order(order_id)
        return resp.get("status") == "success"

    def get_positions(self) -> list[dict]:
        resp = self.client.get_positions()
        data = resp.get("data", []) or []
        return [{"symbol": p.get("tradingSymbol"), "qty": p.get("netQty", 0)} for p in data]

    def get_orders(self) -> list[dict]:
        resp = self.client.get_order_list()
        return resp.get("data", []) or []

    def get_balance(self) -> float:
        resp = self.client.get_fund_limits()
        data = resp.get("data", {}) or {}
        return float(data.get("availabelBalance", data.get("availableBalance", 0.0)))