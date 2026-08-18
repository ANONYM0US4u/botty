from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from engine.api.metrics_emitter import MetricsEmitter

emitter = MetricsEmitter()

_PROMOTION_ORDINAL = {"paper": 0, "staged": 1, "live": 2}


class KillSwitchBody(BaseModel):
    active: bool


class PromotionBody(BaseModel):
    action: str


class TheaterStartBody(BaseModel):
    symbol: str


class Promotion:
    """paper -> staged -> live; approve never skips staging."""

    def __init__(self):
        self.state = "paper"

    def transition(self, action: str) -> str:
        cur = self.state
        if action == "stage" and cur == "paper":
            self.state = "staged"
        elif action == "approve" and cur == "staged":
            self.state = "live"
        elif action == "reject" and cur == "staged":
            self.state = "paper"
        elif action == "revert" and cur in ("staged", "live"):
            self.state = "paper"
        else:
            raise ValueError(f"invalid promotion action '{action}' from '{cur}'")
        return self.state


def create_app(store, risk, cfg: dict, theater=None) -> FastAPI:
    app = FastAPI(title="Trading Bot Engine")
    promotion = Promotion()

    @app.get("/api/equity")
    def equity():
        return store.get_equity()

    @app.get("/api/trades")
    def trades():
        return store.get_trades()

    @app.get("/api/checkpoints")
    def checkpoints():
        return store.get_checkpoints()

    @app.get("/api/metrics/{name}")
    def metrics(name: str, since: str | None = None):
        return store.get_metrics(name, since)

    @app.get("/api/positions")
    def positions():
        return risk.get_positions()

    @app.get("/api/status")
    def status():
        eq = store.get_equity()
        return {"killed": risk.is_killed(),
                "equity": eq[-1]["equity"] if eq else 0.0,
                "day_pnl": 0.0,
                "promotion_state": promotion.state}

    @app.post("/api/killswitch")
    def killswitch(body: KillSwitchBody):
        risk.set_kill_switch(body.active)
        return {"killed": body.active}

    @app.get("/api/promotion")
    def get_promotion():
        return {"state": promotion.state}

    @app.post("/api/promotion")
    def post_promotion(body: PromotionBody):
        try:
            new_state = promotion.transition(body.action)
        except ValueError as e:
            return {"error": str(e), "state": promotion.state}
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        store.append_decision({"ts": ts, "symbol": "promotion",
                               "action": f"{body.action} -> {new_state}",
                               "probs": "[]", "features": "[]", "attribution": "[]"})
        store.append_metric("promotion", _PROMOTION_ORDINAL[new_state], ts)
        return {"state": new_state}

    @app.get("/api/decisions")
    def decisions(symbol: str | None = None, limit: int = 100):
        return store.get_decisions(symbol, limit)

    def _require_theater():
        return None if theater is None else theater

    @app.get("/api/theater/state")
    def theater_state():
        if _require_theater() is None:
            return JSONResponse({"error": "theater not configured"}, status_code=503)
        return theater.state()

    @app.post("/api/theater/start")
    def theater_start(body: TheaterStartBody):
        if _require_theater() is None:
            return JSONResponse({"error": "theater not configured"}, status_code=503)
        try:
            out = theater.start(body.symbol)
        except RuntimeError:
            return JSONResponse({"error": "already running"}, status_code=409)
        if "error" in out:
            return JSONResponse(out, status_code=400)
        return out

    @app.post("/api/theater/stop")
    def theater_stop():
        if _require_theater() is None:
            return JSONResponse({"error": "theater not configured"}, status_code=503)
        return theater.stop()

    @app.post("/api/theater/reset")
    def theater_reset():
        if _require_theater() is None:
            return JSONResponse({"error": "theater not configured"}, status_code=503)
        return theater.reset()

    @app.get("/api/theater/leaderboard")
    def theater_leaderboard():
        if _require_theater() is None:
            return JSONResponse({"error": "theater not configured"}, status_code=503)
        return theater.leaderboard()

    @app.websocket("/ws/metrics")
    async def ws_metrics(ws: WebSocket):
        await ws.accept()
        emitter.register(ws)
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            emitter.unregister(ws)

    return app