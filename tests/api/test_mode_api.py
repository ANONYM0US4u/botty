import pytest
from fastapi.testclient import TestClient

from engine.api.main import create_app
from engine.brokers.simulator import SimulatorAdapter
from engine.data.store import DataStore
from engine.live.risk import RiskGateway
from engine.trading.mode import BotMode

_RISK = {"daily_loss_limit_pct": -3.0, "max_position_pct": 30.0,
         "max_total_exposure_pct": 90.0, "stale_data_seconds": 120,
         "flatten_at": "15:15"}


class FakeMode:
    def __init__(self):
        self._mode = "idle"
        self._market = "crypto"

    def state(self):
        return {"market": self._market, "mode": self._mode,
                "markets": ["crypto", "nse"], "switching": False,
                "trade": {"running": False}, "train": {"status": "idle"}}

    def can_train(self):
        return self._mode != "trade"

    def start(self, symbol):
        return {"status": "running"}

    def stop(self):
        return {"status": "stopped"}

    def wait_idle(self, timeout):
        return True

    def set_market(self, market):
        if market not in ("crypto", "nse"):
            return {"error": f"unknown market {market}"}
        self._market = market
        return self.state()

    def set_mode(self, mode):
        if mode not in ("idle", "train", "trade"):
            return {"error": f"unknown mode {mode}"}
        self._mode = mode
        return self.state()


@pytest.fixture
def client(tmp_path):
    store = DataStore(tmp_path / "t.db", tmp_path / "pq")
    store.init_schema()
    risk = RiskGateway(SimulatorAdapter(), _RISK)
    return TestClient(create_app(store, risk, {},
                                 theater=FakeMode(), mode=FakeMode()))


def test_mode_routes_503_without_mode(tmp_path):
    store = DataStore(tmp_path / "t.db", tmp_path / "pq")
    store.init_schema()
    risk = RiskGateway(SimulatorAdapter(), _RISK)
    c = TestClient(create_app(store, risk, {}))
    assert c.get("/api/mode").status_code == 503
    assert c.post("/api/mode", json={"mode": "trade"}).status_code == 503


def test_mode_get_and_post(client):
    st = client.get("/api/mode").json()
    assert st["market"] == "crypto" and st["mode"] == "idle"
    st = client.post("/api/mode", json={"mode": "trade"}).json()
    assert st["mode"] == "trade"
    st = client.post("/api/mode", json={"market": "nse"}).json()
    assert st["market"] == "nse"


def test_mode_validation(client):
    r = client.post("/api/mode", json={"mode": "nope"})
    assert r.status_code == 400
    r = client.post("/api/mode", json={"market": "eurusd"})
    assert r.status_code == 400
    r = client.post("/api/mode", json={})
    assert r.status_code == 400


def test_theater_start_blocked_in_trade_mode(client):
    client.post("/api/mode", json={"mode": "trade"})
    r = client.post("/api/theater/start", json={"symbol": "BTCUSDT"})
    assert r.status_code == 409
    client.post("/api/mode", json={"mode": "train"})
    assert client.post("/api/theater/start",
                       json={"symbol": "BTCUSDT"}).status_code == 200