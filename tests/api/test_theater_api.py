import pytest
from fastapi.testclient import TestClient
from engine.api.main import create_app
from engine.data.store import DataStore
from engine.live.risk import RiskGateway
from engine.brokers.simulator import SimulatorAdapter

_RISK = {"daily_loss_limit_pct": -3.0, "max_position_pct": 30.0,
         "max_total_exposure_pct": 90.0, "stale_data_seconds": 120,
         "flatten_at": "15:15"}


class FakeTheater:
    def state(self): return {"status": "idle", "symbol": None}
    def start(self, symbol):
        if symbol == "BUSY": raise RuntimeError("already running")
        if symbol == "BAD": return {"error": "fetch failed: no data"}
        return {"status": "running"}
    def stop(self): return {"status": "stopped"}
    def reset(self): return {"status": "idle"}
    def leaderboard(self): return [{"path": "a.zip", "sharpe": 1.2}]


@pytest.fixture
def client(tmp_path):
    store = DataStore(tmp_path / "t.db", tmp_path / "pq")
    store.init_schema()
    risk = RiskGateway(SimulatorAdapter(), _RISK)
    return TestClient(create_app(store, risk, {}, theater=FakeTheater()))


@pytest.fixture
def bare_client(tmp_path):
    store = DataStore(tmp_path / "t.db", tmp_path / "pq")
    store.init_schema()
    risk = RiskGateway(SimulatorAdapter(), _RISK)
    return TestClient(create_app(store, risk, {}))


def test_theater_routes_503_without_theater(bare_client):
    assert bare_client.get("/api/theater/state").status_code == 503
    assert bare_client.post("/api/theater/start",
                            json={"symbol": "RELIANCE.NS"}).status_code == 503
    assert bare_client.post("/api/theater/stop").status_code == 503
    assert bare_client.post("/api/theater/reset").status_code == 503
    assert bare_client.get("/api/theater/leaderboard").status_code == 503


def test_theater_state_and_leaderboard(client):
    assert client.get("/api/theater/state").json()["status"] == "idle"
    lb = client.get("/api/theater/leaderboard").json()
    assert lb[0]["sharpe"] == 1.2


def test_theater_start_ok_and_conflict(client):
    assert client.post("/api/theater/start",
                       json={"symbol": "RELIANCE.NS"}).json()["status"] == "running"
    r = client.post("/api/theater/start", json={"symbol": "BUSY"})
    assert r.status_code == 409
    r2 = client.post("/api/theater/start", json={"symbol": "BAD"})
    assert r2.status_code == 400


def test_theater_stop_reset(client):
    assert client.post("/api/theater/stop").json()["status"] == "stopped"
    assert client.post("/api/theater/reset").json()["status"] == "idle"


def test_decisions_endpoint(client, tmp_path):
    store = DataStore(tmp_path / "t.db", tmp_path / "pq")
    store.init_schema()
    store.append_decision({"ts": "t1", "symbol": "RELIANCE.NS", "action": "long",
                           "probs": "[0.1,0.8,0.1]", "features": "[]",
                           "attribution": "[]"})
    c = TestClient(create_app(store, RiskGateway(SimulatorAdapter(), _RISK), {}))
    rows = c.get("/api/decisions?symbol=RELIANCE.NS&limit=5").json()
    assert len(rows) == 1 and rows[0]["action"] == "long"