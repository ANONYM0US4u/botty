import pytest
from fastapi.testclient import TestClient
from engine.api.main import create_app
from engine.data.store import DataStore
from engine.live.risk import RiskGateway
from engine.brokers.simulator import SimulatorAdapter


@pytest.fixture
def client(tmp_path):
    store = DataStore(tmp_path / "t.db", tmp_path / "pq")
    store.init_schema()
    store.append_equity("2026-01-02 09:25:00", 100_000.0)
    store.append_metric("reward", 1.5, "2026-01-02 09:25:00")
    risk = RiskGateway(SimulatorAdapter(), {"daily_loss_limit_pct": -3.0,
                       "max_position_pct": 30.0, "max_total_exposure_pct": 90.0,
                       "stale_data_seconds": 120, "flatten_at": "15:15"})
    app = create_app(store, risk, {"brokers": {"active": "simulator"}})
    return TestClient(app)


def test_equity_endpoint(client):
    r = client.get("/api/equity")
    assert r.status_code == 200 and len(r.json()) == 1


def test_metrics_endpoint(client):
    r = client.get("/api/metrics/reward")
    assert r.status_code == 200 and r.json()[0]["value"] == 1.5


def test_killswitch_toggle(client):
    r = client.post("/api/killswitch", json={"active": True})
    assert r.status_code == 200
    assert client.get("/api/status").json()["killed"] is True
    client.post("/api/killswitch", json={"active": False})


def test_promotion_requires_two_staged_steps(client):
    assert client.get("/api/promotion").json()["state"] == "paper"
    r1 = client.post("/api/promotion", json={"action": "stage"})
    assert r1.status_code == 200 and r1.json()["state"] == "staged"
    r2 = client.post("/api/promotion", json={"action": "approve"})
    assert r2.status_code == 200 and r2.json()["state"] == "live"
    r3 = client.post("/api/promotion", json={"action": "revert"})
    assert r3.json()["state"] == "paper"