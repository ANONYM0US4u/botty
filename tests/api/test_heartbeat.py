import time

from fastapi.testclient import TestClient

from engine.api.heartbeat import (CLOSE_GRACE_S, HEARTBEAT_TIMEOUT_S,
                                   Heartbeat)
from engine.api.main import create_app


def _hb():
    return Heartbeat()


def test_bare_backend_never_shuts_down():
    hb = _hb()
    for t in range(0, 600, 30):
        assert hb.decide(t) == "alive"


def test_heartbeat_arms_and_keeps_alive():
    hb = _hb()
    assert hb.decide(0) == "alive"
    hb.touch(10)
    assert hb.decide(20) == "alive"
    assert hb.decide(10 + HEARTBEAT_TIMEOUT_S - 1) == "alive"
    assert hb.decide(10 + HEARTBEAT_TIMEOUT_S + 1) == "shutdown"


def test_closing_beacon_shuts_down_after_grace():
    hb = _hb()
    hb.touch(0)
    hb.touch(100, closing=True)
    assert hb.decide(100) == "closing"
    assert hb.decide(100 + CLOSE_GRACE_S - 1) == "closing"
    assert hb.decide(100 + CLOSE_GRACE_S + 1) == "shutdown"


def test_heartbeat_cancels_pending_close():
    hb = _hb()
    hb.touch(0)
    hb.touch(100, closing=True)
    assert hb.decide(105) == "closing"
    hb.touch(110)  # reload: page is back
    assert hb.decide(110 + CLOSE_GRACE_S + 1) == "alive"


def test_closing_never_arms_unseen_backend():
    hb = _hb()
    hb.touch(0, closing=True)
    assert hb.decide(9999) == "alive"


def test_route_arms_and_closes():
    from engine.data.store import DataStore
    import tempfile
    from pathlib import Path
    tmp = Path(tempfile.mkdtemp())
    store = DataStore(tmp / "t.db", tmp / "pq")
    store.init_schema()
    hb = _hb()
    app = create_app(store, None, {"risk": {}}, heartbeat=hb)
    c = TestClient(app)
    assert c.post("/api/heartbeat").status_code == 200
    assert hb.last_seen is not None
    assert c.post("/api/heartbeat?closing=1").status_code == 200
    assert hb.closing_at is not None
    c.post("/api/heartbeat")
    assert hb.closing_at is None


def test_route_without_heartbeat_is_noop():
    import tempfile
    from pathlib import Path
    from engine.data.store import DataStore
    tmp = Path(tempfile.mkdtemp())
    store = DataStore(tmp / "t.db", tmp / "pq")
    store.init_schema()
    app = create_app(store, None, {"risk": {}})
    c = TestClient(app)
    assert c.post("/api/heartbeat").status_code == 200
    assert c.post("/api/heartbeat?closing=1").status_code == 200