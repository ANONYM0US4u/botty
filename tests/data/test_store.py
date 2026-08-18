import polars as pl
import pytest
from engine.data.store import DataStore


@pytest.fixture
def store(tmp_path):
    s = DataStore(tmp_path / "t.db", tmp_path / "parquet")
    s.init_schema()
    return s


def test_bars_roundtrip(store):
    df = pl.DataFrame({
        "time": ["2026-01-02 09:15:00", "2026-01-02 09:20:00"],
        "open": [100.0, 101.0], "high": [101.5, 102.0],
        "low": [99.5, 100.5], "close": [101.0, 101.5], "volume": [1000, 1200],
    })
    store.save_bars("RELIANCE.NS", df, 5)
    out = store.get_bars("RELIANCE.NS")
    assert out.height == 2
    assert out["close"].to_list() == [101.0, 101.5]


def test_append_and_read_ledger(store):
    store.append_order({"id": "o1", "symbol": "RELIANCE.NS", "side": "buy",
                        "qty": 10, "price": 100.0, "status": "pending"})
    store.append_fill({"order_id": "o1", "symbol": "RELIANCE.NS", "side": "buy",
                       "qty": 10, "price": 100.0, "ts": "2026-01-02 09:21:00"})
    store.append_equity("2026-01-02 09:25:00", 100_000.0)
    trades = store.get_trades()
    assert len(trades) == 1 and trades[0]["order_id"] == "o1"
    assert len(store.get_equity()) == 1


def test_checkpoints_and_metrics(store):
    store.append_checkpoint({"path": "checkpoints/ppo_42.zip", "reward": 1.5,
                             "sharpe": 1.2, "ts": 42,
                             "run_id": "r1", "model_id": "m1",
                             "git_commit": "abc", "config_hash": "c1"})
    store.append_metric("reward", 1.5, "2026-01-02 09:25:00")
    store.append_metric("reward", 2.0, "2026-01-02 09:30:00")
    ck = store.get_checkpoints()
    assert len(ck) == 1
    assert ck[0]["run_id"] == "r1" and ck[0]["git_commit"] == "abc"
    assert len(store.get_metrics("reward")) == 2
    assert len(store.get_metrics("reward", since="2026-01-02 09:28:00")) == 1


def test_decisions_roundtrip(store):
    store.append_decision({"ts": "2026-01-02 09:25:00", "symbol": "RELIANCE.NS",
                           "action": "long", "probs": "[0.1,0.8,0.1]",
                           "features": "[]", "attribution": "[]"})
    with store._conn() as conn:
        rows = conn.execute("SELECT action FROM decisions").fetchall()
    assert rows == [("long",)]


def test_conn_busy_timeout(tmp_path):
    store = DataStore(tmp_path / "t.db", tmp_path / "pq")
    store.init_schema()
    conn = store._conn()
    row = conn.execute("PRAGMA busy_timeout").fetchone()
    assert row[0] >= 5000
    conn.close()


def test_get_decisions_filters_and_orders(tmp_path):
    store = DataStore(tmp_path / "t.db", tmp_path / "pq")
    store.init_schema()
    store.append_decision({"ts": "2026-01-02 09:30:00", "symbol": "RELIANCE.NS",
                           "action": "long", "probs": "[0.1,0.8,0.1]",
                           "features": "[]", "attribution": "[]"})
    store.append_decision({"ts": "2026-01-02 09:35:00", "symbol": "BTCUSDT",
                           "action": "flat", "probs": "[0.9,0.05,0.05]",
                           "features": "[]", "attribution": "[]"})
    rows = store.get_decisions(symbol="RELIANCE.NS", limit=10)
    assert len(rows) == 1 and rows[0]["action"] == "long"
    rows2 = store.get_decisions(limit=10)
    assert len(rows2) == 2 and rows2[0]["symbol"] == "BTCUSDT"