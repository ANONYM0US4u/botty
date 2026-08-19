import time

import polars as pl
import pytest

from engine.data.indicators import add_indicators
from engine.data.store import DataStore
from engine.trading.mode import BotMode


def _cfg(tmp_path):
    return {"instruments": {"stocks": ["RELIANCE.NS"],
                            "crypto": ["BTCUSDT"]},
            "training": {"seed": 42, "total_timesteps": 10_000,
                         "window_bars": 120},
            "storage": {"checkpoint_dir": str(tmp_path / "ck")},
            "brokers": {"slippage_bps": 2.0, "latency_bars": 1},
            "risk": {"max_position_pct": 30.0, "daily_loss_limit_pct": -3.0,
                     "max_total_exposure_pct": 90.0, "stale_data_seconds": 120,
                     "flatten_at": "15:15"}}


def _bars(n=400, start="2026-01-02 09:15:00"):
    import numpy as np
    rng = np.random.default_rng(7)
    px = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    rows = [{"time": f"2026-01-02 09:{15 + i:02d}:00", "open": px[i],
             "high": px[i] + 1.0, "low": px[i] - 1.0, "close": px[i],
             "volume": 1000.0} for i in range(n)]
    return add_indicators(pl.DataFrame(rows))


def _stub_policy():
    class P:
        def __init__(self, *a, **kw):
            pass

        def predict(self, obs, deterministic=True):
            import numpy as np
            return (np.array(0), None)

        @property
        def device(self):
            return "cpu"
    return P


def _mode(tmp_path, monkeypatch, fetch=None, policy=None):
    store = DataStore(tmp_path / "t.db", tmp_path / "pq")
    store.init_schema()
    cfg = _cfg(tmp_path)
    if policy is None:
        policy = _stub_policy()
    monkeypatch.setattr("engine.trading.mode.load_policy", lambda p: policy())
    from engine.training.theater import TrainingTheater
    theater = TrainingTheater(store, None, cfg, fetch or (lambda s: _bars()))
    m = BotMode(store, None, cfg, theater, fetch or (lambda s: _bars()))
    return m, store, cfg


def _seed_policy(tmp_path, symbol="BTCUSDT", market="crypto"):
    root = tmp_path / "ck" / "theater" / market
    run = root / f"{symbol}-20260101-000000"
    run.mkdir(parents=True)
    (run / "latest.zip").write_text("policy")
    (run / "ppo_1000.zip").write_text("policy")


def test_state_defaults(tmp_path, monkeypatch):
    m, _, _ = _mode(tmp_path, monkeypatch)
    st = m.state()
    assert st["market"] == "crypto" and st["mode"] == "idle"
    assert st["markets"] == ["crypto", "nse"]


def test_set_market_valid_and_invalid(tmp_path, monkeypatch):
    m, _, _ = _mode(tmp_path, monkeypatch)
    st = m.set_market("nse")
    assert st["market"] == "nse"
    assert "error" in m.set_market("eurusd")


def test_trade_mode_processes_bars_and_records_fills(tmp_path, monkeypatch):
    m, store, cfg = _mode(tmp_path, monkeypatch)
    _seed_policy(tmp_path)
    m.set_market("crypto")
    st = m.set_mode("trade")
    assert st["mode"] == "trade"
    assert st["trade"]["running"]
    for _ in range(200):
        if m._last_bar_seen.get("BTCUSDT"):
            break
        time.sleep(0.05)
    assert m._last_bar_seen.get("BTCUSDT"), "trade loop never processed a bar"
    dec = store.get_decisions(symbol="BTCUSDT", limit=1)
    assert dec, "no decisions recorded"
    m.set_mode("idle")
    assert not m.state()["trade"]["running"]


def test_trade_skips_symbols_without_policy(tmp_path, monkeypatch):
    m, store, cfg = _mode(tmp_path, monkeypatch)
    m.set_market("nse")  # no RELIANCE.NS policy seeded
    m.set_mode("trade")
    time.sleep(1.0)
    assert store.get_decisions(symbol="RELIANCE.NS", limit=1) == []
    m.set_mode("idle")


def test_train_mode_blocks_theater_until_switched(tmp_path, monkeypatch):
    m, _, _ = _mode(tmp_path, monkeypatch)
    assert m.can_train()
    m.set_mode("trade")
    assert not m.can_train()
    m.set_mode("train")
    assert m.can_train()


def test_market_switch_restarts_trade_for_new_market(tmp_path, monkeypatch):
    m, store, _ = _mode(tmp_path, monkeypatch)
    _seed_policy(tmp_path)
    m.set_market("crypto")
    m.set_mode("trade")
    for _ in range(200):
        if m._last_bar_seen.get("BTCUSDT"):
            break
        time.sleep(0.05)
    assert store.get_decisions(symbol="BTCUSDT", limit=1)
    _seed_policy(tmp_path, symbol="RELIANCE.NS", market="nse")
    m.set_market("nse")
    assert m.state()["market"] == "nse"
    assert m.state()["trade"]["running"]
    m.set_mode("idle")