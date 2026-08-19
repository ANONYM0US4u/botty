import time

import polars as pl
import pytest

from engine.data.indicators import add_indicators
from engine.data.store import DataStore
from engine.live.risk import RiskGateway
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
            self.n = 0

        def predict(self, obs, deterministic=True):
            import numpy as np
            # rotate flat/long/short so the loop exercises every branch
            a = self.n % 3
            self.n += 1
            return (np.array(a), None)

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
        if len(store.get_decisions(symbol="BTCUSDT", limit=100)) >= 3:
            break
        time.sleep(0.05)
    assert m._last_bar_seen.get("BTCUSDT"), "trade loop never processed a bar"
    dec = store.get_decisions(symbol="BTCUSDT", limit=100)
    assert dec, "no decisions recorded"
    actions = {d["action"] for d in dec}
    assert len(actions) >= 2, "decisions must vary as the observation window advances (H2)"
    assert any(d["action"] != "flat" for d in dec), "obs-aware policy never went long"
    m.set_mode("idle")
    assert not m.state()["trade"]["running"]


def test_kill_switch_blocks_trades_through_shared_risk(tmp_path, monkeypatch):
    # H1: the API and the trade loop share one RiskGateway; arming the
    # kill switch must stop the loop from placing orders.
    m, store, cfg = _mode(tmp_path, monkeypatch)
    _seed_policy(tmp_path)
    m._risk.set_kill_switch(True)
    m.set_market("crypto")
    m.set_mode("trade")
    for _ in range(200):
        if len(store.get_decisions(symbol="BTCUSDT", limit=200)) >= 3:
            break
        time.sleep(0.05)
    assert m._last_bar_seen.get("BTCUSDT")
    dec = store.get_decisions(symbol="BTCUSDT", limit=200)
    assert len(dec) >= 3
    assert dec and all(d["action"] == "flat" for d in dec)
    assert store.get_trades() == []  # kill switch: no fills, no qty
    m.set_mode("idle")


def test_start_theater_guards_mode_and_market(tmp_path, monkeypatch):
    m, _, _ = _mode(tmp_path, monkeypatch)
    monkeypatch.setattr(m.theater, "start", lambda s: {"status": "running"})
    m.set_mode("trade")
    out = m.start_theater("BTCUSDT")
    assert out.get("code") == "mode" and "error" in out
    m.set_mode("idle")
    m.set_market("nse")
    out = m.start_theater("BTCUSDT")
    assert out.get("code") == "market" and "error" in out
    out = m.start_theater("MISSING.NS")
    assert out.get("code") == "symbol"
    m.set_market("crypto")
    out = m.start_theater("BTCUSDT")
    assert "error" not in out


def test_set_mode_trade_fails_when_training_wont_stop(tmp_path, monkeypatch):
    class StuckTheater:
        def state(self):
            return {"status": "running"}
        def stop(self):
            return {"status": "stopping"}
        def wait_idle(self, timeout):
            return False
        @property
        def ck_root(self):
            return tmp_path / "ck"
    from engine.brokers.simulator import SimulatorAdapter
    from engine.data.store import DataStore
    store = DataStore(tmp_path / "t.db", tmp_path / "pq")
    store.init_schema()
    m = BotMode(store, None, _cfg(tmp_path), StuckTheater(),
                lambda s: _bars(),
                risk=RiskGateway(SimulatorAdapter(), _cfg(tmp_path)["risk"]))
    out = m.set_mode("trade")
    assert "error" in out and "did not stop" in out["error"]
    assert m.state()["mode"] == "idle"


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