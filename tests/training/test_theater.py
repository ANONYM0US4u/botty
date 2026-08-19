import time
import polars as pl
import pytest
from engine.training.theater import TrainingTheater, _VALIDATION_WINDOW
from engine.data.store import DataStore
from engine.data.indicators import add_indicators
from engine.brokers.simulator import SimulatorAdapter
from engine.live.risk import RiskGateway


def _bars(n=400):
    import numpy as np
    rng = np.random.default_rng(1)
    px = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    rows = [{"time": f"2026-01-02 09:{15 + i:02d}:00", "open": px[i],
             "high": px[i] + 1.0, "low": px[i] - 1.0, "close": px[i],
             "volume": 1000.0} for i in range(n)]
    return add_indicators(pl.DataFrame(rows))


def _theater(tmp_path, fetch=None):
    store = DataStore(tmp_path / "t.db", tmp_path / "pq")
    store.init_schema()
    cfg = {"instruments": {"stocks": ["RELIANCE.NS"], "crypto": ["BTCUSDT"]},
           "training": {"seed": 42, "total_timesteps": 10_000},
           "storage": {"checkpoint_dir": str(tmp_path / "ck")},
           "risk": {"max_position_pct": 30.0, "daily_loss_limit_pct": -3.0,
                    "max_total_exposure_pct": 90.0, "stale_data_seconds": 120,
                    "flatten_at": "15:15"}}
    th = TrainingTheater(store, None, cfg, fetch or (lambda s: _bars()))
    return th, store


def _fake_ppo(sleep_sec=1.0):
    class FakePPO:
        def __init__(self, *a, **kw):
            self.learn_calls = 0
        def learn(self, **kw):
            self.learn_calls += 1
            time.sleep(sleep_sec)
            if kw.get("callback"):
                kw["callback"].n_calls = 100
                kw["callback"]._on_step()
        def save(self, path):
            open(path, "w").write("fake")
    return FakePPO


def test_start_stop_flow(tmp_path, monkeypatch):
    monkeypatch.setattr("stable_baselines3.PPO", _fake_ppo())
    th, store = _theater(tmp_path)
    out = th.start("RELIANCE.NS")
    assert out["status"] == "running"
    assert th.state()["status"] == "running"
    time.sleep(0.2)
    th.stop()
    assert th.wait_idle(10)
    assert th.state()["status"] == "stopped"


def test_double_start_conflicts(tmp_path, monkeypatch):
    monkeypatch.setattr("stable_baselines3.PPO", _fake_ppo())
    th, _ = _theater(tmp_path)
    th.start("RELIANCE.NS")
    with pytest.raises(RuntimeError):
        th.start("BTCUSDT")
    th.stop(); th.wait_idle(15)


def test_fetch_failure_returns_error(tmp_path, monkeypatch):
    def bad_fetch(symbol): raise ValueError("no data")
    th, _ = _theater(tmp_path, fetch=bad_fetch)
    out = th.start("RELIANCE.NS")
    assert out["error"] and th.state()["status"] == "idle"


def test_reset_clears_only_current_run(tmp_path, monkeypatch):
    monkeypatch.setattr("stable_baselines3.PPO", _fake_ppo())
    th, _ = _theater(tmp_path)
    (tmp_path / "ck" / "theater" / "crypto" / "keep").mkdir(parents=True)
    (tmp_path / "ck" / "theater" / "crypto" / "keep" / "x.zip").write_text("x")
    th.start("RELIANCE.NS")
    time.sleep(0.2)
    th.stop(); th.wait_idle(10)
    th.reset()
    assert th.state()["status"] == "idle"
    assert (tmp_path / "ck" / "theater" / "crypto" / "keep" / "x.zip").exists()


def test_leaderboard_ranks_by_sharpe(tmp_path, monkeypatch):
    import engine.training.theater as mod
    monkeypatch.setattr(mod, "compute_eval_report",
                        lambda eq, trades: {"sharpe": 1.0, "win_rate": 0.5})
    monkeypatch.setattr(mod, "load_policy",
                        lambda *a, **k: type("P", (), {"predict": lambda self,
                        obs, **kw: (__import__("numpy").zeros(()), None)})())
    th, store = _theater(tmp_path)
    store.save_bars("BTCUSDT", _bars(400), 5)
    (th.ck_root / "crypto" / "runB").mkdir(parents=True)
    (th.ck_root / "crypto" / "runB" / "ppo_b.zip").write_text("b")
    with th._lock:
        th._run_id = "runB"
        th._market = "crypto"
        th._symbol = "BTCUSDT"
        th._lb_ready = False
    rows = th.leaderboard()
    assert rows == []
    for _ in range(100):
        rows = th.leaderboard()
        if rows:
            break
        time.sleep(0.05)
    assert len(rows) == 1
    assert rows[0]["sharpe"] == 1.0


def test_start_creates_market_scoped_run_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("stable_baselines3.PPO", _fake_ppo())
    th, _ = _theater(tmp_path)
    th.start("RELIANCE.NS")
    time.sleep(0.2)
    th.stop(); th.wait_idle(10)
    dirs = list((th.ck_root / "nse").iterdir())
    assert len(dirs) == 1 and dirs[0].name.startswith("RELIANCE.NS-")


def test_start_rejects_unknown_symbol(tmp_path):
    th, _ = _theater(tmp_path)
    out = th.start("MISSING.NS")
    assert out["error"] and th.state()["status"] == "idle"


def test_prunes_old_runs_per_market(tmp_path):
    th, _ = _theater(tmp_path)
    for i in range(4):
        (th.ck_root / "nse" / f"TCS.NS-2026010{i}-000000").mkdir(parents=True)
    (th.ck_root / "crypto" / "BTCUSDT-20260105-000000").mkdir(parents=True)
    th._prune("nse")
    nse = sorted(p.name for p in (th.ck_root / "nse").iterdir())
    assert nse == [f"TCS.NS-2026010{i}-000000" for i in (1, 2, 3)]
    assert (th.ck_root / "crypto" / "BTCUSDT-20260105-000000").exists()