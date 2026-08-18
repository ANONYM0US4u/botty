from pathlib import Path
import tempfile
import polars as pl
from engine.data.store import DataStore
from engine.data.indicators import add_indicators
from engine.env.trading_env import TradingEnv
from engine.agents.ppo import train_ppo, evaluate_ppo, load_policy
from engine.live.risk import RiskGateway
from engine.live.scheduler import Scheduler
from engine.brokers.simulator import SimulatorAdapter
from engine.eval.metrics import compute_eval_report


def synthetic_bars(n=800, seed=1):
    import numpy as np
    rng = np.random.default_rng(seed)
    px = 100.0 + np.cumsum(rng.normal(0.05, 0.4, n))
    rows = [{"time": f"2026-01-02 09:{15 + i:02d}:00",
             "open": px[i], "high": px[i] + 1.0, "low": px[i] - 1.0,
             "close": px[i], "volume": 1000.0 + i} for i in range(n)]
    return add_indicators(pl.DataFrame(rows))


def main():
    tmp = Path(tempfile.mkdtemp())
    bars = synthetic_bars()
    store = DataStore(tmp / "t.db", tmp / "pq")
    store.init_schema()
    store.save_bars("RELIANCE.NS", bars, 5)
    env = TradingEnv("RELIANCE.NS", bars, seed=42)
    path = train_ppo(env, 2_000, tmp / "ck", seed=42, store=store)
    model = load_policy(path)
    rep = evaluate_ppo(model, env, episodes=3, seed=42)
    risk = RiskGateway(SimulatorAdapter(initial_cash=100_000.0),
                       {"daily_loss_limit_pct": -3.0, "max_position_pct": 30.0,
                        "max_total_exposure_pct": 90.0, "stale_data_seconds": 120,
                        "flatten_at": "15:15"})
    sim = risk.broker
    sched = Scheduler(risk, store, model, {"symbols": ["RELIANCE.NS"]})
    for row in bars.tail(20).iter_rows(named=True):
        sched.on_bar_close("RELIANCE.NS", row)
    eq = [p["equity"] for p in store.get_equity()]
    report = compute_eval_report(eq, store.get_trades())
    print("SMOKE OK", report)


if __name__ == "__main__":
    main()