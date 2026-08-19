import time
import threading
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import uvicorn

from engine.agents.ppo import load_policy, train_ppo
from engine.api.main import create_app
from engine.brokers.simulator import SimulatorAdapter
from engine.config import load_config
from engine.data.fetchers.ccxt_crypto import fetch_crypto_bars
from engine.data.fetchers.yfinance_nse import fetch_nse_minute_bars
from engine.data.indicators import add_indicators
from engine.data.store import DataStore
from engine.env.trading_env import TradingEnv
from engine.live.risk import RiskGateway
from engine.live.scheduler import Scheduler


def make_fetch_bars(cfg: dict):
    stocks = set(cfg["instruments"]["stocks"])
    crypto = set(cfg["instruments"]["crypto"])
    end = datetime.now().strftime("%Y-%m-%d")
    crypto_start = (datetime.now() - timedelta(days=33)).strftime("%Y-%m-%d")
    stock_start = (datetime.now() - timedelta(days=55)).strftime("%Y-%m-%d")

    def fetch_bars(symbol: str):
        if symbol in stocks:
            return add_indicators(
                fetch_nse_minute_bars(symbol, stock_start, end, "5m"))
        if symbol in crypto:
            ccxt_symbol = f"{symbol[:-4]}/{symbol[-4:]}:USDT"
            return add_indicators(
                fetch_crypto_bars(ccxt_symbol, crypto_start, end, "5m", "gate"))
        raise ValueError(f"symbol {symbol} not configured")

    return fetch_bars


def main() -> None:
    cfg = load_config()
    store = DataStore(cfg["storage"]["db_path"], cfg["storage"]["parquet_dir"])
    store.init_schema()
    broker = SimulatorAdapter(
        slippage_bps=cfg["brokers"].get("slippage_bps", 2.0),
        latency_bars=cfg["brokers"].get("latency_bars", 1))
    risk = RiskGateway(broker, cfg["risk"])
    from engine.api import main as api_main
    emitter = api_main.emitter

    symbols = list(cfg["instruments"]["crypto"]) + list(cfg["instruments"]["stocks"])
    fetch_bars = make_fetch_bars(cfg)
    ck_dir = cfg["storage"]["checkpoint_dir"]
    last_seen = {}

    for s in symbols:
        try:
            bars = fetch_bars(s)
            if bars is not None and bars.height > 0:
                store.save_bars(s, bars, 5)
                last_seen[s] = bars["time"][-1]
                print(f"seeded {s}: {bars.height} bars")
        except Exception as e:
            print(f"seed {s} failed: {e}")

    def train_loop():
        symbol = symbols[0]
        bars = store.get_bars(symbol)
        env = TradingEnv(symbol, bars,
                         window=cfg["training"].get("window_bars", 120),
                         seed=cfg["training"]["seed"])
        total = cfg["training"]["total_timesteps"]
        train_ppo(env, total, ck_dir, cfg["training"]["seed"],
                  save_every=50_000, store=store, cfg=cfg, emitter=emitter)

    def latest_policy():
        cks = sorted(Path(ck_dir).glob("ppo_*.zip"))
        theater = sorted(Path(ck_dir).glob("theater/*/latest.zip"))
        all_ck = sorted(cks + theater, key=lambda p: p.stat().st_mtime)
        return load_policy(all_ck[-1]) if all_ck else None

    def trade_loop():
        while True:
            try:
                for s in symbols:
                    bars = fetch_bars(s)
                    if bars is None or bars.height == 0:
                        continue
                    new = bars.filter(
                        pl.col("time") > (last_seen.get(s) or "1970-01-01"))
                    if new.height > 0:
                        store.save_bars(s, bars, 5)
                        last_seen[s] = new["time"][-1]
                        policy = latest_policy()
                        if policy is None:
                            continue
                        sched = Scheduler(risk, store, policy,
                                          {"symbols": symbols,
                                           "window_bars": cfg["training"].get(
                                               "window_bars", 120)})
                        for row in new.to_dicts():
                            sched.on_bar_close(s, row)
            except Exception as e:
                print(f"trade loop error: {e}")
            time.sleep(60)

    threading.Thread(target=train_loop, daemon=True).start()
    threading.Thread(target=trade_loop, daemon=True).start()
    app = create_app(store, risk, cfg)
    uvicorn.run(app, host=cfg["api"]["host"], port=cfg["api"]["port"])


if __name__ == "__main__":
    main()