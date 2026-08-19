import os
import subprocess
import sys
import threading

import uvicorn
from datetime import datetime, timedelta

from engine.api.heartbeat import Heartbeat, run_watchdog
from engine.api.main import create_app
from engine.config import load_config
from engine.data.fetchers.ccxt_crypto import fetch_crypto_bars
from engine.data.fetchers.yfinance_nse import fetch_nse_minute_bars
from engine.data.fetchers.cache import load_cached, save_cache
from engine.data.indicators import add_indicators
from engine.data.store import DataStore
from engine.live.risk import RiskGateway
from engine.training.theater import TrainingTheater


def make_fetch_bars(cfg: dict):
    stocks = set(cfg["instruments"]["stocks"])
    crypto = set(cfg["instruments"]["crypto"])
    end = datetime.now().strftime("%Y-%m-%d")
    crypto_start = (datetime.now() - timedelta(days=33)).strftime("%Y-%m-%d")
    stock_start = (datetime.now() - timedelta(days=55)).strftime("%Y-%m-%d")

    def network_fetch(symbol: str):
        if symbol in stocks:
            return add_indicators(
                fetch_nse_minute_bars(symbol, stock_start, end, "5m"))
        if symbol in crypto:
            ccxt_symbol = f"{symbol[:-4]}/{symbol[-4:]}:USDT"  # BTCUSDT -> BTC/USDT:USDT
            return add_indicators(
                fetch_crypto_bars(ccxt_symbol, crypto_start, end, "5m", "gate"))
        raise ValueError(f"symbol {symbol} not configured")

    def fetch_bars(symbol: str):
        try:
            df = network_fetch(symbol)
            save_cache(cfg, symbol, df)
            return df
        except Exception:
            cached = load_cached(cfg, symbol)
            if cached is not None:
                return cached
            raise

    return fetch_bars


def _find_cmd_pid(substr: str) -> int | None:
    """PID of the cmd.exe window whose command line contains substr."""
    ps = ("powershell -NoProfile -Command "
          f"(Get-CimInstance Win32_Process | Where-Object {{ "
          f"$_.Name -eq 'cmd.exe' -and $_.CommandLine -like '*{substr}*' "
          f"}}).ProcessId")
    try:
        out = subprocess.run(ps, capture_output=True, text=True, timeout=15,
                             creationflags=subprocess.CREATE_NO_WINDOW)
        pids = [int(p) for p in out.stdout.split() if p.strip().isdigit()]
        return pids[0] if pids else None
    except Exception:
        return None


def _kill_tree(pid: int) -> None:
    try:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       capture_output=True, timeout=15,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    except Exception:
        pass


def _shutdown_all() -> None:
    """Close the dashboard window + backend window (and their trees), then exit."""
    for substr in ("next start -p 3001", "run_live.py"):
        pid = _find_cmd_pid(substr)
        if pid is not None:
            _kill_tree(pid)
    os._exit(0)


def main() -> None:
    cfg = load_config()
    store = DataStore(cfg["storage"]["db_path"], cfg["storage"]["parquet_dir"])
    store.init_schema()
    from engine.api import main as api_main
    emitter = api_main.emitter  # the singleton the WS route registers clients with
    fetch_bars = make_fetch_bars(cfg)
    from engine.brokers.simulator import SimulatorAdapter
    risk = RiskGateway(
        SimulatorAdapter(slippage_bps=cfg["brokers"].get("slippage_bps", 2.0),
                         latency_bars=cfg["brokers"].get("latency_bars", 1)),
        cfg["risk"], store=store,
        flatten_symbols=set(cfg["instruments"]["stocks"]))
    theater = TrainingTheater(store, emitter, cfg, fetch_bars)
    from engine.trading.mode import BotMode
    mode = BotMode(store, emitter, cfg, theater, fetch_bars, risk=risk)
    hb = Heartbeat()
    app = create_app(store, risk, cfg, theater=theater, mode=mode, heartbeat=hb)
    threading.Thread(target=run_watchdog, args=(hb, _shutdown_all),
                     daemon=True).start()
    uvicorn.run(app, host=cfg["api"]["host"], port=cfg["api"]["port"])


if __name__ == "__main__":
    main()