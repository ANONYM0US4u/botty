import uvicorn
from datetime import datetime, timedelta

from engine.api.main import create_app
from engine.brokers.simulator import SimulatorAdapter
from engine.config import load_config
from engine.data.fetchers.ccxt_crypto import fetch_crypto_bars
from engine.data.fetchers.yfinance_nse import fetch_nse_minute_bars
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

    def fetch_bars(symbol: str):
        if symbol in stocks:
            return add_indicators(
                fetch_nse_minute_bars(symbol, stock_start, end, "5m"))
        if symbol in crypto:
            ccxt_symbol = f"{symbol[:-4]}/{symbol[-4:]}:USDT"  # BTCUSDT -> BTC/USDT:USDT
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
    emitter = api_main.emitter  # the singleton the WS route registers clients with
    theater = TrainingTheater(store, emitter, cfg, make_fetch_bars(cfg))
    app = create_app(store, risk, cfg, theater=theater)
    uvicorn.run(app, host=cfg["api"]["host"], port=cfg["api"]["port"])


if __name__ == "__main__":
    main()