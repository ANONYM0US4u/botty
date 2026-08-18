import uvicorn

from engine.api.main import create_app
from engine.brokers.simulator import SimulatorAdapter
from engine.config import load_config
from engine.data.store import DataStore
from engine.live.risk import RiskGateway


def main() -> None:
    cfg = load_config()
    store = DataStore(cfg["storage"]["db_path"], cfg["storage"]["parquet_dir"])
    store.init_schema()
    broker = SimulatorAdapter(
        slippage_bps=cfg["brokers"].get("slippage_bps", 2.0),
        latency_bars=cfg["brokers"].get("latency_bars", 1))
    risk = RiskGateway(broker, cfg["risk"])
    app = create_app(store, risk, cfg)
    uvicorn.run(app, host=cfg["api"]["host"], port=cfg["api"]["port"])


if __name__ == "__main__":
    main()