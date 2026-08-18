import yaml
from pathlib import Path
import pandas as pd
import polars as pl
import yfinance as yf


def _probe_one(symbol: str, interval: str) -> dict:
    """Fetch the maximum reachable intraday history for one symbol.

    yfinance intraday limits: 1m = ~30 days, 5m/15m = ~60 days. This is the
    authoritative reach check — never assume a window the probe did not prove.
    """
    raw = yf.Ticker(symbol).history(period="60d", interval=interval,
                                    auto_adjust=False)
    if raw is None or raw.empty:
        return {"symbol": symbol, "source": "yfinance", "first_ts": "",
                "last_ts": "", "bars": 0}
    df = raw.reset_index()
    if "Datetime" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Datetime"})
    df = df[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    if df["time"].dt.tz is not None:
        df["time"] = df["time"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    bars = pl.from_pandas(df).sort("time")
    return {"symbol": symbol, "source": "yfinance",
            "first_ts": str(bars["time"][0]), "last_ts": str(bars["time"][-1]),
            "bars": bars.height}


def probe_symbol(symbol: str, interval: str = "5m",
                 out_path: str | Path | None = None) -> dict:
    rep = _probe_one(symbol, interval)
    if out_path is not None:
        out_path = Path(out_path)
        existing = {}
        if out_path.exists():
            existing = yaml.safe_load(out_path.read_text()) or {}
        existing[symbol] = rep
        out_path.write_text(yaml.safe_dump(existing, sort_keys=False))
    return rep


def probe_all(symbols: list[str], interval: str = "5m",
              out_path: str | Path = "config/data_availability.yaml") -> dict:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for sym in symbols:
        probe_symbol(sym, interval=interval, out_path=out_path)
    return yaml.safe_load(out_path.read_text())