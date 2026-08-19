import polars as pl
import pandas as pd
import yfinance as yf


def fetch_nse_minute_bars(symbol: str, start: str, end: str,
                          interval: str = "5m") -> pl.DataFrame:
    raw = yf.Ticker(symbol).history(start=start, end=end, interval=interval,
                                    auto_adjust=False)
    if raw is None or raw.empty:
        raise ValueError(f"no data for {symbol}")
    df = raw.reset_index()
    if "Datetime" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Datetime"})
    df = df[["Datetime", "Open", "High", "Low", "Close", "Volume"]]
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    if df["time"].dt.tz is not None:
        df["time"] = df["time"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df = df.astype({"open": "float64", "high": "float64", "low": "float64",
                    "close": "float64", "volume": "float64"})
    df["time"] = df["time"].astype("datetime64[ns]")
    return pl.from_pandas(df)