import time
import polars as pl
import ccxt


def fetch_crypto_bars(symbol: str, start: str, end: str, interval: str = "5m",
                      exchange_id: str = "gate") -> pl.DataFrame:
    exch_cls = getattr(ccxt, exchange_id, None)
    if exch_cls is None:
        raise ValueError(f"exchange {exchange_id} unavailable")
    ex = exch_cls({"enableRateLimit": True})
    tf_sec = {"1m": 60, "5m": 300, "15m": 900}[interval]
    since = int(time.mktime(time.strptime(start, "%Y-%m-%d"))) * 1000
    until = int(time.mktime(time.strptime(end, "%Y-%m-%d"))) * 1000
    rows = []
    cur = since
    while cur < until:
        batch = ex.fetch_ohlcv(symbol, interval, since=cur, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        cur = batch[-1][0] + tf_sec * 1000
        if len(batch) < 1000:
            break
    if not rows:
        raise ValueError(f"no data for {symbol}")
    df = pl.DataFrame(
        {"time": [time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(r[0] / 1000)) for r in rows],
         "open": [r[1] for r in rows], "high": [r[2] for r in rows],
         "low": [r[3] for r in rows], "close": [r[4] for r in rows],
         "volume": [r[5] for r in rows]}).unique(subset=["time"])
    return df.sort("time")