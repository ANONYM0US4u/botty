import polars as pl


def _ewm(series: pl.Series, span: int) -> pl.Series:
    alpha = 2.0 / (span + 1.0)
    vals = series.to_list()
    out = []
    prev = None
    for v in vals:
        if v is None or v != v:
            out.append(None)
            continue
        if prev is None:
            prev = v
        else:
            prev = alpha * v + (1 - alpha) * prev
        out.append(prev)
    return pl.Series(out)


def _rsi(close: pl.Series, period: int = 14) -> pl.Series:
    diff = close.diff().fill_null(0.0)
    gain = diff.map_elements(lambda x: max(x, 0.0), return_dtype=pl.Float64)
    loss = diff.map_elements(lambda x: max(-x, 0.0), return_dtype=pl.Float64)
    avg_gain = _ewm(gain, period)
    avg_loss = _ewm(loss, period)
    rs = avg_gain / avg_loss.replace(0.0, 1e-12)
    return (100.0 - 100.0 / (1.0 + rs)).replace(float("inf"), 100.0)


def add_indicators(df: pl.DataFrame) -> pl.DataFrame:
    out = df.with_columns([
        pl.col("close").map_batches(lambda s: _ewm(s, 9)).alias("ema9"),
        pl.col("close").map_batches(lambda s: _ewm(s, 21)).alias("ema21"),
        pl.col("close").map_batches(lambda s: _rsi(s)).alias("rsi14"),
    ])
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - pl.col("close").shift(1)).abs(),
        (pl.col("low") - pl.col("close").shift(1)).abs(),
    ).fill_null(0.0)
    out = out.with_columns([
        tr.map_batches(lambda s: _ewm(s, 14)).alias("atr14"),
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("_typical_price"),
    ])
    out = out.with_columns(
        ((pl.col("_typical_price") * pl.col("volume")).cum_sum()
         / pl.col("volume").cum_sum()).alias("vwap")
    )
    out = out.with_columns([
        (pl.col("close") / pl.col("close").shift(1) - 1.0).alias("ret1"),
        pl.col("close").rolling_std(20).alias("vol20"),
        (pl.col("high").rolling_max(30) - pl.col("low").rolling_min(30)).alias("session_band"),
    ])
    return out.drop("_typical_price")