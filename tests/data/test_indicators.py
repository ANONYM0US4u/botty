import polars as pl
from engine.data.indicators import add_indicators


def _df(n=50):
    rows = []
    for i in range(n):
        rows.append({"time": f"2026-01-02 09:{15+i:02d}:00", "open": 100.0 + i,
                     "high": 101.0 + i, "low": 99.0 + i, "close": 100.5 + i,
                     "volume": 1000.0})
    return pl.DataFrame(rows)


def test_columns_added():
    out = add_indicators(_df())
    for col in ["ema9", "ema21", "rsi14", "atr14", "vwap", "session_band", "ret1", "vol20",
                "boll_pos", "macd_hist", "body_pct", "upper_wick_pct",
                "lower_wick_pct", "dist_to_high_pct"]:
        assert col in out.columns


def test_known_rsi():
    out = add_indicators(_df())
    rsi = out["rsi14"].tail(1).item()
    assert rsi > 80.0


def test_atr_positive():
    out = add_indicators(_df())
    assert out["atr14"].drop_nulls().min() > 0.0


def test_vwap_is_session_typical_price_mean():
    # Constant volume => VWAP = cumulative mean of typical prices (causal).
    out = add_indicators(_df())
    tp = ((out["high"] + out["low"] + out["close"]) / 3.0)
    expected = tp.head(50).mean()
    assert out["vwap"].tail(1).item() == expected


def test_indicators_are_causal():
    # No lookahead: values at index i must not change when later bars are removed.
    full = add_indicators(_df())
    cut = add_indicators(_df(30))
    for col in ["ema9", "ema21", "rsi14", "atr14", "vwap", "session_band", "ret1", "vol20",
                "boll_pos", "macd_hist", "body_pct", "upper_wick_pct",
                "lower_wick_pct", "dist_to_high_pct"]:
        a = full[col].head(30).fill_null(0.0).to_list()
        b = cut[col].fill_null(0.0).to_list()
        assert a == b, f"{col} is not causal"


def test_pattern_indicators_bounded():
    out = add_indicators(_df())
    body = out["body_pct"].drop_nulls()
    assert (body >= 0.0).all() and (body <= 1.0).all()
    uw = out["upper_wick_pct"].drop_nulls()
    lw = out["lower_wick_pct"].drop_nulls()
    assert (uw >= 0.0).all() and (uw <= 1.0).all()
    assert (lw >= 0.0).all() and (lw <= 1.0).all()
    assert (out["body_pct"] + out["upper_wick_pct"] +
            out["lower_wick_pct"]).drop_nulls().abs().max() <= 1.0001
    bp = out["boll_pos"].drop_nulls()
    assert bp.abs().max() <= 2.0
    dth = out["dist_to_high_pct"].drop_nulls()
    assert (dth >= 0.0).all() and (dth <= 1.0).all()


def test_pattern_indicators_no_nan_after_warmup():
    out = add_indicators(_df(60))
    tail = out.tail(10)
    for col in ["boll_pos", "macd_hist", "body_pct", "upper_wick_pct",
                "lower_wick_pct", "dist_to_high_pct"]:
        assert tail[col].null_count() == 0, col