import polars as pl
import pytest
from engine.data.fetchers.yfinance_nse import fetch_nse_minute_bars


def test_empty_result_raises(monkeypatch):
    class FakeTicker:
        def history(self, **kw):
            return __import__("pandas").DataFrame()
    monkeypatch.setattr("yfinance.Ticker", lambda s: FakeTicker())
    with pytest.raises(ValueError):
        fetch_nse_minute_bars("RELIANCE.NS", "2026-01-02", "2026-01-03")


def test_normalizes_columns(monkeypatch):
    import pandas as pd
    df = pd.DataFrame({
        "Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [100.5],
        "Volume": [1000],
    }, index=pd.to_datetime(["2026-01-02 09:15:00+05:30"]))
    class FakeTicker:
        def history(self, **kw):
            return df
    monkeypatch.setattr("yfinance.Ticker", lambda s: FakeTicker())
    out = fetch_nse_minute_bars("RELIANCE.NS", "2026-01-02", "2026-01-03")
    assert isinstance(out, pl.DataFrame)
    assert out.columns == ["time", "open", "high", "low", "close", "volume"]
    assert str(out["time"][0]).startswith("2026-01-02 09:15")