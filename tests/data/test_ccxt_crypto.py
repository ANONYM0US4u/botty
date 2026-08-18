import polars as pl
import pytest
from engine.data.fetchers.ccxt_crypto import fetch_crypto_bars


def test_empty_raises(monkeypatch):
    class FakeEx:
        def fetch_ohlcv(self, symbol, tf, since, limit):
            return []
    monkeypatch.setattr("engine.data.fetchers.ccxt_crypto.ccxt", None)
    monkeypatch.setattr("ccxt.bybit", lambda x: FakeEx())
    with pytest.raises(ValueError):
        fetch_crypto_bars("BTC/USDT:USDT", "2026-01-01", "2026-01-02")


def test_normalizes(monkeypatch):
    import time as _t
    ts = int(_t.mktime(_t.strptime("2026-01-02 09:15:00", "%Y-%m-%d %H:%M:%S"))) * 1000
    class FakeEx:
        def fetch_ohlcv(self, symbol, tf, since, limit):
            return [[ts, 100.0, 101.0, 99.0, 100.5, 1000.0]]
    monkeypatch.setattr("ccxt.bybit", lambda x: FakeEx())
    out = fetch_crypto_bars("BTC/USDT:USDT", "2026-01-01", "2026-01-02")
    assert isinstance(out, pl.DataFrame)
    assert out["close"][0] == 100.5