import polars as pl
import pytest

from engine.data.fetchers.cache import cache_path, load_cached, save_cache
from scripts.run_live import make_fetch_bars


def _cfg(tmp_path):
    return {"instruments": {"stocks": ["RELIANCE.NS"],
                            "crypto": ["BTCUSDT"]},
            "storage": {"parquet_dir": str(tmp_path / "pq")}}


def _bars(n=400):
    import numpy as np
    rng = np.random.default_rng(3)
    px = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    return pl.DataFrame({
        "time": [f"2026-01-02 09:{15 + i % 60:02d}:00" for i in range(n)],
        "open": px, "high": px + 1.0, "low": px - 1.0, "close": px,
        "volume": [1000.0] * n})


def test_cache_roundtrip(tmp_path):
    cfg = _cfg(tmp_path)
    save_cache(cfg, "RELIANCE.NS", _bars())
    cached = load_cached(cfg, "RELIANCE.NS")
    assert cached is not None and cached.height == 400
    assert cache_path(cfg, "RELIANCE.NS").exists()


def test_load_cached_missing_returns_none(tmp_path):
    assert load_cached(_cfg(tmp_path), "RELIANCE.NS") is None


def test_fetch_saves_cache_on_success(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.run_live.fetch_crypto_bars",
        lambda *a, **k: _bars())
    fetch = make_fetch_bars(_cfg(tmp_path))
    df = fetch("BTCUSDT")
    assert df.height >= 380
    assert "ema9" in df.columns
    assert cache_path(_cfg(tmp_path), "BTCUSDT").exists()


def test_fetch_falls_back_to_cache(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("no network")
    monkeypatch.setattr(
        "scripts.run_live.fetch_crypto_bars", boom)
    cfg = _cfg(tmp_path)
    save_cache(cfg, "BTCUSDT", _bars())
    fetch = make_fetch_bars(cfg)
    df = fetch("BTCUSDT")
    assert df.height == 400


def test_fetch_fallback_rethrows_without_cache(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("no network")
    monkeypatch.setattr(
        "scripts.run_live.fetch_crypto_bars", boom)
    fetch = make_fetch_bars(_cfg(tmp_path))
    with pytest.raises(ConnectionError):
        fetch("BTCUSDT")