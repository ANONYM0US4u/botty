import polars as pl
import pytest
from engine.data.fetchers.probe_data_availability import probe_symbol


def test_probe_reports_sane_range(monkeypatch, tmp_path):
    import pandas as pd
    n = 100
    idx = pd.date_range("2026-01-02 09:15:00+05:30", periods=n, freq="5min")
    df = pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0,
                       "Close": 100.5, "Volume": 1000.0}, index=idx)
    class FakeTicker:
        def history(self, **kw):
            return df
    monkeypatch.setattr("yfinance.Ticker", lambda s: FakeTicker())
    rep = probe_symbol("RELIANCE.NS", interval="5m", out_path=tmp_path / "avail.yaml")
    assert rep["symbol"] == "RELIANCE.NS"
    assert rep["first_ts"] < rep["last_ts"]
    assert rep["bars"] == n
    assert (tmp_path / "avail.yaml").exists()


def test_probe_writes_yaml(monkeypatch, tmp_path):
    import pandas as pd
    idx = pd.to_datetime(["2026-01-02 09:15:00+05:30", "2026-01-02 09:20:00+05:30"])
    df = pd.DataFrame({"Open": [100.0, 101.0], "High": [101.0, 102.0],
                       "Low": [99.0, 100.0], "Close": [100.5, 101.5],
                       "Volume": [1000, 1200]}, index=idx)
    class FakeTicker:
        def history(self, **kw):
            return df
    monkeypatch.setattr("yfinance.Ticker", lambda s: FakeTicker())
    out = tmp_path / "avail.yaml"
    probe_symbol("RELIANCE.NS", interval="5m", out_path=out)
    import yaml
    data = yaml.safe_load(out.read_text())
    assert data["RELIANCE.NS"]["bars"] == 2