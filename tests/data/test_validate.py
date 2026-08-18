import polars as pl
from datetime import datetime, timedelta
from engine.data.validate import validate_dataset


def _clean_bars(n=50, interval_min=5, start="2026-01-02 09:15:00"):
    start_dt = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    rows = []
    for i in range(n):
        t = start_dt + timedelta(minutes=i * interval_min)
        px = 100.0 + i * 0.1
        rows.append({"time": t.strftime("%Y-%m-%d %H:%M:%S"),
                     "open": px, "high": px + 1.0, "low": px - 1.0,
                     "close": px + 0.5, "volume": 1000.0})
    return pl.DataFrame(rows)


def _set_row(df, idx, col, value):
    return df.with_columns(
        pl.when(pl.int_range(0, pl.len()) == idx)
        .then(pl.lit(value))
        .otherwise(pl.col(col))
        .alias(col)
    )


def test_clean_dataset_passes():
    rep = validate_dataset(_clean_bars(), "RELIANCE.NS", 5)
    assert rep.is_valid(), rep.failed_checks()


def test_duplicate_timestamps_fail():
    df = _set_row(_clean_bars(), 1, "time", "2026-01-02 09:15:00")
    rep = validate_dataset(df, "RELIANCE.NS", 5)
    assert not rep.is_valid()
    assert any(c["check"] == "no_duplicate_timestamps" and not c["passed"] for c in rep.checks)


def test_future_timestamps_fail():
    df = _set_row(_clean_bars(), 0, "time", "2999-01-01 09:15:00")
    rep = validate_dataset(df, "RELIANCE.NS", 5)
    assert not rep.is_valid()
    assert any(c["check"] == "no_future_timestamps" and not c["passed"] for c in rep.checks)


def test_gap_within_session_fails():
    df = _clean_bars()
    t = "2026-01-02 11:00:00"  # 90 min gap vs 5m bars (within NSE session)
    df = _set_row(df, 10, "time", t)
    rep = validate_dataset(df, "RELIANCE.NS", 5)
    assert not rep.is_valid()
    assert any(c["check"] == "no_unexpected_gaps" and not c["passed"] for c in rep.checks)


def test_overnight_break_allowed_for_stocks():
    df = pl.concat([_clean_bars(10), _clean_bars(5, start="2026-01-03 09:15:00")])
    rep = validate_dataset(df, "RELIANCE.NS", 5)
    assert rep.is_valid()


def test_ohlc_invalid_fails():
    df = _set_row(_clean_bars(), 0, "low", 9999.0)  # low > high
    rep = validate_dataset(df, "RELIANCE.NS", 5)
    assert not rep.is_valid()
    assert any(c["check"] == "ohlc_valid" and not c["passed"] for c in rep.checks)


def test_timezone_aware_fails():
    df = _set_row(_clean_bars(), 0, "time", "2026-01-02 09:15:00+05:30")
    rep = validate_dataset(df, "RELIANCE.NS", 5)
    assert not rep.is_valid()
    assert any(c["check"] == "timezone_normalized" and not c["passed"] for c in rep.checks)


def test_corporate_action_skipped_when_not_configured():
    rep = validate_dataset(_clean_bars(), "RELIANCE.NS", 5)
    ca = [c for c in rep.checks if c["check"] == "corporate_action_policy_applied"]
    assert ca and ca[0]["passed"] is True and ca[0]["detail"] == "skipped"


def test_wrong_interval_fails():
    rep = validate_dataset(_clean_bars(), "RELIANCE.NS", 15)
    assert not rep.is_valid()
    assert any(c["check"] == "timeframe_exists" and not c["passed"] for c in rep.checks)


def test_range_exists_check():
    df = _clean_bars(50)  # spans 09:15..13:15 on 2026-01-02
    rep = validate_dataset(df, "RELIANCE.NS", 5, date_range=("2026-01-02 09:20:00",
                                                             "2026-01-02 12:00:00"))
    assert rep.is_valid()
    rep2 = validate_dataset(df, "RELIANCE.NS", 5, date_range=("2026-01-02 09:00:00",
                                                              "2026-01-02 14:00:00"))
    assert not rep2.is_valid()
    assert any(c["check"] == "range_exists" and not c["passed"] for c in rep2.checks)