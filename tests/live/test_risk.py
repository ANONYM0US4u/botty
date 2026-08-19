from engine.live.risk import RiskGateway
from engine.brokers.simulator import SimulatorAdapter


def _gate():
    return RiskGateway(SimulatorAdapter(), {"daily_loss_limit_pct": -3.0,
                       "max_position_pct": 30.0, "max_total_exposure_pct": 90.0,
                       "stale_data_seconds": 120, "flatten_at": "15:15"})


def test_daily_loss_limit_blocks():
    rm = _gate()
    ok, why = rm.check_order({"symbol": "X", "qty": 1}, [], 100_000.0, day_pnl=-4_000.0)
    assert ok is False and "loss" in why


def test_position_size_cap():
    rm = _gate()
    ok, why = rm.check_order({"symbol": "X", "qty": 400, "price": 100.0}, [], 100_000.0,
                             day_pnl=0.0)
    # 400 shares * 100 price = 40k = 40% > 30% cap
    assert ok is False and "position" in why


def test_exposure_cap():
    rm = _gate()
    positions = [{"symbol": "A", "qty": 200, "price": 500.0}]  # 100k = 100%
    ok, why = rm.check_order({"symbol": "B", "qty": 200, "price": 100.0}, positions,
                             100_000.0, day_pnl=0.0)
    assert ok is False and "exposure" in why


def test_kill_switch():
    rm = _gate()
    rm.set_kill_switch(True)
    ok, why = rm.check_order({"symbol": "X", "qty": 1}, [], 100_000.0, day_pnl=0.0)
    assert ok is False and "kill" in why
    rm.set_kill_switch(False)


def test_stale_data_blocks():
    rm = _gate()
    rm.set_last_bar_ts(123.0)  # seconds epoch, 10 min old
    ok, why = rm.check_order({"symbol": "X", "qty": 1}, [], 100_000.0, day_pnl=0.0)
    assert ok is False and "stale" in why


def test_nan_state_blocks():
    rm = _gate()
    ok, why = rm.check_order({"symbol": "X", "qty": 1, "price": float("nan")},
                             [], 100_000.0, day_pnl=0.0)
    assert ok is False and "nan" in why


def test_missing_price_blocks():
    rm = _gate()
    ok, why = rm.check_order({"symbol": "X", "qty": 400}, [], 100_000.0, day_pnl=0.0)
    assert ok is False and "position" in why


def test_execute_order_fails_closed_on_permanent_error():
    class Boom(SimulatorAdapter):
        def place_order(self, order):
            from engine.brokers.errors import PermanentBrokerError
            raise PermanentBrokerError("rejected")
    rm = RiskGateway(Boom(), {"daily_loss_limit_pct": -3.0, "max_position_pct": 30.0,
                              "max_total_exposure_pct": 90.0, "stale_data_seconds": 120,
                              "flatten_at": "15:15"})
    res = rm.execute_order({"symbol": "X", "side": "buy", "qty": 1, "price": 100.0})
    assert res["status"] == "failed" and res["retryable"] is False


def test_flatten_before_close():
    from datetime import datetime
    rm = _gate()
    rm.broker.positions["RELIANCE.NS"] = 10.0
    rm.set_last_price("RELIANCE.NS", 2500.0)
    rm.maybe_flatten_before_close(datetime(2026, 1, 2, 15, 16))
    assert rm.broker.get_positions() == []


def test_flatten_uses_last_price_and_skips_unknown_price():
    from datetime import datetime
    rm = _gate()
    rm.broker.positions["RELIANCE.NS"] = 10.0
    rm.maybe_flatten_before_close(datetime(2026, 1, 2, 15, 16))
    assert rm.broker.cash == 100_000.0  # no last price -> no zero-price sale
    rm.set_last_price("RELIANCE.NS", 2500.0)
    rm.broker.positions["RELIANCE.NS"] = 10.0
    rm.maybe_flatten_before_close(datetime(2026, 1, 2, 15, 16))
    assert rm.broker.get_positions() == []
    assert rm.broker.cash == 100_000.0 + 25_000.0


def test_flatten_skips_symbols_outside_flatten_set():
    from datetime import datetime
    rm = RiskGateway(SimulatorAdapter(),
                     {"daily_loss_limit_pct": -3.0, "max_position_pct": 30.0,
                      "max_total_exposure_pct": 90.0, "stale_data_seconds": 120,
                      "flatten_at": "15:15"},
                     flatten_symbols={"RELIANCE.NS"})
    rm.broker.positions["BTCUSDT"] = 1.0
    rm.set_last_price("BTCUSDT", 60_000.0)
    rm.maybe_flatten_before_close(datetime(2026, 1, 2, 15, 16), "BTCUSDT")
    assert rm.broker.get_positions() != []  # 24/7 market: no NSE close flatten
    rm.broker.positions["RELIANCE.NS"] = 10.0
    rm.set_last_price("RELIANCE.NS", 2500.0)
    rm.maybe_flatten_before_close(datetime(2026, 1, 2, 15, 16), "RELIANCE.NS")
    assert rm.broker.positions["RELIANCE.NS"] == 0.0
    assert rm.broker.positions["BTCUSDT"] == 1.0  # out-of-scope kept


def test_execute_order_records_fill_when_store_present(tmp_path):
    from engine.data.store import DataStore
    store = DataStore(tmp_path / "t.db", tmp_path / "pq")
    store.init_schema()
    rm = RiskGateway(SimulatorAdapter(), {"daily_loss_limit_pct": -3.0,
                                          "max_position_pct": 30.0,
                                          "max_total_exposure_pct": 90.0,
                                          "stale_data_seconds": 120,
                                          "flatten_at": "15:15"},
                     store=store)
    res = rm.execute_order({"symbol": "X", "side": "buy", "qty": 10, "price": 100.0})
    assert res["status"] == "open"       # queued: fills at the NEXT bar open
    assert store.get_trades() == []
    rm.on_bar_close("X", {"time": "2026-01-02 09:25:00", "open": 100.5})  # -> pending
    assert store.get_trades() == []
    rm.on_bar_close("X", {"time": "2026-01-02 09:30:00", "open": 101.0})  # filled
    trades = store.get_trades()
    assert len(trades) == 1
    assert trades[0]["qty"] == 10 and trades[0]["ts"] == "2026-01-02 09:30:00"
    assert rm.broker.get_positions()[0]["qty"] == 10.0


def test_execute_order_buffers_fill_without_store():
    rm = _gate()
    res = rm.execute_order({"symbol": "X", "side": "buy", "qty": 10, "price": 100.0})
    assert res["status"] == "open"
    rm.on_bar_close("X", {"time": "2026-01-02 09:25:00", "open": 100.5})
    rm.on_bar_close("X", {"time": "2026-01-02 09:30:00", "open": 101.0})
    assert len(rm.get_fills()) == 1
    assert rm.get_fills()[0]["price"] == 101.0 * (1.0 + 2.0 / 10_000.0)