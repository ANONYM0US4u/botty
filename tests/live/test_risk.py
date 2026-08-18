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
    rm.maybe_flatten_before_close(datetime(2026, 1, 2, 15, 16))
    assert rm.broker.get_positions() == []