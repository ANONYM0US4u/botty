from hypothesis import given, strategies as st
from engine.live.risk import RiskGateway
from engine.brokers.simulator import SimulatorAdapter

G = {"daily_loss_limit_pct": -3.0, "max_position_pct": 30.0,
     "max_total_exposure_pct": 90.0, "stale_data_seconds": 120,
     "flatten_at": "15:15"}


@given(qty=st.integers(-5000, 5000), price=st.floats(1, 10000),
       equity=st.floats(1, 10_000_000), day_pnl=st.floats(-1_000_000, 1_000_000),
       killed=st.booleans(), stale=st.booleans(), nan_price=st.booleans())
def test_invariants_never_violated(qty, price, equity, day_pnl, killed, stale, nan_price):
    rm = RiskGateway(SimulatorAdapter(), G)
    if killed:
        rm.set_kill_switch(True)
    if stale:
        rm.set_last_bar_ts(0.0)
    p = float("nan") if nan_price else price
    ok, _ = rm.check_order({"symbol": "X", "qty": qty, "price": p}, [], equity, day_pnl)
    if ok:
        assert abs(qty) * price / equity * 100.0 <= 30.0
        assert day_pnl / equity * 100.0 > -3.0
        assert not killed and not stale and not nan_price