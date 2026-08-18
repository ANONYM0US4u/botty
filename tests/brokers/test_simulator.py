import pytest
from engine.brokers.simulator import SimulatorAdapter


def test_place_and_fill_on_next_bar_open():
    sim = SimulatorAdapter(initial_cash=100_000.0, slippage_bps=0.0, latency_bars=0)
    order = sim.place_order({"symbol": "RELIANCE.NS", "side": "buy", "qty": 10})
    assert order["status"] == "open"
    sim.on_bar_close("RELIANCE.NS", {"time": "09:15", "open": 99.0, "close": 100.0})
    assert sim.get_positions() == []  # not filled at decision-bar close
    sim.on_bar_close("RELIANCE.NS", {"time": "09:20", "open": 100.0, "close": 101.0})
    pos = sim.get_positions()
    assert any(p["symbol"] == "RELIANCE.NS" and p["qty"] == 10 for p in pos)
    assert sim.last_fill_price == pytest.approx(100.0)  # next bar OPEN
    assert sim.get_balance() < 100_000.0


def test_slippage_applied_on_next_open():
    sim = SimulatorAdapter(initial_cash=100_000.0, slippage_bps=100.0, latency_bars=0)
    sim.place_order({"symbol": "BTCUSDT", "side": "buy", "qty": 1})
    sim.on_bar_close("BTCUSDT", {"open": 99.0, "close": 100.0})
    sim.on_bar_close("BTCUSDT", {"open": 100.0, "close": 101.0})
    assert sim.last_fill_price == pytest.approx(101.0)


def test_no_lookahead_fill_price():
    # Fill must use next bar open, never the decision bar close.
    sim = SimulatorAdapter(initial_cash=100_000.0, slippage_bps=0.0, latency_bars=0)
    sim.place_order({"symbol": "RELIANCE.NS", "side": "buy", "qty": 10})
    sim.on_bar_close("RELIANCE.NS", {"open": 100.0, "close": 110.0})
    sim.on_bar_close("RELIANCE.NS", {"open": 105.0, "close": 106.0})
    assert sim.last_fill_price == pytest.approx(105.0)  # 110.0 would be lookahead


def test_cancel_order():
    sim = SimulatorAdapter()
    o = sim.place_order({"symbol": "RELIANCE.NS", "side": "buy", "qty": 10})
    assert sim.cancel_order(o["id"]) is True
    assert len(sim.get_orders()) == 0