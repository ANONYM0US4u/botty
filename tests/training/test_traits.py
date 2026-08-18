import pytest
from engine.training.traits import compute_traits


def test_empty_inputs():
    t = compute_traits([], [], {"max_position_pct": 30.0})
    assert t["trades"] == 0 and t["win_rate"] == 0.0
    assert t["avg_hold_bars"] == 0.0 and t["long_short_bias"] == 0.0
    assert t["trade_frequency"] == 0.0
    assert t["max_position_notional_pct"] == 0.0


def test_hold_bars_and_bias():
    fills = [
        {"symbol": "X", "side": "buy", "qty": 20.0, "price": 100.0, "ts": "2026-01-02 09:30:00"},
        {"symbol": "X", "side": "sell", "qty": 10.0, "price": 102.0, "ts": "2026-01-02 09:35:00"},
        {"symbol": "X", "side": "sell", "qty": 5.0, "price": 50.0, "ts": "2026-01-02 09:40:00"},
        {"symbol": "X", "side": "buy", "qty": 5.0, "price": 49.0, "ts": "2026-01-02 09:45:00"},
    ]
    decisions = [{"action": a} for a in ["long", "long", "short", "short", "flat"]]
    t = compute_traits(fills, decisions, {"max_position_pct": 30.0})
    assert t["trades"] == 4
    assert t["trade_frequency"] == pytest.approx(4 / 5)
    assert t["win_rate"] == 1.0          # both round trips profitable
    assert t["long_short_bias"] == pytest.approx(0.25)  # (25-15)/40
    assert t["avg_hold_bars"] == pytest.approx(1.0)     # both pairs are consecutive fills


def test_max_position_pct():
    fills = [{"symbol": "X", "side": "buy", "qty": 300.0, "price": 100.0, "ts": "t1"}]
    t = compute_traits(fills, [{"action": "long"}], {"max_position_pct": 30.0})
    assert t["max_position_notional_pct"] == pytest.approx(0.3)  # 30000/100000