import math
from engine.eval.metrics import (sharpe_ratio, max_drawdown, win_rate,
                                 profit_factor, compute_eval_report)


def test_sharpe_positive_for_uptrend():
    eq = [100_000 + i * 100 for i in range(100)]
    assert sharpe_ratio(eq) > 0


def test_max_drawdown():
    eq = [100_000, 110_000, 90_000, 100_000]
    assert math.isclose(max_drawdown(eq), -0.1818, abs_tol=1e-3)


def test_win_rate_and_profit_factor():
    trades = [
        {"order_id": "b1", "side": "buy", "qty": 10, "price": 100.0},
        {"order_id": "s1", "side": "sell", "qty": 10, "price": 110.0},
        {"order_id": "b2", "side": "buy", "qty": 10, "price": 100.0},
        {"order_id": "s2", "side": "sell", "qty": 10, "price": 90.0},
    ]
    assert win_rate(trades) == 0.5
    assert profit_factor(trades) == 1.0


def test_eval_report_shape():
    eq = [100_000, 101_000, 100_500]
    rep = compute_eval_report(eq, [])
    assert set(["sharpe", "max_drawdown", "win_rate", "profit_factor", "total_return"]) <= set(rep)