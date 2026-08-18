import math


def sharpe_ratio(equity: list[float], periods_per_year: int = 252 * 75) -> float:
    if len(equity) < 3:
        return 0.0
    returns = [equity[i] / equity[i - 1] - 1.0 for i in range(1, len(equity))]
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / len(returns)
    if var == 0:
        return 0.0
    return (mean / math.sqrt(var)) * math.sqrt(periods_per_year)


def max_drawdown(equity: list[float]) -> float:
    peak = -float("inf")
    mdd = 0.0
    for v in equity:
        peak = max(peak, v)
        mdd = min(mdd, (v - peak) / peak)
    return mdd


def _realized_pnl(trades: list[dict]) -> list[dict]:
    lots = []  # (qty, price)
    out = []
    for t in trades:
        if t["side"] == "buy":
            lots.append((t["qty"], t["price"]))
        else:
            qty, price = t["qty"], t["price"]
            while qty > 0 and lots:
                bq, bp = lots.pop(0)
                take = min(bq, qty)
                out.append({**t, "pnl": (price - bp) * take})
                qty -= take
                if bq > take:
                    lots.insert(0, (bq - take, bp))
    return out


def win_rate(trades: list[dict]) -> float:
    closed = _realized_pnl(trades)
    if not closed:
        return 0.0
    return sum(1 for t in closed if t["pnl"] > 0) / len(closed)


def profit_factor(trades: list[dict]) -> float:
    closed = _realized_pnl(trades)
    gross_win = sum(t["pnl"] for t in closed if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in closed if t["pnl"] < 0))
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return gross_win / gross_loss


def compute_eval_report(equity: list[float], trades: list[dict],
                        bench_equity: list[float] | None = None) -> dict:
    rep = {
        "sharpe": round(sharpe_ratio(equity), 3),
        "max_drawdown": round(max_drawdown(equity), 4),
        "win_rate": round(win_rate(trades), 3),
        "profit_factor": round(profit_factor(trades), 3),
        "total_return": round(equity[-1] / equity[0] - 1.0, 4) if len(equity) > 1 else 0.0,
    }
    if bench_equity and len(bench_equity) > 1:
        rep["vs_benchmark_return"] = round(equity[-1] / equity[0] - bench_equity[-1] / bench_equity[0], 4)
    return rep