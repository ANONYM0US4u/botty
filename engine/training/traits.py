"""Behavior fingerprint of a policy checkpoint, computed from its replay."""


def compute_traits(fills: list[dict], decisions: list[dict],
                   risk_cfg: dict) -> dict:
    trades = len(fills)
    n_dec = max(len(decisions), 1)
    trade_frequency = trades / n_dec

    # avg hold bars: bars between an opening fill and the closing fill on the
    # same symbol in the opposite direction (via decision bar indices)
    ts_idx = {d.get("ts"): i for i, d in enumerate(decisions)
              if d.get("ts") is not None}
    hold_gaps = []
    # win rate: close each buy-sell pair on same symbol, price-based
    wins = 0
    pairs = 0
    open_side: dict[str, tuple[str, float, int]] = {}
    for i, f in enumerate(fills):
        sym = f["symbol"]
        if sym in open_side and open_side[sym][0] != f["side"]:
            entry_side, entry_price, entry_i = open_side[sym]
            pnl = (f["price"] - entry_price) if f["side"] == "sell" \
                else (entry_price - f["price"])
            pairs += 1
            wins += 1 if pnl > 0 else 0
            ei = ts_idx.get(fills[entry_i].get("ts"))
            xi = ts_idx.get(f.get("ts"))
            if ei is not None and xi is not None and xi > ei:
                hold_gaps.append(xi - ei)
            else:
                hold_gaps.append(i - entry_i)
            del open_side[sym]
        else:
            open_side[sym] = (f["side"], float(f["price"]), i)
    avg_hold_bars = float(sum(hold_gaps) / len(hold_gaps)) if hold_gaps else 0.0
    win_rate = wins / pairs if pairs else 0.0

    # max position notional vs equity (100k base)
    equity = float(risk_cfg.get("theater_equity_base", 100_000.0))
    max_notional = max((abs(float(f.get("qty", 0))) * float(f.get("price", 0))
                        for f in fills), default=0.0)
    max_position_notional_pct = max_notional / equity if equity else 0.0

    long_qty = sum(f["qty"] for f in fills if f["side"] == "buy")
    short_qty = sum(f["qty"] for f in fills if f["side"] == "sell")
    total = long_qty + short_qty
    long_short_bias = (long_qty - short_qty) / total if total else 0.0

    return {"trades": trades, "avg_hold_bars": avg_hold_bars,
            "trade_frequency": trade_frequency, "win_rate": win_rate,
            "max_position_notional_pct": max_position_notional_pct,
            "long_short_bias": long_short_bias}