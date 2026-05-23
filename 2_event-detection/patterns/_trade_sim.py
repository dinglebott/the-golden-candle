import numpy as np


def simulate_trade(highs, lows, closes, i, n_candles, entry, tp, sl, is_long):
    """Mirrors backtest.run_triple_barrier per-trade logic (BE disabled): pessimistic
    intra-bar ordering — SL is checked before TP, so a single bar that hits both
    barriers resolves as a stop. Returns (outcome, r_multiple) where outcome is one of
    "fill", "stopped", "expired".
    """
    n_rows = len(closes)
    risk = abs(entry - sl)
    if risk == 0:
        return "expired", 0.0

    forward_end = min(i + 1 + n_candles, n_rows)
    for j in range(i + 1, forward_end):
        if is_long:
            stopped = lows[j] <= sl
            filled = highs[j] >= tp
        else:
            stopped = highs[j] >= sl
            filled = lows[j] <= tp
        if stopped:
            return "stopped", -1.0
        if filled:
            return "fill", abs(tp - entry) / risk

    last_j = forward_end - 1
    if last_j > i:
        exit_price = closes[last_j]
        pnl = (exit_price - entry) if is_long else (entry - exit_price)
        return "expired", pnl / risk
    return "expired", 0.0
