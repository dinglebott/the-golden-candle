# Deployment copy of 3_algo-backtest/macd.py.
# Returns at most one entry dict, fired within the last `n_active` H1 bars, else None.
# MACD signal-line crossover gated by EMD regime + volume + low-extension; exits are
# fixed ATR multiples. All entry features come from parseData, so this needs H1 only.
import numpy as np

# === Risk (independent ATR multiples, not reward:risk) ===
SL_ATR_MULT = 1.0
TP_ATR_MULT = 2.0

# === Entry filters (direction-symmetric) ===
USE_TREND_REGIME = True
USE_VOL_RATIO = True
USE_VOL_MOMENTUM = True
VOL_RATIO_MIN = 1.0
USE_LOW_EXTENSION = True
DIST_EMA15_MAX = 0.00224

TRADE_LONGS = True
TRADE_SHORTS = True


def get_entries(df, n_active=6):
    n = len(df)
    if n < 2:
        return None

    hist = df["macd_hist"].to_numpy()
    atr = df["raw_atr"].to_numpy()
    close = df["close"].to_numpy()
    regime = df["regime"].to_numpy()
    vol_ratio = df["vol_ratio"].to_numpy()
    vol_momentum = df["vol_momentum"].to_numpy()
    dist_ema15 = df["dist_ema15"].to_numpy()

    window_start = max(1, n - n_active)
    last_entry = None

    for i in range(window_start, n):
        h = hist[i]
        h_prev = hist[i-1]
        atr_v = atr[i]
        c = close[i]
        if np.isnan(h) or np.isnan(h_prev) or np.isnan(atr_v) or atr_v <= 0:
            continue

        bull_cross = TRADE_LONGS and h_prev <= 0 and h > 0
        bear_cross = TRADE_SHORTS and h_prev >= 0 and h < 0
        if not (bull_cross or bear_cross):
            continue

        if USE_TREND_REGIME and regime[i] != 0:
            continue
        if USE_VOL_RATIO and not (vol_ratio[i] > VOL_RATIO_MIN):
            continue
        if USE_VOL_MOMENTUM and not (vol_momentum[i] > 0):
            continue
        if USE_LOW_EXTENSION and not (abs(dist_ema15[i]) <= DIST_EMA15_MAX):
            continue

        d = 1 if bull_cross else -1
        last_entry = {
            "direction": "long" if d == 1 else "short",
            "entry": float(c),
            "tp": float(c + d * TP_ATR_MULT * atr_v),
            "sl": float(c - d * SL_ATR_MULT * atr_v),
            "tp_type": "atr",
            "time": str(df["time"].iat[i]),
        }

    return last_entry
