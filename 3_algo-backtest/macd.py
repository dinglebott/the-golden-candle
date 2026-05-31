import numpy as np
import pandas as pd

# MACD signal-line crossover. Every histogram sign change is a candidate; the
# filters below gate which crossovers are actually traded. With all filters off
# and SL/TP at 1.5/2.25 ATR this is the pure-crossover baseline. The default
# ON set is the breadth-first winner (see claude_tuning_logs/macd.md): a
# non-cyclic EMD regime plus a volume surge — the only filter family that held
# up out-of-sample. Directional trend filters (ADX, EMA alignment, zero-line,
# H4 trend) were tested and did NOT help, so none are used.

# === Risk ===
# Independent ATR multiples (NOT reward:risk). Fine-tuning found a tight stop +
# far target (RR=2.0) best out-of-sample; widening the stop hurt. See macd.md V5.
SL_ATR_MULT = 1.0           # SL this many ATRs from entry
TP_ATR_MULT = 2.0           # TP this many ATRs from entry

# === Entry filters (all direction-symmetric) ===
USE_TREND_REGIME = True     # require EMD regime == 0 (non-cyclic / trending)
USE_VOL_RATIO = True        # require volume above its rolling average
USE_VOL_MOMENTUM = True     # require rising volume ratio
VOL_RATIO_MIN = 1.0
# Low-extension gate (macd.md V8): the cross lags its impulse, so entering far
# from a fast MA buys exhaustion and the stop sits where price retraces. Require
# the entry to be near EMA15. dist_ema15 = log(close/ema15) is a unitless ratio,
# so this fixed threshold (p67 of the filtered signal set) transfers across price
# levels. Of the six entry-timing ideas tried, only this one improved BOTH splits.
USE_LOW_EXTENSION = True
DIST_EMA15_MAX = 0.00224

# === Direction toggles ===
TRADE_LONGS = True
TRADE_SHORTS = True


def get_entries(df):
    n = len(df)
    entry_arr = np.full(n, np.nan)
    tp_arr = np.full(n, np.nan)
    sl_arr = np.full(n, np.nan)
    tp_type_arr = np.empty(n, dtype=object)

    # macd_hist = (macd - signal) / close, so its sign is the histogram sign;
    # a sign change from one bar to the next is a signal-line crossover.
    hist = df["macd_hist"].to_numpy()
    atr = df["raw_atr"].to_numpy()
    close = df["close"].to_numpy()
    regime = df["regime"].to_numpy()
    vol_ratio = df["vol_ratio"].to_numpy()
    vol_momentum = df["vol_momentum"].to_numpy()
    dist_ema15 = df["dist_ema15"].to_numpy()

    for i in range(1, n):
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
        entry_arr[i] = c
        sl_arr[i] = c - d * SL_ATR_MULT * atr_v
        tp_arr[i] = c + d * TP_ATR_MULT * atr_v
        tp_type_arr[i] = "atr"

    entries = pd.DataFrame({
        "entry": entry_arr,
        "tp": tp_arr,
        "sl": sl_arr,
        "tp_type": tp_type_arr,
    }, index=df.index)
    return entries
