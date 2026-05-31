import numpy as np
import pandas as pd

# Pure MACD signal-line crossover baseline. No trend filter, no S/R, no
# confluence — every histogram sign change is traded in the cross direction.
# Exits are neutral ATR-based barriers so the framework can evaluate; they are
# deliberately not part of any "MACD signal".

# === Risk ===
SL_ATR_MULT = 1.5           # SL placed this many ATRs from entry
TP_RR = 1.5                 # TP at this reward:risk multiple of the SL distance

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

    for i in range(1, n):
        h = hist[i]
        h_prev = hist[i-1]
        atr_v = atr[i]
        c = close[i]
        if np.isnan(h) or np.isnan(h_prev) or np.isnan(atr_v) or atr_v <= 0:
            continue

        bull_cross = TRADE_LONGS and h_prev <= 0 and h > 0
        bear_cross = TRADE_SHORTS and h_prev >= 0 and h < 0

        if bull_cross:
            risk = SL_ATR_MULT * atr_v
            entry_arr[i] = c
            sl_arr[i] = c - risk
            tp_arr[i] = c + TP_RR * risk
            tp_type_arr[i] = "atr_rr"
        elif bear_cross:
            risk = SL_ATR_MULT * atr_v
            entry_arr[i] = c
            sl_arr[i] = c + risk
            tp_arr[i] = c - TP_RR * risk
            tp_type_arr[i] = "atr_rr"

    entries = pd.DataFrame({
        "entry": entry_arr,
        "tp": tp_arr,
        "sl": sl_arr,
        "tp_type": tp_type_arr,
    }, index=df.index)
    return entries
