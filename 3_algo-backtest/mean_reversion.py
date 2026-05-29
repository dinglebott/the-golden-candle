import numpy as np
import pandas as pd

# === TP midline ===
# Ehlers' matched lowpass: SMA over 2 x the EMD bandpass period (dataparser uses
# EMD_PERIOD=20 H1 bars) — has a transfer zero at the cycle frequency, so it
# rejects the bp swings and leaves the trend baseline the cycle reverts to.
MIDLINE_SMA_PERIOD = 40

# === Risk ===
SL_ATR_MULT = 1.0           # SL placed this many ATRs beyond the entry candle's far extreme
MIN_RR = 1.0                # skip trades whose reward:risk to the midline TP is below this

# === Direction toggles ===
TRADE_LONGS = True
TRADE_SHORTS = True


def get_entries(df):
    n = len(df)
    entry_arr = np.full(n, np.nan)
    tp_arr = np.full(n, np.nan)
    sl_arr = np.full(n, np.nan)
    tp_type_arr = np.empty(n, dtype=object)

    # EMD components and regime are pre-computed in dataparser on this timeframe
    bp = df["bp"].to_numpy()
    avg_peak = df["avg_peak"].to_numpy()
    avg_valley = df["avg_valley"].to_numpy()
    regime = df["regime"].to_numpy()
    atr = df["raw_atr"].to_numpy()
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()

    hl2 = (df["high"] + df["low"]) / 2
    midline = hl2.rolling(MIDLINE_SMA_PERIOD).mean().to_numpy()

    # Per-direction lock: blocks re-arming on the same cycle extreme until bp
    # crosses to the opposite half (half-cycle has elapsed)
    bull_locked = False
    bear_locked = False

    for i in range(1, n):
        bp_v = bp[i]
        bp_prev = bp[i-1]
        if bull_locked and bp_v > 0:
            bull_locked = False
        if bear_locked and bp_v < 0:
            bear_locked = False

        if regime[i] != 1:
            continue
        c = close[i]
        mid = midline[i]
        atr_v = atr[i]
        peak_v = avg_peak[i]
        valley_v = avg_valley[i]
        if (np.isnan(peak_v) or np.isnan(valley_v) or np.isnan(atr_v)
                or np.isnan(mid) or atr_v <= 0):
            continue

        # Fade exhaustion: bp at extreme on this bar and turning back toward zero
        bull_setup = (TRADE_LONGS and not bull_locked
                      and bp_v <= valley_v
                      and bp_v > bp_prev)
        bear_setup = (TRADE_SHORTS and not bear_locked
                      and bp_v >= peak_v
                      and bp_v < bp_prev)

        if bull_setup and bear_setup:
            continue
        if bull_setup:
            sl = low[i] - SL_ATR_MULT * atr_v
            if sl < c and mid > c and (mid - c) / (c - sl) >= MIN_RR:
                entry_arr[i] = c
                tp_arr[i] = mid
                sl_arr[i] = sl
                tp_type_arr[i] = "emd_trend"
            bull_locked = True
        elif bear_setup:
            sl = high[i] + SL_ATR_MULT * atr_v
            if sl > c and mid < c and (c - mid) / (sl - c) >= MIN_RR:
                entry_arr[i] = c
                tp_arr[i] = mid
                sl_arr[i] = sl
                tp_type_arr[i] = "emd_trend"
            bear_locked = True

    entries = pd.DataFrame({
        "entry": entry_arr,
        "tp": tp_arr,
        "sl": sl_arr,
        "tp_type": tp_type_arr,
    }, index=df.index)
    return entries
