import numpy as np
import pandas as pd

# === ER regime gate (H4) ===
ER_PERIOD = 14              # Kaufman efficiency ratio lookback, in H4 bars
ER_THRESHOLD = 0.4          # ER strictly below this = ranging, allow trades

# === Bollinger bands (H4) ===
BB_PERIOD = 20              # SMA / stdev window, in H4 bars
BB_STD = 2.0                # band width in standard deviations

# === ATR (shared) ===
ATR_PERIOD = 14

# === Risk ===
SL_ATR_MULT = 1.0           # SL placed this many H4 ATRs beyond the entry candle's far extreme

# === Direction toggles ===
TRADE_LONGS = True
TRADE_SHORTS = True


def _wilder(s, period):
    return s.ewm(alpha=1/period, adjust=False).mean()


def _atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return _wilder(tr, period)


def _efficiency_ratio(close, period):
    direction = (close - close.shift(period)).abs()
    volatility = close.diff().abs().rolling(period).sum()
    er = direction / volatility
    return er.replace([np.inf, -np.inf], np.nan)


def get_entries(df):
    n = len(df)
    entry_arr = np.full(n, np.nan)
    tp_arr = np.full(n, np.nan)
    sl_arr = np.full(n, np.nan)
    tp_type_arr = np.empty(n, dtype=object)

    dt_utc = pd.to_datetime(df["time"], utc=True)
    h4_floor = dt_utc.dt.floor("4h")
    h4_floor_np = h4_floor.to_numpy()

    # === H4 reconstruction from H1 ===
    h4 = pd.DataFrame({
        "open":  df.groupby(h4_floor)["open"].first(),
        "high":  df.groupby(h4_floor)["high"].max(),
        "low":   df.groupby(h4_floor)["low"].min(),
        "close": df.groupby(h4_floor)["close"].last(),
    }).sort_index()

    h4_atr = _atr(h4["high"], h4["low"], h4["close"], ATR_PERIOD)
    h4_mid = h4["close"].rolling(BB_PERIOD).mean()
    h4_std = h4["close"].rolling(BB_PERIOD).std(ddof=0)
    h4_upper = h4_mid + BB_STD * h4_std
    h4_lower = h4_mid - BB_STD * h4_std
    h4_er = _efficiency_ratio(h4["close"], ER_PERIOD)

    h4_close_np = h4["close"].to_numpy()
    h4_high_np = h4["high"].to_numpy()
    h4_low_np = h4["low"].to_numpy()
    h4_atr_np = h4_atr.to_numpy()
    h4_mid_np = h4_mid.to_numpy()
    h4_upper_np = h4_upper.to_numpy()
    h4_lower_np = h4_lower.to_numpy()
    h4_er_np = h4_er.to_numpy()
    h4_idx_np = h4.index.to_numpy()
    h4_count = len(h4)

    # Map each H4 bucket to its final H1 index (the bar that closes the bucket)
    same_as_next = h4_floor_np[:-1] == h4_floor_np[1:]
    h4_close_idx_h1 = np.where(~same_as_next)[0]
    h4_close_idx_h1 = np.append(h4_close_idx_h1, n - 1)
    h4_pos_at_close = np.array([
        np.searchsorted(h4_idx_np, h4_floor_np[i]) for i in h4_close_idx_h1
    ])

    # === Trigger: first H4 close outside a Bollinger band while ER says ranging ===
    prev_outside_upper = False
    prev_outside_lower = False
    for ev_i in range(len(h4_close_idx_h1)):
        h1_close_i = h4_close_idx_h1[ev_i]
        h4_pos = h4_pos_at_close[ev_i]
        if h4_pos < 0 or h4_pos >= h4_count:
            continue

        c = h4_close_np[h4_pos]
        hi = h4_high_np[h4_pos]
        lo = h4_low_np[h4_pos]
        upper = h4_upper_np[h4_pos]
        lower = h4_lower_np[h4_pos]
        mid = h4_mid_np[h4_pos]
        atr_v = h4_atr_np[h4_pos]
        er_v = h4_er_np[h4_pos]

        outside_upper = (not np.isnan(upper)) and c > upper
        outside_lower = (not np.isnan(lower)) and c < lower

        if (np.isnan(mid) or np.isnan(atr_v) or np.isnan(er_v)
                or atr_v <= 0):
            prev_outside_upper = outside_upper
            prev_outside_lower = outside_lower
            continue

        ranging = er_v < ER_THRESHOLD

        if ranging and TRADE_SHORTS and outside_upper and not prev_outside_upper:
            sl = hi + SL_ATR_MULT * atr_v
            if sl > c and mid < c:
                entry_arr[h1_close_i] = c
                tp_arr[h1_close_i] = mid
                sl_arr[h1_close_i] = sl
                tp_type_arr[h1_close_i] = "bb_midline"
        elif ranging and TRADE_LONGS and outside_lower and not prev_outside_lower:
            sl = lo - SL_ATR_MULT * atr_v
            if sl < c and mid > c:
                entry_arr[h1_close_i] = c
                tp_arr[h1_close_i] = mid
                sl_arr[h1_close_i] = sl
                tp_type_arr[h1_close_i] = "bb_midline"

        prev_outside_upper = outside_upper
        prev_outside_lower = outside_lower

    entries = pd.DataFrame({
        "entry": entry_arr,
        "tp": tp_arr,
        "sl": sl_arr,
        "tp_type": tp_type_arr,
    }, index=df.index)
    return entries
