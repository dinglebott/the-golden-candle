import numpy as np
import pandas as pd
from tqdm import tqdm

RANGE_WINDOW = 6    # candles forming the opening range (30 min at M5: 08:00–08:29 London)
TRADE_WINDOW = 12   # candles after the range where breakouts are valid (08:30–09:29 London)
RANGE_MIN = 0.10    # opening range must be >= this multiple of daily ATR
RANGE_MAX = 0.60    # opening range must be <= this multiple of daily ATR
TP_MIN_DIST = 1.0   # TP must be >= this multiple of R away (for Asian session high/low, H1 swing high/low)
TP_MAX_DIST = 6.0   # TP must be <= this multiple of R away (for Asian session high/low, H1 swing high/low)
TP_MULT = 2.0       # TP = entry ± range_size * TP_MULT (fallback)
SL_MULT = 1.0       # SL = near side of range ± range_size * SL_MULT (1.0 = opposite side)


def _select_tp_bullish(entry, risk, asian_high, range_high, range_size, trade_time,
                       swing_high_times, swing_high_vals):
    # 1. Asian session high: valid only if ORB hasn't already broken through it
    if not pd.isna(asian_high) and range_high < asian_high:
        dist = asian_high - entry
        if TP_MIN_DIST * risk <= dist <= TP_MAX_DIST * risk:
            return asian_high

    # 2. Nearest H1 swing high above entry (closest in price, formed before this trade)
    # Swing at bar j is confirmed only when bar j+1 closes, i.e. at t + 2h.
    best = None
    for t, h in zip(swing_high_times, swing_high_vals):
        if t + pd.Timedelta(hours=2) > trade_time or h <= entry:
            continue
        dist = h - entry
        if not (TP_MIN_DIST * risk <= dist <= TP_MAX_DIST * risk):
            continue
        if best is None or h < best:
            best = h
    if best is not None:
        return best

    # 3. Fallback: fixed multiple of opening range
    return entry + range_size * TP_MULT


def _select_tp_bearish(entry, risk, asian_low, range_low, range_size, trade_time,
                       swing_low_times, swing_low_vals):
    # 1. Asian session low: valid only if ORB hasn't already broken through it
    if not pd.isna(asian_low) and range_low > asian_low:
        dist = entry - asian_low
        if TP_MIN_DIST * risk <= dist <= TP_MAX_DIST * risk:
            return asian_low

    # 2. Nearest H1 swing low below entry (closest in price, formed before this trade)
    # Swing at bar j is confirmed only when bar j+1 closes, i.e. at t + 2h.
    best = None
    for t, l in zip(swing_low_times, swing_low_vals):
        if t + pd.Timedelta(hours=2) > trade_time or l >= entry:
            continue
        dist = entry - l
        if not (TP_MIN_DIST * risk <= dist <= TP_MAX_DIST * risk):
            continue
        if best is None or l > best:
            best = l
    if best is not None:
        return best

    # 3. Fallback: fixed multiple of opening range
    return entry - range_size * TP_MULT


def get_entries(df):
    entries = pd.DataFrame({"entry": np.nan, "tp": np.nan, "sl": np.nan}, index=df.index)

    dt_utc    = pd.to_datetime(df["time"], utc=True)
    dt_london = dt_utc.dt.tz_convert("Europe/London")
    london_min = dt_london.dt.hour * 60 + dt_london.dt.minute
    date = dt_london.dt.date

    range_start = 8 * 60                               # 08:00
    range_end   = range_start + RANGE_WINDOW * 5       # 08:30
    trade_end   = range_end   + TRADE_WINDOW * 5       # 09:30

    is_range = (london_min >= range_start) & (london_min < range_end)
    is_trade = (london_min >= range_end)   & (london_min < trade_end)
    is_asian = london_min < range_start    # 00:00–07:55

    # Precompute H1 swing highs and lows by resampling M5 data to H1 bars.
    # Use UTC to floor — UTC has no DST transitions so floor("h") is always unambiguous.
    h1_floor = dt_utc.dt.floor("h")
    h1_hi = df.groupby(h1_floor)["high"].max().sort_index()
    h1_lo = df.groupby(h1_floor)["low"].min().sort_index()
    h1_times  = h1_hi.index.tolist()
    h1_hi_arr = h1_hi.values
    h1_lo_arr = h1_lo.values

    swing_high_times, swing_high_vals = [], []
    swing_low_times,  swing_low_vals  = [], []
    for j in range(1, len(h1_times) - 1):
        if h1_hi_arr[j] > h1_hi_arr[j - 1] and h1_hi_arr[j] > h1_hi_arr[j + 1]:
            swing_high_times.append(h1_times[j])
            swing_high_vals.append(h1_hi_arr[j])
        if h1_lo_arr[j] < h1_lo_arr[j - 1] and h1_lo_arr[j] < h1_lo_arr[j + 1]:
            swing_low_times.append(h1_times[j])
            swing_low_vals.append(h1_lo_arr[j])

    for day in tqdm(date.unique(), desc="london_orb", unit="day"):
        day_mask = date == day
        range_idx = df.index[day_mask & is_range]
        trade_idx = df.index[day_mask & is_trade]
        asian_idx = df.index[day_mask & is_asian]

        if len(range_idx) < RANGE_WINDOW or len(trade_idx) == 0:
            continue

        range_high = df.loc[range_idx, "high"].max()
        range_low  = df.loc[range_idx, "low"].min()
        range_size = range_high - range_low

        daily_atr = df.at[range_idx[-1], "daily_atr"]
        if pd.isna(daily_atr) or daily_atr == 0:
            continue

        if not (RANGE_MIN * daily_atr <= range_size <= RANGE_MAX * daily_atr):
            continue

        asian_high = df.loc[asian_idx, "high"].max() if len(asian_idx) > 0 else np.nan
        asian_low  = df.loc[asian_idx, "low"].min()  if len(asian_idx) > 0 else np.nan

        trade_taken = False
        for i in trade_idx:
            if trade_taken:
                break

            candle = df.loc[i]
            candle_range = candle["high"] - candle["low"]
            if candle_range == 0:
                continue
            if abs(candle["close"] - candle["open"]) / candle_range < 0.5:
                continue

            h_ema20 = candle["h_ema20"]
            if pd.isna(h_ema20):
                continue

            close = candle["close"]
            trade_time = dt_utc.at[i]

            if close > range_high and close > h_ema20:
                sl = range_high - SL_MULT * range_size
                risk = close - sl
                tp = _select_tp_bullish(close, risk, asian_high, range_high,
                                        range_size, trade_time,
                                        swing_high_times, swing_high_vals)
                entries.at[i, "entry"] = close
                entries.at[i, "tp"] = tp
                entries.at[i, "sl"] = sl
                trade_taken = True

            elif close < range_low and close < h_ema20:
                sl = range_low + SL_MULT * range_size
                risk = sl - close
                tp = _select_tp_bearish(close, risk, asian_low, range_low,
                                        range_size, trade_time,
                                        swing_low_times, swing_low_vals)
                entries.at[i, "entry"] = close
                entries.at[i, "tp"] = tp
                entries.at[i, "sl"] = sl
                trade_taken = True

    return entries
