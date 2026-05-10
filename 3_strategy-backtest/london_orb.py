import numpy as np
import pandas as pd
from tqdm import tqdm

RANGE_WINDOW = 6    # candles forming the opening range (30 min at M5: 08:00–08:29 London)
TRADE_WINDOW = 12   # candles after the range where breakouts are valid (08:30–09:29 London)
RANGE_MIN = 0.10    # opening range must be >= this multiple of daily ATR
RANGE_MAX = 0.50    # opening range must be <= this multiple of daily ATR
TP_MULT = 2.0       # TP = entry ± range_size * TP_MULT
SL_MULT = 0.5       # SL = near side of range ± range_size * SL_MULT (1.0 = opposite side)


def get_entries(df):
    entries = pd.DataFrame({"entry": np.nan, "tp": np.nan, "sl": np.nan}, index=df.index)

    dt_london = pd.to_datetime(df["time"], utc=True).dt.tz_convert("Europe/London")
    london_min = dt_london.dt.hour * 60 + dt_london.dt.minute
    date = dt_london.dt.date

    range_start = 8 * 60                               # 08:00
    range_end   = range_start + RANGE_WINDOW * 5       # 08:30
    trade_end   = range_end   + TRADE_WINDOW * 5       # 09:30

    is_range = (london_min >= range_start) & (london_min < range_end)
    is_trade = (london_min >= range_end)   & (london_min < trade_end)

    for day in tqdm(date.unique(), desc="london_orb", unit="day"):
        day_mask  = date == day
        range_idx = df.index[day_mask & is_range]
        trade_idx = df.index[day_mask & is_trade]

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

            h_ema_trend = candle["h_ema_trend"]
            if pd.isna(h_ema_trend):
                continue

            close = candle["close"]

            if close > range_high and h_ema_trend > 0:
                entries.at[i, "entry"] = close
                entries.at[i, "tp"] = close + range_size * TP_MULT
                entries.at[i, "sl"] = range_high - SL_MULT * range_size
                trade_taken = True

            elif close < range_low and h_ema_trend < 0:
                entries.at[i, "entry"] = close
                entries.at[i, "tp"] = close - range_size * TP_MULT
                entries.at[i, "sl"] = range_low + SL_MULT * range_size
                trade_taken = True

    return entries
