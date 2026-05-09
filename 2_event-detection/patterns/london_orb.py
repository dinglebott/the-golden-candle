import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

RANGE_WINDOW = 6 # candles defining the opening range (6 × 5 min = 30 min)
TP_MULT = 2 # take profit as a multiple of the range size

METADATA_FEATURES = ["range_atr_ratio"]

_LONDON_TZ = ZoneInfo("Europe/London")
_RANGE_START_MIN = 8 * 60  # 08:00 London time in minutes-since-midnight


def detect(df: pd.DataFrame) -> list[dict]:
    times = pd.to_datetime(df["time"])
    if times.dt.tz is None:
        times = times.dt.tz_localize("UTC")
    london = times.dt.tz_convert(_LONDON_TZ)
    london_minute = (london.dt.hour * 60 + london.dt.minute).values
    london_date = london.dt.date.values

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    atrs = df["raw_atr"].values
    raw_times = df["time"].values

    range_end_min = _RANGE_START_MIN + RANGE_WINDOW * 5

    instances = []

    for date in np.unique(london_date):
        day_pos = np.where(london_date == date)[0]
        day_min = london_minute[day_pos]

        range_mask = (day_min >= _RANGE_START_MIN) & (day_min < range_end_min)
        range_pos = day_pos[range_mask]

        if len(range_pos) != RANGE_WINDOW:
            continue

        range_high = highs[range_pos].max()
        range_low = lows[range_pos].min()
        range_size = range_high - range_low

        if range_size <= 0:
            continue

        post_pos = day_pos[day_min >= range_end_min]

        for i in post_pos:
            close = closes[i]
            atr = atrs[i]

            if close > range_high:
                instances.append({
                    "index": int(i),
                    "time": raw_times[i],
                    "direction": 1,
                    "range_high": float(range_high),
                    "range_low": float(range_low),
                    "range_size": float(range_size),
                    "breakout_close": float(close),
                    "tp_level": float(close + TP_MULT * range_size),
                    "sl_level": float(range_low),
                    "range_atr_ratio": float(range_size / atr) if atr > 0 else 0.0,
                })
                break
            elif close < range_low:
                instances.append({
                    "index": int(i),
                    "time": raw_times[i],
                    "direction": -1,
                    "range_high": float(range_high),
                    "range_low": float(range_low),
                    "range_size": float(range_size),
                    "breakout_close": float(close),
                    "tp_level": float(close - TP_MULT * range_size),
                    "sl_level": float(range_high),
                    "range_atr_ratio": float(range_size / atr) if atr > 0 else 0.0,
                })
                break

    return instances


def label_instances(df: pd.DataFrame, instances: list[dict], n_candles: int) -> list[dict]:
    highs = df["high"].values
    lows = df["low"].values
    n_rows = len(df)

    labelled = []
    for inst in instances:
        i = inst["index"]
        tp = inst["tp_level"]
        sl = inst["sl_level"]
        direction = inst["direction"]

        label = 0
        outcome = "expired"
        for j in range(i + 1, min(i + 1 + n_candles, n_rows)):
            if direction == 1:
                stopped = lows[j] <= sl
                filled = highs[j] >= tp
            else:
                stopped = highs[j] >= sl
                filled = lows[j] <= tp

            if stopped:
                outcome = "stopped"
                break
            if filled:
                label = 1
                outcome = "fill"
                break

        labelled.append({**inst, "label": label, "outcome": outcome})

    return labelled
