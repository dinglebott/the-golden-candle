import pandas as pd
import numpy as np
from bisect import bisect_left, bisect_right

METADATA_FEATURES = ["wick_atr_ratio", "level_touches", "body_distance_atr"]

MIN_WICK_ATR        = 0.5   # min wick extension beyond level as multiple of ATR
SL_ATR_MULT         = 0.5   # SL beyond wick extreme as multiple of ATR
TP_ATR_MULT         = 1.0   # TP from candle close (opposite direction of breakout) as multiple of ATR

SWING_WINDOW        = 3     # candles each side required to qualify as swing high/low
LOOKBACK            = 150   # candles scanned backwards when searching for established levels
LEVEL_TOLERANCE_ATR = 0.5   # cluster swings whose prices lie within this many ATR of each other
MIN_TOUCHES         = 3     # min clustered swings for a level to count as established


def _find_swings(highs: np.ndarray, lows: np.ndarray, window: int):
    n = len(highs)
    sh, sl = [], []
    for i in range(window, n - window):
        seg_h = highs[i - window:i + window + 1]
        seg_l = lows[i - window:i + window + 1]
        if highs[i] == seg_h.max():
            sh.append(i)
        if lows[i] == seg_l.min():
            sl.append(i)
    return sh, sl


def _cluster_levels(prices: list[float], tolerance: float, min_touches: int):
    """Greedy 1-D clustering. Returns [(mean_price, touch_count), ...]."""
    if not prices:
        return []
    sorted_prices = sorted(prices)
    levels = []
    cluster = [sorted_prices[0]]
    for p in sorted_prices[1:]:
        if p - cluster[0] <= tolerance:
            cluster.append(p)
        else:
            if len(cluster) >= min_touches:
                levels.append((sum(cluster) / len(cluster), len(cluster)))
            cluster = [p]
    if len(cluster) >= min_touches:
        levels.append((sum(cluster) / len(cluster), len(cluster)))
    return levels


def detect(df: pd.DataFrame) -> list[dict]:
    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    atrs   = df["raw_atr"].values
    times  = df["time"].values
    n_rows = len(df)

    sh_idx, sl_idx = _find_swings(highs, lows, SWING_WINDOW)
    sh_idx_arr = np.array(sh_idx, dtype=int)
    sl_idx_arr = np.array(sl_idx, dtype=int)

    instances = []

    for i in range(LOOKBACK, n_rows):
        atr = atrs[i]
        if atr == 0:
            continue

        # Swings must be confirmed before candle i (i.e. swing idx + SWING_WINDOW < i)
        window_start = i - LOOKBACK
        window_end   = i - SWING_WINDOW

        sh_lo = bisect_left(sh_idx_arr, window_start) if len(sh_idx_arr) else 0
        sh_hi = bisect_right(sh_idx_arr, window_end - 1) if len(sh_idx_arr) else 0
        sl_lo = bisect_left(sl_idx_arr, window_start) if len(sl_idx_arr) else 0
        sl_hi = bisect_right(sl_idx_arr, window_end - 1) if len(sl_idx_arr) else 0

        sh_prices = [highs[k] for k in sh_idx_arr[sh_lo:sh_hi]]
        sl_prices = [lows[k]  for k in sl_idx_arr[sl_lo:sl_hi]]

        tol = LEVEL_TOLERANCE_ATR * atr
        resistance_levels = _cluster_levels(sh_prices, tol, MIN_TOUCHES)
        support_levels    = _cluster_levels(sl_prices, tol, MIN_TOUCHES)

        body_high = max(opens[i], closes[i])
        body_low  = min(opens[i], closes[i])
        wick_high = highs[i]
        wick_low  = lows[i]

        # Bearish failed breakout: wick pierces resistance, body closes back below
        # Pick the closest qualifying resistance to the candle body
        best = None
        for level, touches in resistance_levels:
            wick_beyond = wick_high - level
            if wick_beyond >= MIN_WICK_ATR * atr and body_high <= level:
                if best is None or level < best[0]:
                    best = (level, touches, wick_beyond)
        if best is not None:
            level, touches, wick_beyond = best
            instances.append({
                "index":             i,
                "time":              times[i],
                "direction":         -1,
                "level":             level,
                "wick_extreme":      wick_high,
                "wick_atr_ratio":    wick_beyond / atr,
                "level_touches":     touches,
                "body_distance_atr": (level - closes[i]) / atr,
                "fill_target":       closes[i] - TP_ATR_MULT * atr,
            })
            continue  # one signal per candle

        # Bullish failed breakout: wick pierces support, body closes back above
        best = None
        for level, touches in support_levels:
            wick_beyond = level - wick_low
            if wick_beyond >= MIN_WICK_ATR * atr and body_low >= level:
                if best is None or level > best[0]:
                    best = (level, touches, wick_beyond)
        if best is not None:
            level, touches, wick_beyond = best
            instances.append({
                "index":             i,
                "time":              times[i],
                "direction":         1,
                "level":             level,
                "wick_extreme":      wick_low,
                "wick_atr_ratio":    wick_beyond / atr,
                "level_touches":     touches,
                "body_distance_atr": (closes[i] - level) / atr,
                "fill_target":       closes[i] + TP_ATR_MULT * atr,
            })

    return instances


def label_instances(df: pd.DataFrame, instances: list[dict], n_candles: int) -> list[dict]:
    highs  = df["high"].values
    lows   = df["low"].values
    atrs   = df["raw_atr"].values
    n_rows = len(df)

    labelled = []
    for inst in instances:
        i   = inst["index"]
        atr = atrs[i]

        if inst["direction"] == 1:
            stop_level = inst["wick_extreme"] - SL_ATR_MULT * atr
        else:
            stop_level = inst["wick_extreme"] + SL_ATR_MULT * atr

        label = 0
        for j in range(i + 1, min(i + 1 + n_candles, n_rows)):
            if inst["direction"] == 1:
                stopped = lows[j]  <= stop_level
                filled  = highs[j] >= inst["fill_target"]
            else:
                stopped = highs[j] >= stop_level
                filled  = lows[j]  <= inst["fill_target"]

            if stopped:
                break
            if filled:
                label = 1
                break

        labelled.append({**inst, "label": label})

    return labelled
