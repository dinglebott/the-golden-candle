import pandas as pd
import numpy as np

METADATA_FEATURES = ["gap_atr_ratio", "direction"]


def detect(df: pd.DataFrame, min_gap_atr_ratio: float = 0.3) -> list[dict]:
    highs = df["high"].values
    lows = df["low"].values
    atrs = df["raw_atr"].values
    times = df["time"].values

    instances = []

    for i in range(2, len(df)):
        atr = atrs[i]

        # Bullish FVG: gap between candle[i-2] high and candle[i] low
        bull_gap_low = highs[i - 2]
        bull_gap_high = lows[i]
        if bull_gap_high > bull_gap_low:
            gap_size = bull_gap_high - bull_gap_low
            if gap_size >= min_gap_atr_ratio * atr:
                instances.append({
                    "index": i,
                    "time": times[i],
                    "direction": 1,
                    "gap_low": bull_gap_low,
                    "gap_high": bull_gap_high,
                    "gap_size": gap_size,
                    "gap_atr_ratio": gap_size / atr,
                    # price must reach here (from above) to enter the gap
                    "fill_target": bull_gap_high,
                })

        # Bearish FVG: gap between candle[i] high and candle[i-2] low
        bear_gap_low = highs[i]
        bear_gap_high = lows[i - 2]
        if bear_gap_high > bear_gap_low:
            gap_size = bear_gap_high - bear_gap_low
            if gap_size >= min_gap_atr_ratio * atr:
                instances.append({
                    "index": i,
                    "time": times[i],
                    "direction": -1,
                    "gap_low": bear_gap_low,
                    "gap_high": bear_gap_high,
                    "gap_size": gap_size,
                    "gap_atr_ratio": gap_size / atr,
                    # price must reach here (from below) to enter the gap
                    "fill_target": bear_gap_low,
                })

    return instances


def label_instances(df: pd.DataFrame, instances: list[dict], n_candles: int, k: float) -> list[dict]:
    highs = df["high"].values
    lows = df["low"].values
    atrs = df["raw_atr"].values
    n_rows = len(df)

    labelled = []
    for inst in instances:
        i = inst["index"]
        atr = atrs[i]

        if inst["direction"] == 1:
            stop_level = highs[i] + k * atr
        else:
            stop_level = lows[i] - k * atr

        label = 0
        for j in range(i + 1, min(i + 1 + n_candles, n_rows)):
            stopped = highs[j] >= stop_level if inst["direction"] == 1 else lows[j] <= stop_level
            filled = lows[j] <= inst["fill_target"] if inst["direction"] == 1 else highs[j] >= inst["fill_target"]
            if stopped: # same candle hit both = stopped out
                break
            if filled:
                label = 1
                break

        labelled.append({**inst, "label": label})

    return labelled
