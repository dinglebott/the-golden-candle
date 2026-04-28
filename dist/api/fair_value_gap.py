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
                    "candle_high": highs[i],
                    "atr": atrs[i],
                    "fill_target": (bull_gap_low + bull_gap_high) / 2,
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
                    "candle_low": lows[i],
                    "atr": atrs[i],
                    "fill_target": (bear_gap_low + bear_gap_high) / 2,
                })

    return instances
