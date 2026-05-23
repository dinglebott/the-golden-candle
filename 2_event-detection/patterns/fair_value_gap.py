import pandas as pd
import numpy as np

from patterns._trade_sim import simulate_trade

METADATA_FEATURES = ["gap_atr_ratio"]

SL_ATR_MULT = 1.5  # SL beyond the FVG candle's extreme as multiple of ATR


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
                    "fill_target": (bull_gap_low + bull_gap_high) / 2,
                    "candle_high": highs[i],
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
                    "fill_target": (bear_gap_low + bear_gap_high) / 2,
                    "candle_low": lows[i],
                })

    return instances


def label_instances(df: pd.DataFrame, instances: list[dict], n_candles: int) -> list[dict]:
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    atrs = df["raw_atr"].values

    labelled = []
    for inst in instances:
        i = inst["index"]
        atr = atrs[i]
        entry = closes[i]
        tp = inst["fill_target"]

        # A bullish FVG is faded (trader shorts), bearish FVG is bought.
        if inst["direction"] == 1:
            sl = highs[i] + SL_ATR_MULT * atr
            is_long = False
        else:
            sl = lows[i] - SL_ATR_MULT * atr
            is_long = True

        outcome, r = simulate_trade(highs, lows, closes, i, n_candles, entry, tp, sl, is_long)

        labelled.append({
            **inst,
            "label": 1 if outcome == "fill" else 0,
            "outcome": outcome,
            "r_multiple": float(r),
            "entry": float(entry),
            "tp": float(tp),
            "sl": float(sl),
        })

    return labelled
