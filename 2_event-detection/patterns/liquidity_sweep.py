import pandas as pd
import numpy as np

METADATA_FEATURES = ["sweep_atr_ratio", "direction"]


def detect(
    df: pd.DataFrame,
    lookback: int = 50,
    min_sweep_atr_ratio: float = 0.3,
    min_wick_body_ratio: float = 1.0,
    min_reclaim_frac: float = 0.5,
) -> list[dict]:
    """Detect liquidity sweeps with rejection-quality filters.

    min_wick_body_ratio: rejection wick (beyond the swept level's side of the
        candle) must be at least this multiple of the candle body. >=1.0 means
        the wick dominates the body — i.e. a real rejection, not a runaway bar.
    min_reclaim_frac: close must reclaim back inside the level by at least this
        fraction of the sweep size. 0.5 = close at least halfway back through.
    """
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    atrs = df["raw_atr"].values
    times = df["time"].values

    instances = []

    for i in range(lookback + 1, len(df)):
        atr = atrs[i]
        candle_open = opens[i]
        candle_high = highs[i]
        candle_low = lows[i]
        candle_close = closes[i]
        body = abs(candle_close - candle_open)

        prior_highs = highs[i - lookback: i]
        prior_lows = lows[i - lookback: i]
        swing_high = prior_highs.max()
        swing_low = prior_lows.min()

        # Bearish sweep: wick pierces above swing high but candle closes back below it
        if candle_high > swing_high and candle_close < swing_high:
            sweep_size = candle_high - swing_high
            upper_wick = candle_high - max(candle_open, candle_close)
            reclaim = swing_high - candle_close
            if (sweep_size >= min_sweep_atr_ratio * atr
                    and upper_wick >= min_wick_body_ratio * body
                    and reclaim >= min_reclaim_frac * sweep_size):
                instances.append({
                    "index": i,
                    "time": times[i],
                    "direction": -1,
                    "swept_level": swing_high,
                    "sweep_extreme": candle_high,
                    "sweep_size": sweep_size,
                    "sweep_atr_ratio": sweep_size / atr,
                    "fill_target": candle_close - atr,
                    "candle_high": candle_high,
                })

        # Bullish sweep: wick pierces below swing low but candle closes back above it
        if candle_low < swing_low and candle_close > swing_low:
            sweep_size = swing_low - candle_low
            lower_wick = min(candle_open, candle_close) - candle_low
            reclaim = candle_close - swing_low
            if (sweep_size >= min_sweep_atr_ratio * atr
                    and lower_wick >= min_wick_body_ratio * body
                    and reclaim >= min_reclaim_frac * sweep_size):
                instances.append({
                    "index": i,
                    "time": times[i],
                    "direction": 1,
                    "swept_level": swing_low,
                    "sweep_extreme": candle_low,
                    "sweep_size": sweep_size,
                    "sweep_atr_ratio": sweep_size / atr,
                    "fill_target": candle_close + atr,
                    "candle_low": candle_low,
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

        # Stop is placed beyond the sweep wick — if price revisits that extreme it invalidates the reversal
        if inst["direction"] == 1:
            stop_level = inst["sweep_extreme"] - k * atr
        else:
            stop_level = inst["sweep_extreme"] + k * atr

        label = 0
        for j in range(i + 1, min(i + 1 + n_candles, n_rows)):
            if inst["direction"] == 1:
                stopped = lows[j] <= stop_level
                filled = highs[j] >= inst["fill_target"]
            else:
                stopped = highs[j] >= stop_level
                filled = lows[j] <= inst["fill_target"]

            if stopped:
                break  # same-candle hit of both barriers counts as stopped out
            if filled:
                label = 1
                break

        labelled.append({**inst, "label": label})

    return labelled
