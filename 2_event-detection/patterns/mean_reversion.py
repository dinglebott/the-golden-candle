import pandas as pd
import numpy as np

from patterns._trade_sim import simulate_trade

METADATA_FEATURES = ["breakout_atr_ratio"]

KC_PERIOD    = 20    # EMA period for the Keltner channel midline
KC_ATR_MULT  = 1.5   # channel half-width as a multiple of ATR
SL_ATR_MULT  = 1.0   # SL beyond the entry candle's far extreme, in ATR


def detect(df: pd.DataFrame) -> list[dict]:
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    atrs   = df["raw_atr"].values
    times  = df["time"].values
    n_rows = len(df)

    midline = df["close"].ewm(span=KC_PERIOD, adjust=False).mean().to_numpy()
    upper   = midline + KC_ATR_MULT * atrs
    lower   = midline - KC_ATR_MULT * atrs

    instances = []

    # Fire only on the bar that first closes outside the channel; re-arm once a
    # later bar closes back inside, so one excursion produces a single entry.
    armed = True
    for i in range(n_rows):
        atr = atrs[i]
        if atr <= 0 or np.isnan(atr) or np.isnan(midline[i]):
            continue

        above = closes[i] > upper[i]
        below = closes[i] < lower[i]

        if armed and above:
            # Overextended up — fade short. Bearish geometry (mirrored to long).
            instances.append({
                "index":             i,
                "time":              times[i],
                "direction":         -1,
                "midline":           midline[i],
                "breakout_atr_ratio": (closes[i] - upper[i]) / atr,
            })
        elif armed and below:
            # Overextended down — fade long. Canonical bullish geometry.
            instances.append({
                "index":             i,
                "time":              times[i],
                "direction":         1,
                "midline":           midline[i],
                "breakout_atr_ratio": (lower[i] - closes[i]) / atr,
            })

        armed = not (above or below)

    return instances


def label_instances(df: pd.DataFrame, instances: list[dict], n_candles: int) -> list[dict]:
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    atrs   = df["raw_atr"].values

    labelled = []
    for inst in instances:
        i     = inst["index"]
        atr   = atrs[i]
        entry = closes[i]
        tp    = inst["midline"]

        # TP is the channel midline; SL sits beyond the entry candle's far extreme.
        if inst["direction"] == 1:
            sl = lows[i] - SL_ATR_MULT * atr
            is_long = True
        else:
            sl = highs[i] + SL_ATR_MULT * atr
            is_long = False

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
