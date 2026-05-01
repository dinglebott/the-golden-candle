import pandas as pd
import numpy as np

METADATA_FEATURES = ["impulse_atr_ratio", "zone_size_atr_ratio", "candles_elapsed"]

SL_ATR_MULT       = 0.5  # SL beyond the OB zone as multiple of ATR
IMPULSE_ATR_MULT  = 1.5  # min net move over impulse window to qualify as impulse
IMPULSE_WINDOW    = 3    # candles over which net impulse move is measured
OB_EXPIRY         = 48   # candles before untouched zone expires


def detect(df: pd.DataFrame) -> list[dict]:
    opens  = df["open"].values
    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    atrs   = df["raw_atr"].values
    times  = df["time"].values
    n_rows = len(df)

    # Phase 1: identify OB candles — last opposing candle before a strong impulse
    zones = []
    for i in range(1, n_rows - IMPULSE_WINDOW):
        atr = atrs[i]
        if atr == 0:
            continue

        net_up   = closes[i + IMPULSE_WINDOW] - closes[i]
        net_down = closes[i] - closes[i + IMPULSE_WINDOW]

        # Bullish OB: bearish candle followed by strong upward impulse
        if closes[i] < opens[i] and net_up >= IMPULSE_ATR_MULT * atr:
            zones.append({
                "ob_index":    i,
                "ob_time":     times[i],
                "direction":   1,
                "ob_high":     highs[i],
                "ob_low":      lows[i],
                "impulse_size": net_up,
                "atr":         atr,
            })

        # Bearish OB: bullish candle followed by strong downward impulse
        if closes[i] > opens[i] and net_down >= IMPULSE_ATR_MULT * atr:
            zones.append({
                "ob_index":    i,
                "ob_time":     times[i],
                "direction":   -1,
                "ob_high":     highs[i],
                "ob_low":      lows[i],
                "impulse_size": net_down,
                "atr":         atr,
            })

    # Phase 2: scan forward for price returning to each zone
    instances = []
    for zone in zones:
        ob_i      = zone["ob_index"]
        ob_high   = zone["ob_high"]
        ob_low    = zone["ob_low"]
        direction = zone["direction"]

        for j in range(ob_i + IMPULSE_WINDOW + 1, min(ob_i + IMPULSE_WINDOW + 1 + OB_EXPIRY, n_rows)):
            # Invalidate: price closes beyond the full OB range
            if direction == 1  and closes[j] < ob_low:
                break
            if direction == -1 and closes[j] > ob_high:
                break

            # Touch: price enters the zone
            touched = (direction == 1 and lows[j] <= ob_high) or \
                      (direction == -1 and highs[j] >= ob_low)

            if touched:
                instances.append({
                    "index":               j,
                    "time":                times[j],
                    "ob_time":             zone["ob_time"],
                    "direction":           direction,
                    "ob_high":             ob_high,
                    "ob_low":              ob_low,
                    "fill_target":         closes[j] + atrs[j] if direction == 1 else closes[j] - atrs[j],
                    "impulse_atr_ratio":   zone["impulse_size"] / zone["atr"],
                    "zone_size_atr_ratio": (ob_high - ob_low) / zone["atr"],
                    "candles_elapsed":     j - ob_i,
                    "atr":                 atrs[j],
                })
                break  # first touch only; zone expires

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

        # SL: beyond the full OB range
        if inst["direction"] == 1:
            stop_level = inst["ob_low"] - SL_ATR_MULT * atr
        else:
            stop_level = inst["ob_high"] + SL_ATR_MULT * atr

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
