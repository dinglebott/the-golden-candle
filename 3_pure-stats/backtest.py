import sys
import json
import importlib
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data_processing.dataparser import parseData

with open(Path(__file__).parent / "env.json") as f:
    env = json.load(f)

INSTRUMENT = env["instrument"]
GRANULARITY = env["granularity"]
INDICATOR = env["indicator"]
K_TP = env["indicators"][INDICATOR]["k_tp"]
K_SL = env["indicators"][INDICATOR]["k_sl"]
N = env["indicators"][INDICATOR]["n"]


def triple_barrier_label(df: pd.DataFrame, signal_idx: int, k_tp: float, k_sl: float, n: int) -> int:
    """
    Returns 1 if TP hit first, -1 if SL hit first, 0 if time barrier expires.
    Signal direction determines which barrier is TP and which is SL.
    """
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    atr = df["raw_atr"].to_numpy()

    direction = df["signal"].iloc[signal_idx]
    tp_dist = k_tp * atr[signal_idx]
    sl_dist = k_sl * atr[signal_idx]
    if direction == 1:
        upper = close[signal_idx] + tp_dist
        lower = close[signal_idx] - sl_dist
    else:
        upper = close[signal_idx] + sl_dist
        lower = close[signal_idx] - tp_dist

    for j in range(signal_idx + 1, min(signal_idx + 1 + n, len(df))):
        hit_up = high[j] >= upper
        hit_dn = low[j] <= lower
        if hit_up and hit_dn:
            # ambiguous same-candle hit - call it a loss
            return -1
        if hit_up:
            return 1 if direction == 1 else -1
        if hit_dn:
            return 1 if direction == -1 else -1

    return 0  # time barrier


def simpleLabel(df: pd.DataFrame, signal_idx: int, k: float, n: int):
    """
    Labels based on final distance from close price in terms of ATR.
    """
    close = df["close"].to_numpy()
    atr = df["raw_atr"].to_numpy()

    direction = df["signal"].iloc[signal_idx]
    threshold = k * atr[signal_idx]
    if close[signal_idx + n] > close[signal_idx] + threshold:
        return 1 if direction == 1 else -1
    elif close[signal_idx + n] < close[signal_idx] - threshold:
        return -1 if direction == 1 else 1
    else:
        return 0


def run_backtest(df: pd.DataFrame, signals: pd.Series, k_tp: float, k_sl: float, n: int) -> pd.DataFrame:
    df = df.copy()
    signals = signals.reindex(df.index).fillna(0).astype(int)
    df["signal"] = signals

    active = df[df["signal"] != 0].copy()
    results = []

    for ts, row in active.iterrows():
        idx = df.index.get_loc(ts)
        if idx + n >= len(df):
            continue  # not enough bars remaining for a full window
        outcome = triple_barrier_label(df, idx, k_tp, k_sl, n)
        results.append({
            "timestamp": ts,
            "signal": row["signal"],
            "outcome": outcome,
            "close": row["close"],
        })

    return pd.DataFrame(results).set_index("timestamp") if results else pd.DataFrame()


def print_summary(results: pd.DataFrame):
    if results.empty:
        print("No signals found.")
        return

    total = len(results)
    wins   = (results["outcome"] == 1).sum()
    losses = (results["outcome"] == -1).sum()
    neutral = (results["outcome"] == 0).sum()

    win_rate = wins / total * 100
    loss_rate = losses / total * 100
    neutral_rate = neutral / total * 100

    longs = results[results["signal"] == 1]
    shorts = results[results["signal"] == -1]

    print(f"\n{'='*40}")
    print(f"Indicator: {INDICATOR}")
    print(f"Instrument: {INSTRUMENT} {GRANULARITY}")
    print(f"Barriers: k_tp={K_TP}, k_sl={K_SL}, n={N}")
    print(f"{'='*40}")
    print(f"Total signals: {total}")
    print(f"  Long signals: {len(longs)}")
    print(f"  Short signals: {len(shorts)}")
    print(f"Wins    : {wins}  ({win_rate:.1f}%)")
    print(f"Losses  : {losses}  ({loss_rate:.1f}%)")
    print(f"Neutral : {neutral}  ({neutral_rate:.1f}%)")

    if len(longs) > 0:
        l_wins = (longs["outcome"] == 1).sum()
        print(f"\nLong win rate : {l_wins}/{len(longs)} ({l_wins/len(longs)*100:.1f}%)")
    if len(shorts) > 0:
        s_wins = (shorts["outcome"] == 1).sum()
        print(f"Short win rate: {s_wins}/{len(shorts)} ({s_wins/len(shorts)*100:.1f}%)")
    print(f"{'='*40}\n")


if __name__ == "__main__":
    print(f"Loading data for {INSTRUMENT} {GRANULARITY}...")
    filepath = Path(__file__).parent.parent / f"raw_data/{INSTRUMENT}_{GRANULARITY}_2005-01-01_2026-04-01.json"
    df = parseData(filepath)

    print(f"Computing signals from {INDICATOR}...")
    module = importlib.import_module(INDICATOR)
    signals = module.get_signals(df)

    results = run_backtest(df, signals, K_TP, K_SL, N)
    print_summary(results)
