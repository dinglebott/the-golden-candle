"""
Grid-search fast/slow ultSmoother period pairs as a crossover signal.

Pipeline per (fast, slow) pair:
  1. Compute fast & slow ultSmoother on close.
  2. Find crossovers (fast crossing slow). Bullish cross -> long, bearish -> short.
  3. Triple-barrier label each signal: symmetric +/- K * ATR barriers, N-candle time barrier.
  4. Score by hit rate, expectancy (in R), and signal count.
  5. Compare against a random-entry / random-direction baseline.
  6. Chronological train/test split to flag overfit pairs.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.dataparser import parseData, ultSmoother

# ---------- config ----------
INSTRUMENT = "EUR_USD"
GRANULARITY = "H1"
DATA_FILE = Path(__file__).resolve().parents[1] / "raw_data" / f"{INSTRUMENT}_{GRANULARITY}_2005-01-01_2026-04-01.json"

K = 2.5 # ATR multiple for symmetric upper/lower barriers (1:1 R:R)
N = 10 # time-barrier length in candles

FAST_PERIODS = [4, 6, 8, 10, 12, 14, 16, 20]
SLOW_PERIODS = [10, 14, 18, 22, 26, 30, 40, 50, 70, 100]

TRAIN_FRAC = 0.7    # chronological split
MIN_SIGNALS = 100    # minimum crossovers in a split for the score to be reported
RANDOM_SEED = 42
# ----------------------------


def find_crossovers(fast: np.ndarray, slow: np.ndarray):
    """Return (indices, directions) where direction is +1 (bullish) or -1 (bearish)."""
    diff = fast - slow
    sign = np.sign(diff)
    flip = (sign[1:] != sign[:-1]) & (sign[1:] != 0) & (sign[:-1] != 0)
    idx = np.where(flip)[0] + 1  # candle whose close confirms the cross
    direction = sign[idx].astype(np.int8)
    return idx, direction


def triple_barrier_outcomes(entry_idx, direction, close, high, low, raw_atr, k, n):
    """For each entry, return +1 (TP hit), -1 (SL hit), or 0 (time expiry / ambiguous)."""
    total = len(close)
    outcomes = np.zeros(len(entry_idx), dtype=np.int8)
    for s, (i, d) in enumerate(zip(entry_idx, direction)):
        if i + n >= total:
            outcomes[s] = 0
            continue
        barrier = k * raw_atr[i]
        if d > 0:  # long
            tp, sl = close[i] + barrier, close[i] - barrier
        else:      # short
            tp, sl = close[i] - barrier, close[i] + barrier
        result = 0
        for j in range(i + 1, i + 1 + n):
            hit_tp = (high[j] >= tp) if d > 0 else (low[j] <= tp)
            hit_sl = (low[j] <= sl) if d > 0 else (high[j] >= sl)
            if hit_tp and hit_sl:
                result = 0  # ambiguous same-candle hit
                break
            if hit_tp:
                result = 1
                break
            if hit_sl:
                result = -1
                break
        outcomes[s] = result
    return outcomes


def score(outcomes):
    """Hit rate among decisive outcomes, expectancy in R (1:1), and counts."""
    n_total = len(outcomes)
    if n_total == 0:
        return {"n": 0, "hit_rate": np.nan, "expectancy_R": np.nan, "n_decisive": 0}
    wins   = int((outcomes ==  1).sum())
    losses = int((outcomes == -1).sum())
    decisive = wins + losses
    hit_rate = wins / decisive if decisive else np.nan
    expectancy = (wins - losses) / n_total  # time-expiry counted as 0 R
    return {"n": n_total, "hit_rate": hit_rate, "expectancy_R": expectancy, "n_decisive": decisive}


def run_split(df, fast_p, slow_p, k, n):
    close = df["close"].to_numpy()
    high  = df["high"].to_numpy()
    low   = df["low"].to_numpy()
    atr   = df["raw_atr"].to_numpy()

    fast = ultSmoother(df["close"], fast_p)
    slow = ultSmoother(df["close"], slow_p)

    # warmup: ignore early bars where the recurrence hasn't settled
    warmup = max(fast_p, slow_p) * 3
    valid_mask = np.ones(len(fast), dtype=bool)
    valid_mask[:warmup] = False

    idx, direction = find_crossovers(fast, slow)
    keep = valid_mask[idx]
    idx, direction = idx[keep], direction[keep]

    outcomes = triple_barrier_outcomes(idx, direction, close, high, low, atr, k, n)
    return idx, direction, outcomes


def random_baseline(df, n_signals, k, n, rng):
    close = df["close"].to_numpy()
    high  = df["high"].to_numpy()
    low   = df["low"].to_numpy()
    atr   = df["raw_atr"].to_numpy()
    total = len(close)
    if n_signals == 0 or total < 100:
        return np.array([], dtype=np.int8)
    idx = rng.integers(low=50, high=total - n - 1, size=n_signals)
    direction = rng.choice([-1, 1], size=n_signals).astype(np.int8)
    return triple_barrier_outcomes(idx, direction, close, high, low, atr, k, n)


def main():
    print(f"Loading {DATA_FILE.name} ...")
    df = parseData(DATA_FILE)
    df = df.dropna(subset=["raw_atr"]).reset_index(drop=True)
    n_total = len(df)
    split_idx = int(n_total * TRAIN_FRAC)
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_test  = df.iloc[split_idx:].reset_index(drop=True)
    print(f"Loaded {n_total} candles  |  train={len(df_train)}  test={len(df_test)}")
    print(f"Triple-barrier params: k={K} (symmetric, 1:1 R:R), n={N} candles\n")

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for fast_p in FAST_PERIODS:
        for slow_p in SLOW_PERIODS:
            if fast_p >= slow_p:
                continue
            _, _, out_tr = run_split(df_train, fast_p, slow_p, K, N)
            _, _, out_te = run_split(df_test,  fast_p, slow_p, K, N)
            s_tr = score(out_tr)
            s_te = score(out_te)
            rows.append({
                "fast": fast_p, "slow": slow_p,
                "n_tr": s_tr["n"], "hit_tr": s_tr["hit_rate"], "E[R]_tr": s_tr["expectancy_R"],
                "n_te": s_te["n"], "hit_te": s_te["hit_rate"], "E[R]_te": s_te["expectancy_R"],
            })

    results = pd.DataFrame(rows)
    qual = results[(results["n_tr"] >= MIN_SIGNALS) & (results["n_te"] >= MIN_SIGNALS)]

    if not qual.empty:
        med_tr = int(qual["n_tr"].median())
        med_te = int(qual["n_te"].median())
        base_tr = score(random_baseline(df_train, med_tr, K, N, rng))
        base_te = score(random_baseline(df_test,  med_te, K, N, rng))
        print(f"Random baseline (train, n={med_tr}):  hit={base_tr['hit_rate']:.3f}  E[R]={base_tr['expectancy_R']:+.4f}")
        print(f"Random baseline (test,  n={med_te}):  hit={base_te['hit_rate']:.3f}  E[R]={base_te['expectancy_R']:+.4f}\n")

    qual_sorted = qual.sort_values("E[R]_te", ascending=False)
    print("Pairs meeting MIN_SIGNALS in both splits, sorted by out-of-sample expectancy:\n")
    with pd.option_context("display.float_format", lambda x: f"{x:+.4f}"):
        print(qual_sorted.to_string(index=False))

    out_path = Path(__file__).with_suffix(".csv")
    results.to_csv(out_path, index=False)
    print(f"\nFull results written to {out_path.name}")


if __name__ == "__main__":
    main()
