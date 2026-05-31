import sys
import json
import glob
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.dataparser import parseData

# ---- Config ----
GRANULARITIES = ["H1", "H4"]  # timeframes to render per trade
LOOKBACK = 100                       # candles before trigger, per timeframe
LOOKAHEAD = 50                      # candles after trigger, per timeframe
COUNT = 10                           # number of random trades to render from across the span
SEED = 64                            # RNG seed for reproducible sampling (set None for fresh draw)

ENV_PATH = Path(__file__).parent / "env.json"
DATA_DIR = Path(__file__).resolve().parents[1] / "raw_data"
OUT_DIR = Path(__file__).parent / "trade_snapshots"

# Floor-frequency string per granularity, matches the strategy resampling.
FREQ_MAP = {
    "H1": "1h",
    "H2": "2h",
    "H4": "4h",
    "H6": "6h",
    "H8": "8h",
    "H12": "12h",
    "D": "D",
    "W": "W",
}


def load_env():
    with open(ENV_PATH) as f:
        return json.load(f)


def find_data_file(instrument, granularity):
    pattern = str(DATA_DIR / f"{instrument}_{granularity}_2010-*.json")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No data file found for {instrument} {granularity} in {DATA_DIR}")
    return matches[0]


def load_strategy(strategy_name):
    sys.path.insert(0, str(Path(__file__).parent))
    mod = importlib.import_module(strategy_name)
    return mod.get_entries


def resample_ohlc(df_h1, freq):
    if freq == "1h":
        out = df_h1[["open", "high", "low", "close"]].copy()
    else:
        floor = df_h1.index.floor(freq)
        out = pd.DataFrame({
            "open":  df_h1.groupby(floor)["open"].first(),
            "high":  df_h1.groupby(floor)["high"].max(),
            "low":   df_h1.groupby(floor)["low"].min(),
            "close": df_h1.groupby(floor)["close"].last(),
        })
    out.columns = ["Open", "High", "Low", "Close"]
    return out


def render_trade(trade_idx, row, df_h1, tf_frames, strategy, instrument, out_path):
    entry_time = df_h1.index[trade_idx]
    entry = float(row["entry"])
    tp = float(row["tp"])
    sl = float(row["sl"])
    is_long = tp > entry
    direction = "long" if is_long else "short"
    tp_type = row["tp_type"] if "tp_type" in row.index and pd.notna(row["tp_type"]) else ""

    n_panels = len(GRANULARITIES)
    fig, axes = plt.subplots(n_panels, 1, figsize=(13, 3.8 * n_panels), squeeze=False)
    axes = axes[:, 0]

    for ax, gran in zip(axes, GRANULARITIES):
        freq = FREQ_MAP[gran]
        tf_df = tf_frames[gran]
        trigger_ts = entry_time if freq == "1h" else entry_time.floor(freq)

        if trigger_ts not in tf_df.index:
            ax.set_title(f"{gran}: trigger bar not found", fontsize=9)
            continue

        trigger_pos = tf_df.index.get_loc(trigger_ts)
        start = max(0, trigger_pos - LOOKBACK)
        end = min(len(tf_df), trigger_pos + LOOKAHEAD + 1)
        window = tf_df.iloc[start:end]

        # Trigger candle marker (arrow below for long, above for short).
        marker_series = pd.Series(np.nan, index=window.index)
        price_range = window["High"].max() - window["Low"].min()
        offset = price_range * 0.04
        if is_long:
            marker_y = window.loc[trigger_ts, "Low"] - offset
            marker = "^"
        else:
            marker_y = window.loc[trigger_ts, "High"] + offset
            marker = "v"
        marker_series.loc[trigger_ts] = marker_y

        ap = mpf.make_addplot(
            marker_series, type="scatter", marker=marker, markersize=90, color="blue", ax=ax
        )

        mpf.plot(
            window,
            type="candle",
            style="charles",
            ax=ax,
            addplot=ap,
            datetime_format="%Y-%m-%d" if gran == "D" else "%m-%d %H:%M",
            xrotation=0,
        )

        # Entry / TP / SL horizontal lines, drawn on every panel.
        ax.axhline(entry, color="#1f77b4", linestyle="--", linewidth=0.8)
        ax.axhline(tp, color="#2ca02c", linestyle="--", linewidth=0.8)
        ax.axhline(sl, color="#d62728", linestyle="--", linewidth=0.8)
        ax.set_ylabel(gran, fontsize=10)

    title = (f"{strategy}  |  {instrument}  |  trade #{trade_idx}  |  {direction.upper()}"
             f"{f'  |  tp={tp_type}' if tp_type else ''}"
             f"  |  {entry_time.strftime('%Y-%m-%d %H:%M UTC')}")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    env = load_env()
    instrument = env["instrument"]
    granularity = env["granularity"]
    strategy = env.get("strategy", "")
    dataset = env.get("dataset", "train")

    if not strategy:
        print("No strategy set in env.json")
        return

    data_path = find_data_file(instrument, granularity)
    df = parseData(data_path)
    df = df.reset_index(drop=True)

    # Same train/test split as backtest.py so trade indexes line up.
    n = len(df)
    train_end = int(n * 0.8)
    if dataset == "train":
        df = df.iloc[:train_end].reset_index(drop=True)
    elif dataset == "test":
        df = df.iloc[train_end:].reset_index(drop=True)

    get_entries = load_strategy(strategy)
    entries = get_entries(df)

    valid = entries.dropna(subset=["entry", "tp", "sl"])
    if len(valid) == 0:
        print("No valid trades to render.")
        return
    n_sample = min(COUNT, len(valid))
    selected = valid.sample(n=n_sample, random_state=SEED).sort_index()

    df_h1 = df[["open", "high", "low", "close"]].copy()
    df_h1.index = pd.to_datetime(df["time"], utc=True)

    tf_frames = {gran: resample_ohlc(df_h1, FREQ_MAP[gran]) for gran in GRANULARITIES}

    out_dir = OUT_DIR / strategy
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Rendering {len(selected)} random trades (total valid {len(valid)}, seed {SEED}) to {out_dir}")
    for trade_idx, row in selected.iterrows():
        entry_time = pd.to_datetime(df.at[trade_idx, "time"], utc=True)
        direction = "long" if row["tp"] > row["entry"] else "short"
        fname = f"{trade_idx:05d}_{direction}_{entry_time.strftime('%Y%m%d_%H%M')}.png"
        render_trade(trade_idx, row, df_h1, tf_frames, strategy, instrument, out_dir / fname)
        print(f"  saved {fname}")


if __name__ == "__main__":
    main()
