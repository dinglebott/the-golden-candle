import sys
import json
import glob
import importlib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.dataparser import parseData

ENV_PATH = Path(__file__).parent / "env.json"
DATA_DIR = Path(__file__).resolve().parents[1] / "raw_data"

BE_TRIGGER_R = 0.0  # 0 = disabled. e.g. 0.5 moves SL to entry once price reaches +0.5R favorable.


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


def run_triple_barrier(df, entries, n_candles):
    trades = []

    optional_cols = [c for c in ("tp_type",) if c in entries.columns]
    valid = entries.dropna(subset=["entry", "tp", "sl"])
    for i, row in valid.iterrows():
        entry = row["entry"]
        tp = row["tp"]
        sl = row["sl"]
        is_long = tp > entry
        risk = abs(entry - sl)
        if risk == 0:
            continue

        forward = df.iloc[i + 1 : i + 1 + n_candles]

        outcome = None
        mfe = 0.0  # in price units, in trade direction
        mae = 0.0
        bars_to_outcome = 0
        effective_sl = sl
        be_triggered = False
        for k, (_, candle) in enumerate(forward.iterrows(), start=1):
            if is_long:
                fav = candle["high"] - entry
                adv = entry - candle["low"]
            else:
                fav = entry - candle["low"]
                adv = candle["high"] - entry
            if fav > mfe:
                mfe = fav
            if adv > mae:
                mae = adv

            # Pessimistic intra-bar ordering: assume adverse leg first.
            if is_long:
                if candle["low"] <= effective_sl:
                    outcome = "be" if be_triggered else "loss"
                    bars_to_outcome = k
                    break
                if candle["high"] >= tp:
                    outcome = "win"
                    bars_to_outcome = k
                    break
            else:
                if candle["high"] >= effective_sl:
                    outcome = "be" if be_triggered else "loss"
                    bars_to_outcome = k
                    break
                if candle["low"] <= tp:
                    outcome = "win"
                    bars_to_outcome = k
                    break

            # Move SL to entry once MFE crosses BE_TRIGGER_R (after barrier checks for this bar).
            if BE_TRIGGER_R > 0 and not be_triggered and fav >= BE_TRIGGER_R * risk:
                effective_sl = entry
                be_triggered = True

        if outcome is None:
            if len(forward) > 0:
                exit_price = forward.iloc[-1]["close"]
                bars_to_outcome = len(forward)
            else:
                exit_price = entry
            pnl = (exit_price - entry) if is_long else (entry - exit_price)
            r = pnl / risk
            outcome = "timeout"
        elif outcome == "win":
            r = abs(tp - entry) / risk
        elif outcome == "be":
            r = 0.0
        else:
            r = -1.0

        trade = {
            "time": df.at[i, "time"],
            "entry": entry,
            "tp": tp,
            "sl": sl,
            "direction": "long" if is_long else "short",
            "outcome": outcome,
            "r_multiple": r,
            "tp_planned_r": abs(tp - entry) / risk,
            "mfe_r": mfe / risk,
            "mae_r": mae / risk,
            "bars_to_outcome": bars_to_outcome,
        }
        for col in optional_cols:
            trade[col] = row[col]
        trades.append(trade)

    return trades


def _block_stats(df_subset, label, lines):
    n = len(df_subset)
    if n == 0:
        lines.append(f"  [{label}]  no trades")
        return
    wins = (df_subset["outcome"] == "win").sum()
    losses = (df_subset["outcome"] == "loss").sum()
    timeouts = (df_subset["outcome"] == "timeout").sum()
    wr = wins / n
    exp = df_subset["r_multiple"].mean()
    avg_win = df_subset.loc[df_subset["r_multiple"] > 0, "r_multiple"].mean() if wins > 0 else 0.0
    avg_loss = df_subset.loc[df_subset["r_multiple"] < 0, "r_multiple"].abs().mean() if losses > 0 else 0.0
    lines.append(f"  [{label}]  n={n}  W/L/TO={wins}/{losses}/{timeouts}  "
                 f"WR={wr:.1%}  +{avg_win:.2f}R/-{avg_loss:.2f}R  exp={exp:+.3f}R")


def format_results(trades, instrument, granularity, strategy, n_value, dataset="train"):
    lines = []
    lines.append(f"  {strategy}  |  {instrument} {granularity}  |  [{dataset}]")
    lines.append(f"{'='*60}")

    if not trades:
        lines.append("  No trades found.")
        lines.append(f"{'='*60}\n")
        return "\n".join(lines)

    df = pd.DataFrame(trades)
    total = len(df)
    wins = (df["outcome"] == "win").sum()
    losses = (df["outcome"] == "loss").sum()
    timeouts = (df["outcome"] == "timeout").sum()
    bes = (df["outcome"] == "be").sum()

    win_rate = wins / total
    r_multiples = df["r_multiple"]
    expectancy = r_multiples.mean()

    winning = r_multiples[r_multiples > 0]
    losing = r_multiples[r_multiples < 0].abs()
    avg_win_r = winning.mean() if len(winning) > 0 else 0.0
    avg_loss_r = losing.mean() if len(losing) > 0 else 0.0

    lines.append(f"  n-value:     {n_value}")
    be_seg = f"  BE: {bes}" if bes > 0 else ""
    lines.append(f"  Trades:      {total}  (W: {wins}  L: {losses}  T/O: {timeouts}{be_seg})")
    lines.append(f"  Win rate:    {win_rate:.1%}")
    lines.append(f"  Avg win R:   +{avg_win_r:.2f}R")
    lines.append(f"  Avg loss R:  -{avg_loss_r:.2f}R")
    lines.append(f"  Expectancy:  {expectancy:+.3f}R")
    lines.append(f"{'-'*60}")

    # By direction
    if "direction" in df.columns:
        for d in ("long", "short"):
            _block_stats(df[df["direction"] == d], f"dir={d}", lines)
        lines.append(f"{'-'*60}")

    # By TP type
    if "tp_type" in df.columns:
        for tp_type in sorted(df["tp_type"].dropna().unique()):
            _block_stats(df[df["tp_type"] == tp_type], f"tp={tp_type}", lines)
        lines.append(f"{'-'*60}")

    # MFE/MAE forensics
    if "mfe_r" in df.columns:
        losers = df[df["r_multiple"] < 0]
        winners = df[df["r_multiple"] > 0]
        timeouts_df = df[df["outcome"] == "timeout"]
        lines.append(f"  Losers MFE:  median={losers['mfe_r'].median():.2f}R  "
                     f"mean={losers['mfe_r'].mean():.2f}R  "
                     f"%>=0.5R={(losers['mfe_r']>=0.5).mean():.1%}  "
                     f"%>=0.8R={(losers['mfe_r']>=0.8).mean():.1%}  "
                     f"%<0.2R={(losers['mfe_r']<0.2).mean():.1%}")
        lines.append(f"  Winners MAE: median={winners['mae_r'].median():.2f}R  "
                     f"mean={winners['mae_r'].mean():.2f}R  "
                     f"%>=0.5R={(winners['mae_r']>=0.5).mean():.1%}")
        if len(timeouts_df) > 0:
            lines.append(f"  Timeouts:    avg_mfe={timeouts_df['mfe_r'].mean():.2f}R  "
                         f"avg_r={timeouts_df['r_multiple'].mean():+.2f}R")
        lines.append(f"  Avg bars to outcome: {df['bars_to_outcome'].mean():.1f}")
        lines.append(f"  Planned TP R: median={df['tp_planned_r'].median():.2f}  "
                     f"mean={df['tp_planned_r'].mean():.2f}")

    lines.append(f"{'='*60}\n")
    return "\n".join(lines)


def main():
    env = load_env()
    instrument = env["instrument"]
    granularity = env["granularity"]
    strategy = env.get("strategy", "")
    n_candles = env.get("n_value", 50)
    log_results = env.get("log_results", False)

    if not strategy:
        print("No strategy set in env.json")
        return

    dataset = env.get("dataset", "train")

    data_path = find_data_file(instrument, granularity)
    df = parseData(data_path)
    df = df.reset_index(drop=True)

    n = len(df)
    train_end = int(n * 0.8)
    if dataset == "train":
        df = df.iloc[:train_end].reset_index(drop=True)
    elif dataset == "test":
        df = df.iloc[train_end:].reset_index(drop=True)

    get_entries = load_strategy(strategy)
    entries = get_entries(df)

    trades = run_triple_barrier(df, entries, n_candles)
    output = format_results(trades, instrument, granularity, strategy, n_candles, dataset)
    print(output)

    if log_results:
        log_path = Path(__file__).parent / "backtest_results.log"
        with open(log_path, "a") as log_file:
            log_file.write(f"{'='*50}\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(output + "\n")


if __name__ == "__main__":
    main()
