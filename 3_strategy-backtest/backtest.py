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
        for _, candle in forward.iterrows():
            if is_long:
                if candle["high"] >= tp:
                    outcome = "win"
                    break
                if candle["low"] <= sl:
                    outcome = "loss"
                    break
            else:  # short
                if candle["low"] <= tp:
                    outcome = "win"
                    break
                if candle["high"] >= sl:
                    outcome = "loss"
                    break

        if outcome is None:
            # time barrier: exit at last candle's close
            if len(forward) > 0:
                exit_price = forward.iloc[-1]["close"]
            else:
                exit_price = entry
            pnl = (exit_price - entry) if is_long else (entry - exit_price)
            r = pnl / risk
            outcome = "timeout"
        elif outcome == "win":
            r = abs(tp - entry) / risk
        else:
            r = -1.0

        trades.append({
            "time": df.at[i, "time"],
            "entry": entry,
            "tp": tp,
            "sl": sl,
            "outcome": outcome,
            "r_multiple": r,
        })

    return trades


def format_results(trades, instrument, granularity, strategy, n_value, dataset="train"):
    lines = []
    lines.append(f"  {strategy}  |  {instrument} {granularity}  |  [{dataset}]")
    lines.append(f"{'='*50}")

    if not trades:
        lines.append("  No trades found.")
        lines.append(f"{'='*50}\n")
        return "\n".join(lines)

    df = pd.DataFrame(trades)
    total = len(df)
    wins = (df["outcome"] == "win").sum()
    losses = (df["outcome"] == "loss").sum()
    timeouts = (df["outcome"] == "timeout").sum()

    win_rate = wins / total
    r_multiples = df["r_multiple"]
    expectancy = r_multiples.mean()

    winning = r_multiples[r_multiples > 0]
    losing = r_multiples[r_multiples < 0].abs()
    avg_win_r = winning.mean() if len(winning) > 0 else 0.0
    avg_loss_r = losing.mean() if len(losing) > 0 else 0.0

    lines.append(f"  n-value:     {n_value}")
    lines.append(f"  Trades:      {total}  (W: {wins}  L: {losses}  T/O: {timeouts})")
    lines.append(f"  Win rate:    {win_rate:.1%}")
    lines.append(f"  Avg win R:   +{avg_win_r:.2f}R")
    lines.append(f"  Avg loss R:  -{avg_loss_r:.2f}R")
    lines.append(f"  Expectancy:  {expectancy:+.3f}R")
    lines.append(f"{'='*50}\n")
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
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    if dataset == "train":
        df = df.iloc[:train_end].reset_index(drop=True)
    elif dataset == "val":
        df = df.iloc[train_end:val_end].reset_index(drop=True)
    elif dataset == "test":
        df = df.iloc[val_end:].reset_index(drop=True)

    get_entries = load_strategy(strategy)
    entries = get_entries(df)

    trades = run_triple_barrier(df, entries, n_candles)
    output = format_results(trades, instrument, granularity, strategy, n_candles, dataset)
    print(output)

    if log_results:
        log_path = Path(__file__).parent / "backtest_results.log"
        with open(log_path, "a") as log_file:
            log_file.write(f"{'='*50}\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(output + "\n\n")


if __name__ == "__main__":
    main()
