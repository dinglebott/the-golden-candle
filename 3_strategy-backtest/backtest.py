import sys
import json
import glob
import importlib
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.dataparser import parseData

ENV_PATH = Path(__file__).parent / "env.json"
DATA_DIR = Path(__file__).resolve().parents[1] / "raw_data"


def load_env():
    with open(ENV_PATH) as f:
        return json.load(f)


def find_data_file(instrument, granularity):
    pattern = str(DATA_DIR / f"{instrument}_{granularity}_*.json")
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


def print_results(trades, instrument, granularity, strategy):
    print(f"\n{'='*50}")
    print(f"  {strategy}  |  {instrument} {granularity}")
    print(f"{'='*50}")

    if not trades:
        print("  No trades found.")
        print(f"{'='*50}\n")
        return

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

    print(f"  Trades:      {total}  (W: {wins}  L: {losses}  T/O: {timeouts})")
    print(f"  Win rate:    {win_rate:.1%}")
    print(f"  Avg win R:   +{avg_win_r:.2f}R")
    print(f"  Avg loss R:  -{avg_loss_r:.2f}R")
    print(f"  Expectancy:  {expectancy:+.3f}R")
    print(f"{'='*50}\n")


def main():
    env = load_env()
    instrument = env["instrument"]
    granularity = env["granularity"]
    strategy = env.get("strategy", "")
    n_candles = env.get("n_value", 50)

    if not strategy:
        print("No strategy set in env.json")
        return

    data_path = find_data_file(instrument, granularity)
    df = parseData(data_path)
    df = df.reset_index(drop=True)

    get_entries = load_strategy(strategy)
    entries = get_entries(df)

    split = int(len(df) * 0.9)
    sections = [
        ("front 90%", df.iloc[:split], entries.iloc[:split]),
        ("back 10%", df.iloc[split:], entries.iloc[split:]),
    ]

    for label, section_df, section_entries in sections:
        # re-index to 0-based so triple barrier positional lookups stay within section
        section_df = section_df.reset_index(drop=True)
        section_entries = section_entries.reset_index(drop=True)
        trades = run_triple_barrier(section_df, section_entries, n_candles)
        print_results(trades, instrument, granularity, f"{strategy}  [{label}]")


if __name__ == "__main__":
    main()
