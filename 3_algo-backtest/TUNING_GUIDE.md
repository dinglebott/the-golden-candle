# Strategy Tuning & Validation Guide

How to refine a `3_algo-backtest` strategy from a raw idea into a validated edge.
Worked example: `macd.py` (see `claude_tuning_logs/macd.md` for the full run this
process produced). Follow this workflow and log findings the same way.

## Orientation

- A strategy is a module exposing `get_entries(df) -> DataFrame` indexed by `df.index`
  with columns `entry`, `tp`, `sl`, and optional `tp_type`. Direction is implied:
  `tp > entry` is long, else short.
- `backtest.py` loads `env.json`, slices train/test, runs triple-barrier, prints metrics.
- `env.json`: `strategy`, `n_value` (time barrier), `dataset` (`train` = first 80%,
  `test` = last 20%, else full). Set `log_results: true` to append runs to
  `backtest_results.log`.
- Features come from `data_processing/dataparser.parseData()` (~40 cols: returns, ATR,
  Bollinger, EMAs/ultSmoother, RSI, MACD, ADX, Williams %R, volume, EMD regime,
  directional/extension features). Use existing columns before engineering new ones.
- **Keep strategies direction-symmetric** unless there is a real geometric reason not
  to. Disabling a whole direction because it underperformed *on train* is overfitting,
  not a filter.

## The harness pattern (do this — it is what makes the search cheap)

In this framework each signal is evaluated independently (forward window, no position
overlap). Exploit that:

- Write a scratch harness (e.g. `_<strategy>_lab.py`) that calls `parseData` **once**,
  generates all candidate signals, and runs triple-barrier.
- **If exits are fixed**, run triple-barrier once over all signals, attach each signal's
  feature values at its entry bar, and evaluate any filter as a boolean **pandas mask**
  on the cached trade set. Hundreds of filter combos cost nothing.
- **If a change alters entry/tp/sl** (e.g. stop width), you must re-run triple-barrier
  per scheme — masking no longer suffices.
- Cache the base trade set (`.pkl`) so reruns are instant.
- Mirror `backtest.py`'s triple-barrier semantics exactly (pessimistic intra-bar
  ordering, timeout handling, any BE logic) so harness numbers match the real pipeline.
  **Always re-verify the final config through `backtest.py` before trusting it.**
- Prefix scratch files with `_` and delete them when done; the `.md` log is the record.

## Step 1 — Baseline

Establish the expectancy of the **core idea alone**, no filters or confluences
(macd: pure signal-line crosses). Use neutral, fixed ATR exits so the framework can
evaluate — exits are not part of the idea yet. This is the number every later change
must beat. Read the diagnostics (win rate, per-direction expectancy, losers' MFE,
timeout MFE) to understand the failure mode before adding anything.

## Step 2 — Breadth-first filter search (train only)

Goal: **find filter/confluence combinations that work**, not perfect thresholds.

- Sweep many candidate filters **in isolation** first, each as a direction-aware mask,
  one coarse threshold each. Categorize by what gap they fill (trend, momentum,
  volatility/regime, location, volume, session). Prefer one filter per category over
  several that say the same thing.
- Keep thresholds coarse here. A filter that only works at one hand-tuned threshold is
  fragile; you want effects robust enough to show up at a round number.
- Identify the filters with a real, direction-symmetric lift and decent trade count,
  then test their **combinations**. Watch trade count — a combo that works but leaves
  <~150 trades is hard to trust.
- Don't over-invest: **≤3 tuning attempts per combination**, then move on.

## Step 3 — Fine-tune + test (the discipline that matters most)

Take the few most promising combinations and tune their exact parameters (thresholds,
SL/TP, time barrier). Then validate on the held-out **test** split.

- **Only changes that improve the TEST split count.** Train gains that don't transfer
  are overfitting — the single most common outcome (in the macd run, ADX/EMA/H4-trend
  filters, wider stops, histogram-magnitude floors, higher volume thresholds, and BE
  all looked fine on train and died on test).
- **Pick robust regions, not the train maximum.** When sweeping a grid, choose a cell
  that improves *both* splits and whose neighbors are also positive — never a lone
  spike that happens to peak on train.
- **Derive thresholds from train, apply unchanged to test.** A clean way to sweep
  "strictness" without unit-guessing: set a threshold at a percentile of the train
  signal distribution. For deployment, hardcode that train-derived value; prefer
  unitless/normalized features (log-ratios, ATR-relative) so a fixed threshold transfers
  across price levels and regimes.
- **Mind sample size and significance.** ~exp/(std/√n) is a rough t-stat; an edge within
  ~1 SE of zero is unproven regardless of how positive it looks. Low trade counts
  (e.g. <~100 on test) cannot confirm an edge — flag it rather than trust it.
- **Re-check interacting parameters.** If an entry change shifts where entry sits
  relative to the stop, re-verify the exit; don't assume earlier-optimal exits still hold.
- **Trust aggregate stats over anecdotes.** Chart snapshots (`visualise.py`, set `COUNT`
  for random samples across the span) are great for spotting *mechanisms* and failure
  modes, but a handful of charts will mislead on magnitude and direction. Every
  snapshot-derived hypothesis must be confirmed statistically before it's adopted.
- The test split degrades from a pristine hold-out the more you peek at it. For a final
  read, validate on a **fresh instrument** (e.g. another pair) as genuine out-of-sample.

## Logging

Keep a running `claude_tuning_logs/<strategy>.md`, versioned (V0 baseline, V1…), each
section a short rationale + a compact metrics table. **Record refuted ideas too** —
knowing what failed and why (usually train-only overfitting) is as valuable as the
wins, and stops the next agent re-running dead ends. End with the final config,
its train/test numbers, and the still-untested leads.

## One-line summary

Baseline the bare idea → breadth-first hunt for filter *combinations* that work on
train → fine-tune the finalists and keep only what survives the **test** split, picking
robust regions over train peaks, and confirm through `backtest.py`.
