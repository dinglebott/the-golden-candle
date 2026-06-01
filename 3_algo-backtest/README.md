## PROBLEM FRAMING
This is a somewhat simpler approach. Instead of machine learning, I develop hardcoded conditions for entry and rules for take profit/stop loss. Then, I run a backtest on the entire dataset and compute various metrics to measure performance.\
<br/>


## METHODOLOGY
### Backtesting
Each indicator/metric is scored against a standardised backtesting dataset (2010-2026 data). Performance is measured by computing expectancy in terms of *R*, amount risked per trade. For example, an expectancy of 0.2*R* means that if a trader risks $100 on every trade, he can expect to win $20 per trade on average.

### Labelling
The target variable is determined by triple-barrier labelling (Marcos López de Prado, 2018). Three barriers are set relative to each entry candle - stop loss, take profit, and time barrier. The time barrier is set *n* candles after the signal is given, as set in `env.json`. The SL and TP barriers are set in their respective strategy module. Labels are then computed based on which barrier is hit first.\
A "win" is only labelled if price hits the TP barrier before any other barrier. Hitting both SL and TP in the same candle results in a "loss", and hitting the time barrier results in a "time-out".\
<br/>


## FILE STRUCTURE
At the experiment root is an `env.json` for config - see below for details.\
Each strategy has its own module, for example `london_orb.py`. They all expose a `get_entries()` function for the backtesting framework to slot in.\
`backtest.py` is the standardised backtesting framework. It imports the `get_entries()` function from the specified indicator file to produce a Pandas dataframe of signals from that indicator.\
`backtest_results.log` contains results from each strategy. They are automatically logged by `backtest.py`, provided `log_results` is set to true in `env.json`. Each new log is appended on to the end of the same file.\
`visualise.py` generates PNGs of the trades that are produced by the strategies. Set your desired strategy and instrument in `env.json`, and set the module constants at the top of `visualise.py`. The candlestick charts of the selected trades are dumped in `trade_snapshots/` for your eyeballing.\
<br/>

`claude_tuning_logs/` contains records of Claude agents optimising expectancy by tuning parameters and filters.\
`TUNING_GUIDE.md` is a methodology framework for agents to reference in searching for and fine-tuning strategies.\
<br/>


## STRATEGIES
#### MACD Cross
Triggers when the standard MACD (12/26/9) crosses its signal line. Trades in the direction of the cross, with fixed ATR-multiple based TP and SL.\
**Filters**
- EMD regime = 0 (trending market)
- Volume above rolling average
- Volume ratio rising
- Distance from EMA15 below minimum threshold
**Entry trigger(s)**
- MACD crossover (histogram sign change)
**Targets**
- TP - 2&times;ATR from entry
- SL - 1&times;ATR from entry
<br/>


## USAGE
### Adding new strategies
1. Create a new module at the experiment root, e.g. `my_strategy.py`.
2. Expose a `get_entries(df)` function. `df` is the parsed H1 dataframe (OHLCV + engineered features from `data_processing.dataparser`). It must return a Pandas dataframe **with the same length and index as `df`**, containing the columns:
    - `entry` - entry price (NaN on bars without a signal)
    - `tp` - take-profit price
    - `sl` - stop-loss price
    - `tp_type` *(optional)* - string tag for the TP source (e.g. `"asian_high"`, `"swing"`, `"fallback"`); if present, results are broken down by tag in the printout
3. Direction is inferred from the prices - `tp > entry` is treated as a long, `tp < entry` as a short.
4. Put strategy-specific knobs (SL/TP multiples, lookback windows, filter thresholds) as module-level constants at the top of the file.
5. Point `env.json` at your module by setting `"strategy": "my_strategy"` (the module name, no `.py`).

### Backtesting
1. Make sure the data for your instrument exists in `../raw_data/` (use `fetch_data.py` if not).
2. Edit `env.json` at the experiment root:
    - `instrument` / `granularity` - which raw data file to load (e.g. `"EUR_USD"` / `"H1"`)
    - `strategy` - module name of the strategy to run
    - `n_value` - time-barrier length, in candles
    - `dataset` - `"train"` (first 80%), `"test"` (last 20%), or anything else for the full series
    - `log_results` - when true, appends the run's printout to `backtest_results.log`
3. From this folder, run:
    ```bash
    python backtest.py
    ```
4. The framework loads the strategy, calls `get_entries(df)`, runs triple-barrier evaluation against each signal, and prints a summary: total trades, win rate, avg win/loss R, expectancy, and breakdowns by direction (and `tp_type` if provided) plus MFE/MAE forensics.
5. Optional: `BE_TRIGGER_R` at the top of `backtest.py` moves the SL to entry once price reaches that R-multiple of favourable excursion (set to `0` to disable).

### Deploying
1. Copy the strategy module into `dist/api/`, streamlined for live inference:
    - Signature becomes `get_entries(df, ..., n_active=6) -> dict | None`. Return a single entry `{direction, entry, tp, sl, tp_type?, time}` or `None`
    - Walk the state machine over the full df but only retain entries triggered within the last `n_active` H1 bars
    - Skip arming/triggering on the trailing H1 bar if its H4 bucket is still incomplete
    - Drop `tqdm` and any other backtest-only scaffolding
    - If the strategy needs long-lookback features (e.g. daily EMA200), accept them as separate kwargs rather than resampling from a longer H1 fetch — see `trend_pullback.py` for the pattern
2. If the strategy needs more than H1 candles, add or reuse a fetcher in `dist/api/main.py` (`_fetch_h1_only`, `_fetch_h1_plus_daily`). Fetchers return `(kwargs_for_get_entries, timestamp)`.
3. Register the strategy in `STRATEGY_REGISTRY` (`dist/api/main.py`) with `module`, `n_active`, and `fetch`.
4. Add an entry to `STRATEGY_VERSIONS` in `dist/api/inference.py`. Version is purely a UI badge; bump it whenever rules or constants change.
5. Add a card config to `STRATEGY_CONFIGS` in `web_interface/js/config.js` with `id`, `label`, `endpoint`, and `renderMeta(metaEl, meta)`.
6. Redeploy `dist/` to Railway.
<br/>
