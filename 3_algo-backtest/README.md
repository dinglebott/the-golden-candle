## PROBLEM FRAMING
This is a much simpler approach to forex. Instead of machine learning, I develop and implement custom strategies. Each strategy has hardcoded conditions for entry and rules for take profit/stop loss. Then, I run a backtest on the entire dataset and compute various metrics to measure performance.\
<br/>


## METHODOLOGY
### Backtesting
Each indicator/metric is scored against a standardised backtesting dataset (2005-2026 data). Performance is measured by computing expectancy in terms of *R*, amount risked per trade. For example, an expectancy of 0.2*R* means that if a trader risks $100 on every trade, he can expect to win $20 per trade on average.

### Labelling
The target variable is determined by triple-barrier labelling (Marcos López de Prado, 2018). Three barriers are set relative to each entry candle - stop loss, take profit, and time barrier. The time barrier is set *n* candles after the signal is given, as set in `env.json`. The SL and TP barriers are set in their respective strategy module. Labels are then computed based on which barrier is hit first.\
A "win" is only labelled if price hits the TP barrier before any other barrier. Hitting both SL and TP in the same candle results in a "loss", and hitting the time barrier results in a "time-out".\
<br/>


## FILE STRUCTURE
At the experiment root is an `env.json` for config - see below for details.\
Each strategy has its own module, for example `london_orb.py`. They all expose a `get_entries()` function for the backtesting framework to slot in.\
`backtest.py` is the standardised backtesting framework. It imports the `get_entries()` function from the specified indicator file to produce a Pandas dataframe of signals from that indicator.\
`backtest_results.log` contains results from each strategy. They are automatically logged by `backtest.py`, provided `log_results` is set to true in `env.json`. Each new log is appended on to the end of the same file.\
<br/>


## STRATEGIES
#### London Opening Range Breakout (ORB)
Defines the opening range as 08:00 - 08:30 London time, accounting for daylight savings. Breakout is defined as the first candle that closes outside this range (above for bullish, below for bearish). The trade is taken in the direction of the breakout, with TP and SL defined in terms of opening range size.\
**Filters**
- Breakout must be within 60min of range end (08:30 - 09:30)
- H1 EMA-20 must be above/below H1 EMA-50 for bullish/bearish breakout respectively
- Range size must be between 0.15&times; and 0.5&times; daily ATR
- Breakout candle must be >50% body (no long wicks)
**Entry trigger(s)**
- Breakout candle
**Targets**
- TP - 2&times;OR beyond entry price
- SL - Midpoint of range

#### Trend Pullback and Continuation
Searches H4 candles for a trend pullback, enters trade in the direction of the trend, expecting a trend continuation. Testing revealed that downtrends perform worse, so the final implementation only scans for uptrends.\
**Filters**
- ADX >= 25, +DI > -DI
- RSI remains above 40 for longs (below 60 for shorts)
- Daily EMA50 > EMA200
- Price > daily EMA50
**Entry trigger(s)**
- Break of structure on H1 candle
**Targets**
- TP - Search for previous H4 swing highs (excluding the impulse leg right before the pullback), else fall back to 2.0&times;R
- SL - 1.5&times;ATR below the pullback low
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
4. Put strategy-specific knobs (SL/TP multiples, lookback windows, filter thresholds) as module-level constants at the top of the file. See `london_orb.py` and `trend_pullback.py` for the convention.
5. Point `env.json` at your module by setting `"strategy": "my_strategy"` (the module name, no `.py`).

### Backtesting
1. Make sure the H1 data for your instrument exists in `../raw_data/` (run `python fetch_data.py` from the repo root if not).
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

<br/>
