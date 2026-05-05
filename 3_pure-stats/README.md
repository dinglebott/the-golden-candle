## PROBLEM FRAMING
This is a much simpler approach to forex. Instead of machine learning, I develop and implement custom indicators or metrics, based purely on traditional statistics. These are intended as an aid for trading decisions, by giving signals and revealing market biases, rather than hard predictions like my previous ML models. Tried-and-tested examples include things like EMA crossovers, RSI divergence, etc.\
<br/>


## METHODOLOGY
### Backtesting
Each indicator/metric is scored against a standardised backtesting dataset (2005-2026 data). Performance is measured by computing win rates, based on whether the price resolved in the signalled direction for every signal given by the algorithm across the dataset.

### Labelling
The target variable is determined by triple-barrier labelling (Marcos López de Prado, 2018). Given parameters *k* and *n*, three barriers are set relative to the candle of interest - stop loss, take profit, and time barrier. The SL barrier is set *k* &times; ATR below/above the reference point (for bullish/bearish signals respectively). The TP barrier is set according to the expected resolution of the setup. The time barrier is set *n* candles after the signal is given. Labels are then computed based on which barrier is hit first.\
A "win" is only labelled if price hits the TP barrier before any other barrier. Hitting both SL and TP in the same candle results in a "loss", and hitting the time barrier results in a "time-out".

### Feature engineering
<br/>


## FILE STRUCTURE
At the experiment root is an `env.json` for config - see below for details.\
Each indicator has its own module, for example `pctR_trend_exhaustion.py`. They all expose a `get_signals()` function for the backtesting framework to slot in.\
`backtest.py` is the standardised backtesting framework. It imports the `get_signals()` function from the specified indicator file to produce a Pandas series of signals from that indicator.\
<br/>


## INDICATORS
**%R Trend Exhaustion (upslidedown)**\
This is a Python implementation of an indicator developed by [upslidedown](https://www.tradingview.com/u/upslidedown/) for TradingView. It watches for the confluence of two standard Williams %R indicators, a fast-period and slow-period. When both indicators are in the overbought/oversold zone, the setup period begins. The actual signal is given when at least one of them leaves the zone (back to normal range).\
<br/>


## USAGE
### Adding new indicators


### Backtesting


### Deploying

<br/>
