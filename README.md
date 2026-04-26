## ABOUT PROJECT
This is a collection of training and testing pipelines for various model architectures (listed above). I attempt various approaches to hopefully gain an edge over the forex market. The main focus of this research is the EUR/USD pair, although most of the pipelines can be easily configured to target other pairs. See below for the various problem framings and models I experimented with.\
Selected models are deployed to the repository website [The Golden Candle](https://dinglebott.github.io/the-golden-candle). The deployment history is at the bottom of this README.\
<br/>

**IMPORTANT:** Please use the website responsibly, as each inference/site reload incurs server costs that come from my own pocket D:\
<br/>


## PROJECT STRUCTURE
### Root
`data_processing/` - Functions to fetch and process data, shared across all pipelines\
`raw_data/` - Historical OHLCV data, pulled from the OANDA API\
Experiments - Each experiment has its own folder, formatted as `<experiment-no.>_<experiment-name>`. See below for a summary of the experiments. Each one has its own README.md within its folder.

### Deployment
`dist/` - Server-side Docker container
- `api/` - Pulls live data, runs inference, returns response
- `artifacts/` - Models and model configs, sorted by binary type

`web_interface/` - Standard website front-end
- `config.js` - API URL, model registry
- `api.js` - Requests from back-end
- `ui.js` - Draws to DOM
- `main.js` - Entry point
<br/>


## EXPERIMENTS
**#1 Double binary**\
This experiment frames the problem as two binary classifications. Firstly, does the price move in a direction (up/down), or does it remain flat? Secondly, given that it moves in a direction, which direction did it move in?\
<br/>

**#2 Event detection**\
This experiment filters out specific patterns using hardcoded criteria, then predicts the probability of the expected resolution for each instance.\
</br>


## MODELS
- **XGBoost** uses a gradient-boosted decision tree framework. It builds decision trees sequentially, with each tree improving on the previous one via a gradient descent algorithm. As a traditional machine learning algorithm, it is faster than its modern deep learning counterparts, and performs well on structured, tabular data.
- **CNN-LSTM** is a hybrid of Convolutional Neural Networks and Long Short-Term Memory networks, which are a variant of Recurrent Neural Networks. The CNN extracts local patterns and reduce noise. The output of the CNN is passed to the LSTM layers, which capture temporal patterns and long-term relationships. This makes them good for time series data like forex markets.
- **PatchTST** (Patch Time Series Transformer) breaks time series into smaller patches, which serve as input tokens. The transformer backbone uses a series of encoder blocks to understand the data and generate predicted data. The encoder prominently uses self-attention heads, which allow it to understand cross-token context. Finally, the output goes through an output head which produces forecasted values (or classification probabilities).
<br/>


## DEPLOYMENT CHANGELOG
### 1.3
- Added FVG-CNN-LSTM v2
- Removed old XGBoost models
### 1.2
- Added first pattern detection - FVG-XGBoost v1
### 1.1
- Added Gate-PatchTST v1
### 1.0
- Added first model - Gate-XGBoost v1
<br/>