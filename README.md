## ABOUT PROJECT
**Goal -** Use machine learning to predict price movements in the forex market\
**Models -** XGBoost, CNN-LSTM hybrid, PatchTST\
This is a collection of training and testing pipelines for various model architectures (listed above). The problem is framed as two binary classifications. Firstly, will the price move significantly over the forecasting horizon? Secondly, given that it moves, does it move up or down?\
The model architectures used are explained below:
- **XGBoost** uses a gradient-boosted decision tree framework. It builds decision trees sequentially, with each tree improving on the previous one via a gradient descent algorithm. As a traditional machine learning algorithm, it is faster than its modern deep learning counterparts, and performs well on structured, tabular data.
- **CNN-LSTM** is a hybrid of Convolutional Neural Networks and Long Short-Term Memory networks, which are a variant of Recurrent Neural Networks. The CNN extracts local patterns and reduce noise. The output of the CNN is passed to the LSTM layers, which capture temporal patterns and long-term relationships. This makes them good for time series data like forex markets.
- **PatchTST** (Patch Time Series Transformer) breaks time series into smaller patches, which serve as input tokens. The transformer backbone uses a combination of encoder and decoder blocks to understand the data and generate predicted data. The encoder prominently uses self-attention heads, which allow it to understand cross-token context. Finally, the output goes through an output head which produces forecasted values, or classification probabilities in this project.
<br/>

## PROJECT STRUCTURE
### Root
`data_processing/` - Functions to fetch and process data, shared across all pipelines\
`model_training/` - Full pipelines for training and testing, sorted by model architecture\
`raw_data/` - Historical OHLCV data, pulled from the OANDA API\
`env.json` - Config variables, including shared configs and pipeline-specific configs (see below for details)
### Pipeline
`select_features.py` - Feature selection tool\
`tune_params.py` - Hyperparameter tuning (Optuna)\
`results/` - Output location for feature selection, hyperparameter tuning, model test metrics\
`model_configs/` - Manually curated feature sets and hyperparameters, sorted by version\
`train_model.py` - Trains and tests a new model\
`models/` - Trained models, named by version number as set in configs\
`use_model.py` - Run live inference with a trained model
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

## METHODOLOGY
### Labelling
The target variable is determined by triple-barrier labelling (Marcos López de Prado, 2018). Given parameters *k* and *n*, three barriers are set relative to the close price **C** of the latest candle - upper, lower, and time barrier. The upper and lower barriers are set *k* &times; ATR above and below **C**, and the time barrier is set *n* candles after the latest one. Labels are then computed based on which barrier is hit first.
### Feature Engineering
<br/>

## USAGE
### Training
### Deployment
<br/>