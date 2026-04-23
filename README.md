## ABOUT PROJECT
**Goal -** Use machine learning to predict price movements in the forex market\
**Models -** XGBoost, CNN-LSTM hybrid, Temporal Fusion Transformer\
This is a collection of training and testing pipelines for various model architectures (listed above). The problem is framed as two binary classifications. Firstly, will the price move significantly over the forecasting horizon? Secondly, given that it moves, does it move up or down?
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