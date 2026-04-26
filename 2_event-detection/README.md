## PROBLEM FRAMING
Here I take a 2-step approach. First, use hardcoded numerical criteria to filter out specific candlestick patterns from the dataset. Second, train a model to predict the probability that the setup resolves as expected. Thus the model only outputs a prediction when its specific pattern is detected; otherwise it has no output. It gives sparser but (hopefully) higher quality signals, instead of coin-tossing on every single candle.\
<br/>


## METHODOLOGY
### Labelling
The target variable is determined by triple-barrier labelling (Marcos López de Prado, 2018). Given parameters *k* and *n*, three barriers are set relative to some reference point (depending on pattern) - stop loss, take profit, and time barrier. The SL barrier is set *k* &times; ATR below/above the reference point (for bullish/bearish setups respectively). The TP barrier is set according to the expected resolution of the setup. The time barrier is set *n* candles after the pattern emerges. Labels are then computed based on which barrier is hit first.\
A "fill" is only labelled if price hits the TP barrier before any other barrier. Hitting both SL and TP in the same candle results in a "no fill", as does hitting the time barrier.

### Feature Engineering
<br/>


## FILE STRUCTURE
At the experiment root is an `env.json` for config - see below for details.\
`patterns/` contains an event detection script for each pattern - see below for the list of patterns implemented so far.\
Each model architecture has its own folder: XGBoost, CNN-LSTM, PatchTST. Within each folder:\
`select_features.py` - Feature selection tool\
`tune_params.py` - Hyperparameter tuning (Optuna)\
`results/` - Output location for feature selection, hyperparameter tuning, model test metrics\
`model_configs/` - Manually curated feature sets and hyperparameters, sorted by version\
`train_model.py` - Trains and tests a new model\
`models/` - Trained models, named by version number as set in configs\
`use_model.py` - Run live inference with a trained model\
<br/>


## PATTERNS
**Fair value gap (FVG)**\
3-candle pattern, defined as the gap between the extremes (highs/lows) of the 1st and 3rd candles. Indicates a large buying/selling impulse which causes a region of unfilled orders. The expected resolution is that the market returns to touch the gap before continuing its trend.\
TP - Nearer end of the gap (upper end for bullish FVG, lower end for bearish FVG)\
SL - 1.5 &times; ATR beyond the extreme of the 3rd candle\
<br/>


## USAGE
### Training
First, set the correct config variables in `env.json`. Below are the variables that may need some clarification:
- `k_value` - Coefficient to multiply ATR(14) by, determines stop loss
- `n_value` - Length of time barrier from pattern emergence, expressed in candles
- `train_split`, `val_split` - Determines split ratio of the dataset (test set is whatever is left over)
- `pattern` - Determines the training task for the models
- `log_metrics` - Controls whether or not to record test metrics when running `train_model.py`
- `train_version` - Controls the version name of the model produced when running `train_model.py`
- `use_version` - Controls the model version used when running `use_model.py`
Next, manually curate a set of features and hyperparameters, placed in the correctly-versioned subfolder (`train_version`). Copy the format you see in `model_configs/`. Also copy the naming scheme (prefix with "gate" or "dir" depending on which binary task you are training for).\
You can use `select_features.py` and `tune_params.py` to assist with your curation. See below for details.\
Now you are ready to train. Run `train_model.py` and the model will be saved to `models/`, sorted by instrument.

### Tuning
| &nbsp; | `select_features.py` | `tune_params.py` |
| --- | --- | --- |
| XGBoost | SHAP importances | Optuna<br/>Feature set follows `train_version` |

### Running
To use a model from the terminal, run `use_model.py` with the correct `use_version` set in `env.json`. Live data is fetched and inference is run on it, with the prediction being printed to the terminal.

### Deploying
1. Copy the trained model file and its feature list from `XGBoost/models/<instrument>/` into `dist/artifacts/<pattern_name>/`, following the existing naming scheme (`XGBoost_EUR_USD_H1_2026_v<N>.json` and `xgbFeatures_v<N>.json`).
2. In `dist/api/inference.py`, add the pattern name and version to `PATTERN_VERSIONS`.
3. In `dist/api/main.py`, add an entry to `PATTERN_REGISTRY` with the detector module, detector kwargs, active-window size, prediction labels, features to inject from the detected instance, and a `get_meta` function that extracts the human-readable response fields.
4. In `web_interface/js/config.js`, add an entry to `PATTERN_CONFIGS` with the endpoint (`/pattern/<name>`), the class labels and colours for the probability bars, and an optional `renderMeta` function for any pattern-specific metadata to display in the card footer.
<br/>
