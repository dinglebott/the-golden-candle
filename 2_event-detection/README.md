## PROBLEM FRAMING
Here I take a 2-step approach. First, use hardcoded numerical criteria to filter out specific candlestick patterns from the dataset. Second, train a model to predict the probability that the setup resolves as expected. Thus the model only outputs a prediction when its specific pattern is detected; otherwise it has no output. It gives sparser but (hopefully) higher quality signals, instead of coin-tossing on every single candle.\
<br/>


## METHODOLOGY
### Labelling
The target variable is determined by triple-barrier labelling (Marcos López de Prado, 2018). Given parameters *k* and *n*, three barriers are set relative to some reference point (depending on pattern) - stop loss, take profit, and time barrier. The SL barrier is set *k* &times; ATR below/above the reference point (for bullish/bearish setups respectively). The TP barrier is set according to the expected resolution of the setup. The time barrier is set *n* candles after the pattern emerges. Labels are then computed based on which barrier is hit first.

### Feature Engineering
<br/>


## FILE STRUCTURE
At the experiment root is an `env.json` for config - see below for details.\
`patterns/` contains an event detection script for each pattern.\
Each model architecture has its own folder: XGBoost, CNN-LSTM, PatchTST. Within each folder:\
`select_features.py` - Feature selection tool\
`tune_params.py` - Hyperparameter tuning (Optuna)\
`results/` - Output location for feature selection, hyperparameter tuning, model test metrics\
`model_configs/` - Manually curated feature sets and hyperparameters, sorted by version\
`train_model.py` - Trains and tests a new model\
`models/` - Trained models, named by version number as set in configs\
`use_model.py` - Run live inference with a trained model\
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
To use a model from the terminal, run `use_model.py` with the correct `use_version` set in `env.json`. Live data is fetched and inference is run on it, with the prediction being printed to the terminal.\
<br/>