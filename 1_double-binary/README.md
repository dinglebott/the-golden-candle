## PROBLEM FRAMING
With this approach, I divide the problem into 2 binary classifications. Firstly, does the price move in a direction (up/down), or does it remain flat? Secondly, given that it moves in a direction, which direction did it move in? Thus there are 2 types of models being trained here: flat/directional prediction, and up/down prediction. Both output a pair of probabilities, representing the chances of each outcome in their respective binaries.\
<br/>


## METHODOLOGY
### Labelling
The target variable is determined by triple-barrier labelling (Marcos López de Prado, 2018). Given parameters *k* and *n*, three barriers are set relative to the close price **C** of the latest candle - upper, lower, and time barrier. The upper and lower barriers are set *k* &times; ATR above and below **C**, and the time barrier is set *n* candles after the latest one. Labels are then computed based on which barrier is hit first.

### Feature Engineering
<br/>


## FILE STRUCTURE
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
- `k_value` - Coefficient to multiply ATR(14) by, determines distance of upper and lower barriers
- `n_value` - Length of time barrier from latest complete candle, expressed in candles
- `train_split`, `val_split` - Determines split ratio of the dataset (test set is whatever is left over)
- `binary` - Determines the training task for the models: 0 &rarr; flat/directional and 1 &rarr; up/down
- `corr_pair` - Controls the correlated pair for which to engineer additional features (0 for none)
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
| PatchTST | Permutation importances<br/>Model and feature set follow `use_version` | Optuna<br/>Feature set follows `train_version`<br/>Use --trials flag to control no. of trials (default 50) |

### Running
To use a model from the terminal, run `use_model.py` with the correct `use_version` set in `env.json`. Live data is fetched and inference is run on it, with the prediction being printed to the terminal.\
<br/>