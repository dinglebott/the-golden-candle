import sys
import xgboost as xgb
from sklearn.metrics import log_loss
import optuna
import pandas as pd
import numpy as np
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from data_processing import dataparser

# shut up
optuna.logging.set_verbosity(optuna.logging.WARNING)

# LOAD CONFIG
with open(Path(__file__).parent.parent.parent / "env.json", "r") as file:
    env = json.load(file)
yearNow = env["year_now"]
instrument = env["instrument"]
granularity = env["granularity"]
k = env["k_value"]
n = env["n_value"]
trainSplit = env["train_split"]
valSplit = env["val_split"]
device = env["device"]
binary = env["binary"]

# LOAD FEATURES
with open("results/feature_rankings.json", "r") as file:
    featureRankings = json.load(file)
features = list(featureRankings.keys())
# manual selection - comment out if not using
features = [
    "volatility_regime", "vol_lag4", "vol_return", "vol_ratio", "vol_momentum",
    "vol_lag3", "vol_lag1", "vol_lag2", "atr_14", "gbp_rolling_corr_20",
    "bb_width", "hl_spread", "macd_hist", "lower_wick", "rsi_14", "upper_wick"
]
print(f"Tuning on {len(features)} features: {features}")

# LOAD DATA
_rawDataDir = Path(__file__).parent.parent.parent / "raw_data"
df = dataparser.parseData(_rawDataDir / f"{instrument}_{granularity}_{yearNow - 21}-01-01_{yearNow}-04-01.json")
df_corr = dataparser.parseCorrelated(instrument, "GBP_USD")
df = df.merge(df_corr, on="time", how="inner") # merge by union
valEnd = int((trainSplit + valSplit) * len(df))
df = df[:valEnd] # exclude test set
if binary == 0:
    df = dataparser.addGateTarget(df, k, n)
elif binary == 1:
    df = dataparser.addDirectionTarget(df, k, n)
else:
    raise Exception("Binary must be 0 or 1")

X = df[features]
y = df["target"]

# OPTUNA MAGIC
def objective(trial):
    params = {
        "verbosity": 0,
        "max_depth": trial.suggest_categorical("max_depth", [2, 3, 4, 5, 6]),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3),
        "subsample": trial.suggest_float("subsample", 0.4, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8),
        "min_child_weight": trial.suggest_int("min_child_weight", 10, 100, log=True),
        "reg_alpha": trial.suggest_int("reg_alpha", 15, 40),
        "reg_lambda": trial.suggest_int("reg_lambda", 18, 40),
        "device": device,
        "tree_method": "hist"
    }

    def crossValSplit(n_samples, n_splits, val_ratio):
        allIdxs = []
        for split in range(n_splits):
            trainEnd = int(n_samples * (1 - (n_splits - split) * val_ratio))
            valEnd = trainEnd + int(n_samples * val_ratio)
            allIdxs.append((range(trainEnd), range(trainEnd, valEnd)))
        return allIdxs

    foldScores = []
    for trainIdxs, valIdxs in crossValSplit(len(X), 4, 0.1):
        X_train, X_val = X.iloc[trainIdxs], X.iloc[valIdxs]
        y_train, y_val = y.iloc[trainIdxs], y.iloc[valIdxs]

        model = xgb.XGBClassifier(
            **params, eval_metric="mlogloss",
            n_estimators=1000,
            early_stopping_rounds=50,
            random_state=42
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)], verbose=False
        )
        foldScores.append(log_loss(y_val, model.predict_proba(X_val)))

    return np.mean(foldScores)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100, show_progress_bar=True)

finalParams = pd.Series(study.best_params)
finalParams["max_depth"] = round(finalParams["max_depth"])
finalParams["min_child_weight"] = round(finalParams["min_child_weight"])
print("\nFinal hyperparameters:")
print(finalParams)

Path("results").mkdir(exist_ok=True)
finalParams.to_json(Path("results/tuned_params.json"), indent=4)
