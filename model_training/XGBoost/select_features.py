import sys
import xgboost as xgb
import pandas as pd
import numpy as np
import shap
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from data_processing import dataparser

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

features = [
    "open_return", "high_return", "low_return", "close_return", "vol_return",
    "atr_14", "volatility_regime",
    "bb_width", "bb_position",
    "hl_spread", "upper_wick", "lower_wick",
    "dist_ema15", "dist_ema50", "ema_cross",
    "dist_smooth14", "dist_smooth35", "smooth_cross",
    "rsi_14", "macd_hist", "vol_ratio", "vol_momentum", "adx", "di_diff",
    "fast_pct_R", "slow_pct_R",
    "close_lag1", "close_lag2", "close_lag3", "close_lag4",
    "vol_lag1", "vol_lag2", "vol_lag3", "vol_lag4",
    "gbp_close_return", "gbp_return_spread", "gbp_rolling_corr_20", "gbp_cross_zscore"
]

# LOAD AND SPLIT DATA
_rawDataDir = Path(__file__).parent.parent.parent / "raw_data"
df = dataparser.parseData(_rawDataDir / f"{instrument}_{granularity}_{yearNow - 21}-01-01_{yearNow}-04-01.json")
df_corr = dataparser.parseCorrelated(instrument, "GBP_USD")
df = df.merge(df_corr, on="time", how="inner") # merge by union
if binary == 0:
    df = dataparser.addGateTarget(df, k, n)
elif binary == 1:
    df = dataparser.addDirectionTarget(df, k, n)
else:
    raise Exception("Binary must be 0 or 1")

trainEnd = int(trainSplit * len(df))
valEnd = int((trainSplit + valSplit) * len(df))
dfTrain = df.iloc[:trainEnd]
dfVal = df.iloc[trainEnd:valEnd]
dfTest = df.iloc[valEnd:]

X_train = dfTrain[features]
y_train = dfTrain["target"]
X_val = dfVal[features]
y_val = dfVal["target"]
X_test = dfTest[features]

model = xgb.XGBClassifier(
    max_depth=5,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=30,
    reg_alpha=20.0,
    reg_lambda=30.0,
    device=device,
    tree_method="hist",
    eval_metric="mlogloss",
    n_estimators=1000,
    early_stopping_rounds=50,
    random_state=42
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# RANK FEATURES
explainer = shap.TreeExplainer(model, X_train, feature_perturbation="interventional")
shapValues = explainer(X_val, check_additivity=False)
avgShaps = np.mean(np.abs(shapValues.values), axis=0)

shaps = pd.Series(avgShaps, index=features)
shaps.sort_values(ascending=False, inplace=True)
print(f"\n{shaps}")

Path("results").mkdir(exist_ok=True)
shaps.to_json(Path("results/feature_rankings.json"), indent=4)
