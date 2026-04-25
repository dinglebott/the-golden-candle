import sys
import xgboost as xgb
import pandas as pd
import numpy as np
import shap
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_processing import dataparser
from patterns import registry

# LOAD CONFIGS
with open(Path(__file__).parent.parent / "env.json", "r") as f:
    env = json.load(f)
year_now = env["year_now"]
instrument = env["instrument"]
granularity = env["granularity"]
n = env["n_value"]
k = env["k_value"]
train_split = env["train_split"]
val_split = env["val_split"]
device = env["device"]
pattern = env["pattern"]

# LOAD EVENT DETECTOR
pattern_module = registry.load(pattern)

candidate_features = [
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
    "tod_sin", "tod_cos",
    "body_to_range", "close_in_bar",
    "cum_return_3", "cum_return_6", "cum_return_12", "cum_return_24",
    "return_accel_3_12", "ema15_slope_3", "ema50_slope_5",
    "breakout_dist_high_24", "breakout_dist_low_24", "range_pos_24",
    "momentum_consistency_8", "vol_adj_return_6", "trend_pressure_8", "return_zscore_24",
] + pattern_module.METADATA_FEATURES

# LOAD AND PARSE DATA
_raw_data_dir = Path(__file__).parent.parent.parent / "raw_data"
df = dataparser.parseData(_raw_data_dir / f"{instrument}_{granularity}_{year_now - 21}-01-01_{year_now}-04-01.json")

# FILTER AND SPLIT DATA
instances = pattern_module.detect(df)
labelled = pattern_module.label_instances(df, instances, n, k)

indices = [inst["index"] for inst in labelled]
event_df = df.iloc[indices].copy().reset_index(drop=True)
event_df["target"] = [inst["label"] for inst in labelled]
for feat in pattern_module.METADATA_FEATURES:
    event_df[feat] = [inst[feat] for inst in labelled]

train_end = int(train_split * len(event_df))
val_end = int((train_split + val_split) * len(event_df))
df_train = event_df.iloc[:train_end]
df_val = event_df.iloc[train_end:val_end]

X_train = df_train[candidate_features]
y_train = df_train["target"]
X_val = df_val[candidate_features]
y_val = df_val["target"]

# TRAIN MODEL
model = xgb.XGBClassifier(
    max_depth=4,
    learning_rate=0.074398296,
    subsample=0.7995419972,
    colsample_bytree=0.7558848453,
    min_child_weight=7.0,
    reg_alpha=29.0,
    reg_lambda=11.0,
    scale_pos_weight= 0.5,
    device=device,
    tree_method="hist",
    n_estimators=1000,
    early_stopping_rounds=50,
    random_state=42,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# EVALUATE FEATURES
explainer = shap.TreeExplainer(model, X_train, feature_perturbation="interventional")
shap_values = explainer(X_val, check_additivity=False)
avg_shaps = np.mean(np.abs(shap_values.values), axis=0)

shaps = pd.Series(avg_shaps, index=candidate_features)
shaps.sort_values(ascending=False, inplace=True)
print(f"\n{shaps}")

(Path(__file__).parent / "results").mkdir(exist_ok=True)
shaps.to_json(Path(__file__).parent / f"results/{pattern}_feature_rankings.json", indent=4)
