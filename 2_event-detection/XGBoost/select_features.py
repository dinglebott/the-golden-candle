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
version = env["xgb"]["train_version"]

# LOAD EVENT DETECTOR
pattern_module = registry.load(pattern)

# LOAD MODEL PARAMS — use versioned configs if available, else fall back to defaults
_params_path = Path(__file__).parent / f"model_configs/v{version}/{pattern}_params.json"
if _params_path.exists():
    with open(_params_path) as f:
        _p = json.load(f)
    _model_params = {
        "max_depth":        int(_p["max_depth"]),
        "learning_rate":    _p["learning_rate"],
        "subsample":        _p["subsample"],
        "colsample_bytree": _p["colsample_bytree"],
        "min_child_weight": int(_p["min_child_weight"]),
        "reg_alpha":        _p["reg_alpha"],
        "reg_lambda":       _p["reg_lambda"],
    }
    print(f"Loaded params from model_configs/v{version}/{pattern}_params.json")
else:
    _model_params = {
        "max_depth": 4, "learning_rate": 0.1, "subsample": 0.8,
        "colsample_bytree": 0.8, "min_child_weight": 10, "reg_alpha": 1, "reg_lambda": 1,
    }
    print(f"No model_configs found for v{version} — using default params")

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
pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
model = xgb.XGBClassifier(
    **_model_params,
    scale_pos_weight=pos_weight,
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
