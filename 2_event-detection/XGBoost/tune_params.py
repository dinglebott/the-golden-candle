import sys
import xgboost as xgb
from sklearn.metrics import average_precision_score
import optuna
import pandas as pd
import numpy as np
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_processing import dataparser
from patterns import registry
from symmetry import apply_flip_df

# shut up optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# LOAD CONFIGS
with open(Path(__file__).parent.parent / "env.json", "r") as f:
    env = json.load(f)
year_now = env["year_now"]
instrument = env["instrument"]
granularity = env["granularity"]
n = env["n_value"]
train_split = env["train_split"]
val_split = env["val_split"]
device = env["device"]
pattern = env["pattern"]

# LOAD EVENT DETECTOR
pattern_module = registry.load(pattern)

# LOAD MODEL CONFIGS
with open(Path(__file__).parent / f"model_configs/training_models/{pattern}_features.json", "r") as f:
    features = json.load(f)["features"]
print(f"Tuning on {len(features)} features: {features}")

# LOAD AND PARSE DATA
_raw_data_dir = Path(__file__).parent.parent.parent / "raw_data"
df = dataparser.parseData(_raw_data_dir / f"{instrument}_{granularity}_{year_now - 21}-01-01_{year_now}-04-01.json")

# FILTER AND SPLIT DATA
instances = pattern_module.detect(df)
labelled = pattern_module.label_instances(df, instances, n)

indices = [inst["index"] for inst in labelled]
event_df = df.iloc[indices].copy().reset_index(drop=True)
event_df["target"] = [inst["label"] for inst in labelled]
for feat in pattern_module.METADATA_FEATURES:
    event_df[feat] = [inst[feat] for inst in labelled]
apply_flip_df(event_df, [inst["direction"] for inst in labelled])

val_end = int((train_split + val_split) * len(event_df))
event_df = event_df.iloc[:val_end]

X = event_df[features]
y = event_df["target"]


# OPTUNA MAGIC
def objective(trial):
    params = {
        "verbosity": 0,
        "max_depth": trial.suggest_categorical("max_depth", [3, 4, 5, 6, 7]),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 0.8),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.8),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 30, log=True),
        "reg_alpha": trial.suggest_int("reg_alpha", 1, 30, log=True),
        "reg_lambda": trial.suggest_int("reg_lambda", 1, 30, log=True),
        "device": device,
        "tree_method": "hist",
    }

    def cross_val_splits(n_samples, n_splits, val_ratio):
        for split in range(n_splits):
            train_end = int(n_samples * (1 - (n_splits - split) * val_ratio))
            val_end = train_end + int(n_samples * val_ratio)
            yield range(train_end), range(train_end, val_end)

    fold_scores = []
    for train_idxs, val_idxs in cross_val_splits(len(X), 4, 0.1):
        X_train, X_val = X.iloc[train_idxs], X.iloc[val_idxs]
        y_train, y_val = y.iloc[train_idxs], y.iloc[val_idxs]
        model = xgb.XGBClassifier(
            **params,
            n_estimators=1000,
            early_stopping_rounds=50,
            random_state=42,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        fold_scores.append(average_precision_score(y_val, model.predict_proba(X_val)[:, 1]))

    return np.mean(fold_scores)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=60, show_progress_bar=True)

# RECORD RESULTS
final_params = pd.Series(study.best_params)
final_params["max_depth"] = round(final_params["max_depth"])
final_params["min_child_weight"] = round(final_params["min_child_weight"])
print("\nFinal hyperparameters:")
print(final_params)

(Path(__file__).parent / "results").mkdir(exist_ok=True)
final_params.to_json(Path(__file__).parent / f"results/{pattern}_tuned_params.json", indent=4)
