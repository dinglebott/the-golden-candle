import sys
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_score, matthews_corrcoef, confusion_matrix
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

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
pattern = env["pattern"]
log_metrics = env["log_metrics"]
version = env["xgb"]["train_version"]

# LOAD EVENT DETECTOR
pattern_module = registry.load(pattern)

# LOAD MODEL CONFIGS
with open(Path(__file__).parent / f"model_configs/v{version}/{pattern}_features.json", "r") as f:
    best_features = json.load(f)["features"]
print(f"Features ({len(best_features)}):", best_features)

with open(Path(__file__).parent / f"model_configs/v{version}/{pattern}_params.json", "r") as f:
    best_params = json.load(f)
best_params["max_depth"] = int(best_params["max_depth"])
best_params["min_child_weight"] = int(best_params["min_child_weight"])
print("Hyperparameters:", best_params)

# LOAD AND PARSE DATA
_raw_data_dir = Path(__file__).parent.parent.parent / "raw_data"
df = dataparser.parseData(_raw_data_dir / f"{instrument}_{granularity}_{year_now - 21}-01-01_{year_now}-04-01.json")

# FILTER AND SPLIT DATA
instances = pattern_module.detect(df)
labelled = pattern_module.label_instances(df, instances, n, k)
print(f"Detected {len(labelled)} {pattern.upper()} instances | fill rate: {sum(i['label'] for i in labelled) / len(labelled):.2%}")

def build_event_df(df, instances):
    indices = [inst["index"] for inst in instances]
    event_df = df.iloc[indices].copy().reset_index(drop=True)
    event_df["target"] = [inst["label"] for inst in instances]
    for feat in pattern_module.METADATA_FEATURES:
        event_df[feat] = [inst[feat] for inst in instances]
    return event_df

train_end = int(train_split * len(labelled))
val_end = int((train_split + val_split) * len(labelled))
df_train = build_event_df(df, labelled[:train_end])
df_val = build_event_df(df, labelled[train_end:val_end])
df_test = build_event_df(df, labelled[val_end:])

X_train, y_train = df_train[best_features], df_train["target"]
X_val, y_val = df_val[best_features], df_val["target"]
X_test, y_test = df_test[best_features], df_test["target"]

# TRAIN MODEL
model = xgb.XGBClassifier(
    **best_params,
    n_estimators=1000,
    early_stopping_rounds=50,
    random_state=42,
    device="cpu",
    tree_method="hist",
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# PREDICT
y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)
y_prob = model.predict_proba(X_test)

# EVALUATE MODEL
avgPrecision = average_precision_score(y_test, y_prob[:, 1])
precision1 = precision_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
train_mcc = matthews_corrcoef(y_train, y_pred_train)
cmatrix = confusion_matrix(y_test, y_pred)
cmatrix_df = pd.DataFrame(cmatrix, index=["Real 0", "Real 1"], columns=["Pred 0", "Pred 1"])
cmatrix_df["Count"] = cmatrix_df.sum(axis=1)
cmatrix_df.loc["Count"] = cmatrix_df.sum(axis=0)

total_candles = len(df)
instance_pct = len(labelled) / total_candles * 100

lines = []
lines.append(f"=== v{version} | {instrument} {granularity} | {pattern} | XGBoost | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
lines.append(f"Pattern instances: {len(labelled)} / {total_candles} candles ({instance_pct:.1f}%)")
lines.append(f"Train instances: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
lines.append(f"\nAverage precision: {avgPrecision:.4f}")
lines.append(f"Precision (fill): {precision1:.4f}")
lines.append(f"MCC: {mcc:.4f}")
lines.append(f"MCC (train set): {train_mcc:.4f}")
lines.append(f"\nConfusion matrix:\n{cmatrix_df}")
for true_class, name in enumerate(["no_fill", "fill"]):
    mask = y_test == true_class
    avg_probs = y_prob[mask].mean(axis=0)
    lines.append(f"True={name}: avg P(no_fill)={avg_probs[0]:.3f} P(fill)={avg_probs[1]:.3f}")

output = "\n".join(lines) + "\n"
print(output)

# LOG METRICS
if log_metrics:
    (Path(__file__).parent / "results").mkdir(exist_ok=True)
    with open(Path(__file__).parent / "results/test_metrics.log", "a") as f:
        f.write(output + "\n")

# SAVE MODEL
model_path = Path(__file__).parent / f"models/{instrument}/{pattern}_XGBoost_{instrument}_{granularity}_{year_now}_v{version}.json"
model_path.parent.mkdir(parents=True, exist_ok=True)
model.save_model(model_path)
