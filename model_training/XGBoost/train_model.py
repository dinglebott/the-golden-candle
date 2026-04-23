import sys
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, log_loss, roc_auc_score, confusion_matrix
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

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
binary = env["binary"]
corrPair = env["corr_pair"]
log_metrics = env["log_metrics"]

version = env["xgb"]["train_version"]

# LOAD AND SPLIT DATAFRAMES
_rawDataDir = Path(__file__).parent.parent.parent / "raw_data"
df = dataparser.parseData(_rawDataDir / f"{instrument}_{granularity}_{yearNow - 21}-01-01_{yearNow}-04-01.json")
# conditional correlated pair loading
if isinstance(corrPair, str):
    df_corr = dataparser.parseCorrelated(instrument, corrPair)
    df = df.merge(df_corr, on="time", how="inner") # merge by union
elif corrPair != 0:
    raise Exception("corr_pair must be 0 or a valid currency pair string")
# add target
if binary == 0:
    df = dataparser.addGateTarget(df, k, n)
elif binary == 1:
    df = dataparser.addDirectionTarget(df, k, n)
else:
    raise Exception("Binary must be 0 or 1")

trainEndIdx = int(trainSplit * len(df))
valEndIdx = int((trainSplit + valSplit) * len(df))
dfTrain = df[:trainEndIdx]
dfVal = df[trainEndIdx:valEndIdx]
dfTest = df[valEndIdx:]

# DEFINE FEATURES
modelMode = "gate" if binary == 0 else "dir"
with open(Path(f"model_configs/v{version}/{modelMode}_features.json"), "r") as file:
    bestFeatures = json.load(file)["features"]
print(f"Best {len(bestFeatures)} features:", bestFeatures)

# DEFINE HYPERPARAMETERS (use results from Phase 3)
with open(Path(f"model_configs/v{version}/{modelMode}_params.json"), "r") as file:
    bestParams = json.load(file)
# cast floats to ints where necessary
bestParams["max_depth"] = int(bestParams["max_depth"])
bestParams["min_child_weight"] = int(bestParams["min_child_weight"])
print("Best hyperparameters:", bestParams)

# DEFINE DATASETS
X_train = dfTrain[bestFeatures]
y_train = dfTrain["target"]
X_val = dfVal[bestFeatures]
y_val = dfVal["target"]
X_test = dfTest[bestFeatures]
y_test = dfTest["target"]

# BUILD MODEL
model = xgb.XGBClassifier(
    **bestParams,
    n_estimators=1000, # high ceiling
    early_stopping_rounds=50, # stop after metric plateaus for 50 rounds
    random_state=42,
    device="cpu",
    tree_method="hist"
)

# TRAIN MODEL
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)], verbose=False
)

# TEST MODEL
y_pred = model.predict(X_test)
y_predTrain = model.predict(X_train) # for overfitting evaluation
# returns 1D array of shape (n_samples)
# values 0 | 1
y_prob = model.predict_proba(X_test)
# returns 2D array of shape (n_samples, n_classes)
# chance of 0 | 1 for each datapoint

# EVALUATE MODEL
accuracy = balanced_accuracy_score(y_test, y_pred) * 100
mcc = matthews_corrcoef(y_test, y_pred)
trainMcc = matthews_corrcoef(y_train, y_predTrain)
logLossScore = log_loss(y_test, y_prob)
rocAucScore = roc_auc_score(y_test, y_prob[:, 1])
cmatrix = confusion_matrix(y_test, y_pred)
cmatrixDf = pd.DataFrame(cmatrix, index=["Real 0", "Real 1"], columns=["Pred 0", "Pred 1"])
cmatrixDf["Count"] = cmatrixDf.sum(axis=1)
cmatrixDf.loc["Count"] = cmatrixDf.sum(axis=0)

# RECORD RESULTS
lines = []
lines.append(f"=== v{version} | {instrument} {granularity} | {modelMode} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
lines.append(f"Accuracy (balanced): {accuracy:.2f}%")
lines.append(f"MCC: {mcc:.4f}")
lines.append(f"MCC (train set): {trainMcc:.4f}")
lines.append(f"Log loss score: {logLossScore:.4f}")
lines.append(f"ROC-AUC score: {rocAucScore:.4f}")
lines.append(f"\nConfusion matrix:\n{cmatrixDf}")
# average probability distribution per class per true class
for true_class, name in enumerate(["flat", "directional"] if binary == 0 else ["down", "up"]):
    mask = y_test == true_class
    avg_probs = y_prob[mask].mean(axis=0)
    labels = ["P(flat)", "P(directional)"] if binary == 0 else ["P(down)", "P(up)"]
    lines.append(f"True={name}: avg {labels[0]}={avg_probs[0]:.3f} {labels[1]}={avg_probs[1]:.3f}")

output = "\n".join(lines) + "\n"
print(output)

if log_metrics:
    Path("results").mkdir(exist_ok=True)
    with open(Path("results/test_metrics.log"), "a") as logFile:
        logFile.write(output + "\n")

Path(f"models/{instrument}").mkdir(parents=True, exist_ok=True)
model.save_model(Path(f"models/{instrument}/{modelMode}_XGBoost_{instrument}_{granularity}_{yearNow}_v{version}.json"))
