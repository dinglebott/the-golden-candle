import sys
import xgboost as xgb
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from data_processing import datafetcher, dataparser

# GLOBAL VARIABLES
with open(Path(__file__).parent.parent.parent / "env.json", "r") as file:
    env = json.load(file)
yearNow = env["year_now"]
instrument = env["instrument"]
granularity = env["granularity"]
binary = env["binary"]
corrPair = env["corr_pair"]

version = env["xgb"]["use_version"]

# DEFINE FEATURES (copy-paste from the model training features exactly)
modelMode = "gate" if binary == 0 else "dir"
with open(Path(f"model_configs/{modelMode}_features_v{version}.json"), "r") as file:
    features = json.load(file)["features"]

# LOAD MODEL
if binary == 0:
    modelMode = "gate"
elif binary == 1:
    modelMode = "dir"
else:
    raise Exception("Binary must be 0 or 1.")

model = xgb.XGBClassifier()
try:
    model.load_model(Path(f"models/{instrument}/{modelMode}_XGBoost_{instrument}_{granularity}_{yearNow}_v{version}.json"))
except (xgb.core.XGBoostError, FileNotFoundError) as e:
    raise RuntimeError(f"Error loading model: {e}")

# FETCH AND PARSE CURRENT DATA
jsonData = datafetcher.getData(instrument, granularity, 200)
df = dataparser.parseData(jsonData)
# conditional correlated pair loading
if isinstance(corrPair, str):
    jsonCorr = datafetcher.getData(corrPair, granularity, 200)
    df_corr = dataparser.parseLiveCorrelated(jsonData, jsonCorr, corrPair)
    df = df.merge(df_corr, on="time", how="inner")
elif corrPair != 0:
    raise Exception("corr_pair must be 0 or a valid currency pair string")

# GET PREDICTION
latestCandle = df[features].iloc[[-1]] # slice out last row (last candle)
prediction = model.predict(latestCandle)[0] # gets the only element of the 1D numpy array [n_samples]
probabilities = model.predict_proba(latestCandle)[0] # gets the only row of the 2D numpy array [n_samples, n_classes]
probabilities *= 100 # convert to percentages

# DISPLAY RESULTS
if binary == 0:
    classLabels = ["FLAT", "DIRECTIONAL"]
else:
    classLabels = ["DOWN", "UP"]

print("")
for label, prob in zip(classLabels, probabilities):
    print(f"{label}: {prob:.2f}%")

print(f"\nFinal prediction: {classLabels[prediction]}")
