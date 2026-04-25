import sys
import xgboost as xgb
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_processing import datafetcher, dataparser
from patterns import registry

# LOAD CONFIGS
with open(Path(__file__).parent.parent / "env.json", "r") as f:
    env = json.load(f)
year_now = env["year_now"]
instrument = env["instrument"]
granularity = env["granularity"]
pattern = env["pattern"]
version = env["xgb"]["use_version"]

# LOAD EVENT DETECTOR
pattern_module = registry.load(pattern)

# LOAD MODEL + CONFIGS
with open(Path(__file__).parent / f"model_configs/v{version}/{pattern}_features.json", "r") as f:
    features = json.load(f)["features"]

model = xgb.XGBClassifier()
model_path = Path(__file__).parent / f"models/{instrument}/{pattern}_XGBoost_{instrument}_{granularity}_{year_now}_v{version}.json"
try:
    model.load_model(model_path)
except (xgb.core.XGBoostError, FileNotFoundError) as e:
    raise RuntimeError(f"Error loading model: {e}")

# FETCH LIVE DATA
json_data, _ = datafetcher.getData(instrument, granularity, 500)
df = dataparser.parseData(json_data)

instances = pattern_module.detect(df)
if not instances:
    print(f"No {pattern.upper()} instances detected in latest data.")
else:
    latest = instances[-1]
    row = df.iloc[[latest["index"]]].copy()
    for feat in pattern_module.METADATA_FEATURES:
        row[feat] = latest[feat]

    probabilities = model.predict_proba(row[features])[0] * 100
    direction_label = "bullish" if latest["direction"] == 1 else "bearish"

    print(f"\nLatest {pattern.upper()} detected at {latest['time']} ({direction_label})")
    print(f"Gap: {latest['gap_low']:.5f} – {latest['gap_high']:.5f} | gap/ATR: {latest['gap_atr_ratio']:.2f}")
    print(f"\nNO FILL: {probabilities[0]:.2f}%")
    print(f"FILL:    {probabilities[1]:.2f}%")
    print(f"\nFinal prediction: {'FILL' if probabilities[1] > probabilities[0] else 'NO FILL'}")
