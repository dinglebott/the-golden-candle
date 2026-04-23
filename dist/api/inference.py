import xgboost as xgb
import pandas as pd
from pathlib import Path

GATE_ARTIFACTS = Path("artifacts/gate")
DIR_ARTIFACTS = Path("artifacts/dir")

xgbGateVersion = 1
    
# Loaded once at startup
xgbGateModel = None

def loadModels():
    global xgbGateModel

    # load xgb gate
    xgbGateModel = xgb.XGBClassifier()
    xgbGateModel.load_model(GATE_ARTIFACTS / f"XGBoost_EUR_USD_H1_2026_v{xgbGateVersion}.json")


def predict(xgbGateFeaturesDf: pd.DataFrame) -> dict:
    # xgb gate prediction
    latestCandle = xgbGateFeaturesDf.iloc[[-1]]
    xgbGatePred = xgbGateModel.predict(latestCandle)[0]
    xgbGateProbs = xgbGateModel.predict_proba(latestCandle)[0]

    xgbGatePredMap = {"0": "FLAT", "1": "DIR"}
    xgbGateProbsDict = {
        "0": xgbGateProbs[0],
        "1": xgbGateProbs[1]
    }

    return {
        "xgbGatePred": xgbGatePredMap[str(xgbGatePred)],
        "xgbGateProbs": xgbGateProbsDict
    }
