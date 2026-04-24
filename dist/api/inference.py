import numpy as np
import torch
import xgboost as xgb
import pandas as pd
from pathlib import Path

from api.patchtst import PatchTST

GATE_ARTIFACTS = Path("artifacts/gate")
DIR_ARTIFACTS = Path("artifacts/dir")

xgbGateVersion = 1
patchTstGateVersion = 1

# Loaded once at startup
xgbGateModel = None
patchTstGateModel = None
patchTstGateCheckpoint = None

def loadModels():
    global xgbGateModel, patchTstGateModel, patchTstGateCheckpoint

    # load xgb gate
    xgbGateModel = xgb.XGBClassifier()
    xgbGateModel.load_model(GATE_ARTIFACTS / f"XGBoost_EUR_USD_H1_2026_v{xgbGateVersion}.json")

    # load patchtst gate
    checkpoint = torch.load(
        GATE_ARTIFACTS / f"PatchTST_EUR_USD_H1_2026_v{patchTstGateVersion}.pt",
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["config"]
    core_features = checkpoint["core_features"]
    corr_features = checkpoint["corr_features"]
    patchTstGateModel = PatchTST(len(core_features), len(corr_features), config)
    patchTstGateModel.load_state_dict(checkpoint["model_state_dict"])
    patchTstGateModel.eval()
    patchTstGateCheckpoint = checkpoint


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


def predictPatchTST(featuresDf: pd.DataFrame) -> dict:
    config = patchTstGateCheckpoint["config"]
    core_features = patchTstGateCheckpoint["core_features"]
    corr_features = patchTstGateCheckpoint["corr_features"]
    norm = patchTstGateCheckpoint["normalization"]

    lookback = config["lookback"]
    core_mean = np.array(norm["core_mean"], dtype=np.float32)
    core_std = np.array(norm["core_std"], dtype=np.float32)

    core_window = featuresDf[core_features].to_numpy(dtype=np.float32)[-lookback:]
    core_window = (core_window - core_mean) / core_std
    core_tensor = torch.tensor(core_window, dtype=torch.float32).unsqueeze(0)

    if corr_features:
        corr_mean = np.array(norm["corr_mean"], dtype=np.float32)
        corr_std = np.array(norm["corr_std"], dtype=np.float32)
        corr_window = featuresDf[corr_features].to_numpy(dtype=np.float32)[-lookback:]
        corr_window = (corr_window - corr_mean) / corr_std
        corr_tensor = torch.tensor(corr_window, dtype=torch.float32).unsqueeze(0)
    else:
        corr_tensor = torch.zeros(1, lookback, 0, dtype=torch.float32)

    with torch.no_grad():
        logit = patchTstGateModel(core_tensor, corr_tensor)
        probDir = torch.sigmoid(logit).item()

    probFlat = 1.0 - probDir
    pred = "DIR" if probDir >= 0.5 else "FLAT"

    return {
        "patchTstGatePred": pred,
        "patchTstGateProbs": {"0": probFlat, "1": probDir},
    }
