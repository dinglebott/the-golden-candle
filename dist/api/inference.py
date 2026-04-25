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

# Add new pattern names and their model versions here when deploying a new pattern model.
PATTERN_VERSIONS: dict[str, int] = {
    "fvg": 1,
}

_xgbGateModel = None
_patchTstGateModel = None
_patchTstGateCheckpoint = None
_patternModels: dict[str, xgb.XGBClassifier] = {}


def loadModels():
    global _xgbGateModel, _patchTstGateModel, _patchTstGateCheckpoint

    _xgbGateModel = xgb.XGBClassifier()
    _xgbGateModel.load_model(GATE_ARTIFACTS / f"XGBoost_EUR_USD_H1_2026_v{xgbGateVersion}.json")

    for name, version in PATTERN_VERSIONS.items():
        model = xgb.XGBClassifier()
        model.load_model(Path(f"artifacts/{name}") / f"XGBoost_EUR_USD_H1_2026_v{version}.json")
        _patternModels[name] = model

    checkpoint = torch.load(
        GATE_ARTIFACTS / f"PatchTST_EUR_USD_H1_2026_v{patchTstGateVersion}.pt",
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["config"]
    core_features = checkpoint["core_features"]
    corr_features = checkpoint["corr_features"]
    _patchTstGateModel = PatchTST(len(core_features), len(corr_features), config)
    _patchTstGateModel.load_state_dict(checkpoint["model_state_dict"])
    _patchTstGateModel.eval()
    _patchTstGateCheckpoint = checkpoint


def predict(xgbGateFeaturesDf: pd.DataFrame) -> dict:
    latestCandle = xgbGateFeaturesDf.iloc[[-1]]
    xgbGatePred = _xgbGateModel.predict(latestCandle)[0]
    xgbGateProbs = _xgbGateModel.predict_proba(latestCandle)[0]

    xgbGatePredMap = {"0": "FLAT", "1": "DIR"}
    xgbGateProbsDict = {
        "0": xgbGateProbs[0],
        "1": xgbGateProbs[1]
    }

    return {
        "xgbGatePred": xgbGatePredMap[str(xgbGatePred)],
        "xgbGateProbs": xgbGateProbsDict
    }


def predictPattern(
    name: str,
    df: pd.DataFrame,
    instance: dict,
    inject_keys: list[str],
    features: list[str],
    pred_labels: dict[int, str],
) -> dict:
    row = df.iloc[[instance["index"]]].copy()
    for key in inject_keys:
        row[key] = instance[key]
    probs = _patternModels[name].predict_proba(row[features])[0]
    pred = pred_labels[1] if probs[1] >= probs[0] else pred_labels[0]
    return {
        "pred": pred,
        "probs": {"0": float(probs[0]), "1": float(probs[1])},
    }


def predictPatchTST(featuresDf: pd.DataFrame) -> dict:
    config = _patchTstGateCheckpoint["config"]
    core_features = _patchTstGateCheckpoint["core_features"]
    corr_features = _patchTstGateCheckpoint["corr_features"]
    norm = _patchTstGateCheckpoint["normalization"]

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
        logit = _patchTstGateModel(core_tensor, corr_tensor)
        probDir = torch.sigmoid(logit).item()

    probFlat = 1.0 - probDir
    pred = "DIR" if probDir >= 0.5 else "FLAT"

    return {
        "patchTstGatePred": pred,
        "patchTstGateProbs": {"0": probFlat, "1": probDir},
    }
