from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import json

from api.inference import loadModels, predict, predictPatchTST, predictPattern, PATTERN_VERSIONS
from api.models import PredictionResponse, CandleInfo, PatternResponse
from api.data_processing import getData, parseData, parseLiveCorrelated
import api.fair_value_gap as fvg_detector

logger = logging.getLogger(__name__)
GATE_ARTIFACTS = Path("artifacts/gate")
DIR_ARTIFACTS = Path("artifacts/dir")
K_VALUE = 1.5
N_VALUE = 6
MIN_GAP_ATR_RATIO = 0.3

xgbGateVersion = 1
patchTstGateVersion = 1

# Registry for pattern endpoints. To add a new pattern:
#   1. Implement a detector module with detect(df, **kwargs) -> list[dict]
#      Each instance dict must have "index" (row in df) and all keys listed in inject_features.
#   2. Add an entry here with the config keys shown below.
#   3. Add the version to PATTERN_VERSIONS in inference.py.
#   4. Add the pattern to PATTERN_CONFIGS in web_interface/js/config.js.
PATTERN_REGISTRY: dict[str, dict] = {
    "fvg": {
        "detector": fvg_detector,
        "detector_kwargs": {"min_gap_atr_ratio": MIN_GAP_ATR_RATIO},
        "n_active": N_VALUE,                         # trailing bars that count as active signal
        "pred_labels": {0: "NO_FILL", 1: "FILL"},   # maps class index to prediction string
        "inject_features": ["gap_atr_ratio", "direction"],  # instance keys to inject as model features
        "get_meta": lambda inst: {                   # extracts human-readable metadata for the response
            "direction": "bullish" if inst["direction"] == 1 else "bearish",
            "gap_low": inst["gap_low"],
            "gap_high": inst["gap_high"],
            "gap_atr_ratio": inst["gap_atr_ratio"],
            "detection_time": str(inst["time"]),
        },
    },
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup — load models into memory
    logger.info("Loading models...")
    loadModels()
    logger.info("Models loaded.")
    yield
    # Runs at shutdown (cleanup if needed)


app = FastAPI(
    title="Golden Candle API",
    version="1.2",
    lifespan=lifespan
)

# Allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://dinglebott.github.io"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# guard against silent failure
assert (GATE_ARTIFACTS / f"XGBoost_EUR_USD_H1_2026_v{xgbGateVersion}.json").exists(), \
    f"XGB gate model not found for version {xgbGateVersion}"
assert (GATE_ARTIFACTS / f"xgbFeatures_v{xgbGateVersion}.json").exists(), \
    f"XGB feature list not found for version {xgbGateVersion}"
assert (GATE_ARTIFACTS / f"PatchTST_EUR_USD_H1_2026_v{patchTstGateVersion}.pt").exists(), \
    f"PatchTST gate model not found for version {patchTstGateVersion}"
for _name, _version in PATTERN_VERSIONS.items():
    assert (Path(f"artifacts/{_name}") / f"XGBoost_EUR_USD_H1_2026_v{_version}.json").exists(), \
        f"XGB {_name} model not found for version {_version}"
    assert (Path(f"artifacts/{_name}") / f"xgbFeatures_v{_version}.json").exists(), \
        f"XGB {_name} feature list not found for version {_version}"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict", response_model=PredictionResponse)
def getPrediction():
    with open(GATE_ARTIFACTS / f"xgbFeatures_v{xgbGateVersion}.json", "r") as file:
        xgbGateFeatureList = json.load(file)["features"]

    try:
        jsonData, timestamp = getData("EUR_USD", "H1", 500)
        jsonCorr, _ = getData("GBP_USD", "H1", 500)
        featuresDf = parseData(jsonData)
        df_corr = parseLiveCorrelated(jsonData, jsonCorr, "GBP_USD")
        featuresDf = featuresDf.merge(df_corr, on="time", how="inner")
        xgbGateFeaturesDf = featuresDf[xgbGateFeatureList]

        xgbResult = predict(xgbGateFeaturesDf)
        patchTstResult = predictPatchTST(featuresDf)
        return PredictionResponse(
            **xgbResult,
            xgbGateVersion=f"{xgbGateVersion}",
            **patchTstResult,
            patchTstGateVersion=f"{patchTstGateVersion}",
            timestamp=timestamp,
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pattern/{name}", response_model=PatternResponse)
def getPatternPrediction(name: str):
    if name not in PATTERN_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Pattern '{name}' not registered")

    cfg = PATTERN_REGISTRY[name]
    version = PATTERN_VERSIONS[name]
    feature_path = Path(f"artifacts/{name}") / f"xgbFeatures_v{version}.json"

    with open(feature_path, "r") as file:
        featureList = json.load(file)["features"]

    try:
        jsonData, timestamp = getData("EUR_USD", "H1", 500)
        df = parseData(jsonData)

        instances = cfg["detector"].detect(df, **cfg["detector_kwargs"])
        active = [inst for inst in instances if inst["index"] >= len(df) - cfg["n_active"]] # avoid stale signals

        if not active:
            return PatternResponse(
                detected=False,
                pred="NO_SIGNAL",
                probs=None,
                version=str(version),
                meta=None,
                timestamp=timestamp,
            )

        latest = active[-1]
        result = predictPattern(name, df, latest, cfg["inject_features"], featureList, cfg["pred_labels"])
        return PatternResponse(
            detected=True,
            pred=result["pred"],
            probs=result["probs"],
            version=str(version),
            meta=cfg["get_meta"](latest),
            timestamp=timestamp,
        )
    except Exception as e:
        logger.error(f"{name} prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/candle", response_model=CandleInfo)
def getCandleInfo():
    try:
        jsonData, timestamp = getData("EUR_USD", "H1", 500)
        df = parseData(jsonData) # incomplete candle dropped here
        lastCompleteCandle = df.iloc[-1]
        return CandleInfo(
            open=lastCompleteCandle["open"].item(),
            high=lastCompleteCandle["high"].item(),
            low=lastCompleteCandle["low"].item(),
            close=lastCompleteCandle["close"].item(),
            barrier=lastCompleteCandle["raw_atr"].item() * K_VALUE,
            timestamp=timestamp
        )
    except Exception as e:
        logger.error(f"Candle retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
