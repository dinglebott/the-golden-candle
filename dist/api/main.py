from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import json

from api.inference import loadModels, predict, predictPatchTST
from api.models import PredictionResponse, CandleInfo
from api.data_processing import getData, parseData, parseLiveCorrelated

logger = logging.getLogger(__name__)
GATE_ARTIFACTS = Path("artifacts/gate")
DIR_ARTIFACTS = Path("artifacts/dir")
K_VALUE = 1.5

xgbGateVersion = 1
patchTstGateVersion = 1

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
    version="1.1",
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
assert (GATE_ARTIFACTS / f"xgbFeatures_v{xgbGateVersion}.json").exists(), \
    f"XGB feature list not found for version {xgbGateVersion}"
assert (GATE_ARTIFACTS / f"gate_PatchTST_EUR_USD_H1_2026_v{patchTstGateVersion}.pt").exists(), \
    f"PatchTST gate model not found for version {patchTstGateVersion}"

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
