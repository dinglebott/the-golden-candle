from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from pathlib import Path

from api.inference import loadModels, predictPatchTST, predictCnnLstm, PATTERN_VERSIONS, patchTstGateVersion
from api.models import PredictionResponse, CandleInfo, PatternResponse
from api.data_processing import getData, parseData, parseLiveCorrelated
import api.fair_value_gap as fvg_detector

logger = logging.getLogger(__name__)
GATE_ARTIFACTS = Path("artifacts/gate")
N_VALUE = 6
MIN_GAP_ATR_RATIO = 0.3

# Registry for pattern endpoints. To add a new pattern:
#   1. Implement a detector module with detect(df, **kwargs) -> list[dict]
#      Each instance dict must have "index" (row in df) and all keys listed in get_meta.
#   2. Add an entry here with the config keys shown below.
#   3. Add the version to PATTERN_VERSIONS in inference.py.
#   4. Add the pattern to PATTERN_CONFIGS in web_interface/js/config.js.
PATTERN_REGISTRY: dict[str, dict] = {
    "fvg": {
        "detector": fvg_detector,
        "detector_kwargs": {"min_gap_atr_ratio": MIN_GAP_ATR_RATIO},
        "n_active": N_VALUE,                         # trailing bars that count as active signal
        "pred_labels": {0: "NO FILL", 1: "FILL"},   # maps class index to prediction string
        "get_meta": lambda inst: {                   # extracts human-readable metadata for the response
            "direction": "bullish" if inst["direction"] == 1 else "bearish",
            "gap_low": inst["gap_low"],
            "gap_high": inst["gap_high"],
            "gap_atr_ratio": inst["gap_atr_ratio"],
            "detection_time": str(inst["time"]),
            "tp": inst["gap_low"] + 0.5 * inst["gap_size"],
            "sl": inst["candle_high"] + 1.5 * (inst["gap_size"] / inst["gap_atr_ratio"])
                  if inst["direction"] == 1
                  else inst["candle_low"] - 1.5 * (inst["gap_size"] / inst["gap_atr_ratio"]),
        },
    },
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading models...")
    loadModels()
    logger.info("Models loaded.")
    yield


app = FastAPI(
    title="Golden Candle API",
    version="1.3",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://dinglebott.github.io"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

assert (GATE_ARTIFACTS / f"PatchTST_EUR_USD_H1_2026_v{patchTstGateVersion}.pt").exists(), \
    f"PatchTST gate model not found for version {patchTstGateVersion}"
for _name, _version in PATTERN_VERSIONS.items():
    assert (Path(f"artifacts/{_name}") / f"CNN-LSTM_EUR_USD_H1_2026_v{_version}.pt").exists(), \
        f"CNN-LSTM {_name} model not found for version {_version}"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict", response_model=PredictionResponse)
def getPrediction():
    try:
        jsonData, timestamp = getData("EUR_USD", "H1", 500)
        jsonCorr, _ = getData("GBP_USD", "H1", 500)
        featuresDf = parseData(jsonData)
        df_corr = parseLiveCorrelated(jsonData, jsonCorr, "GBP_USD")
        featuresDf = featuresDf.merge(df_corr, on="time", how="inner")

        patchTstResult = predictPatchTST(featuresDf)
        return PredictionResponse(
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

    try:
        jsonData, timestamp = getData("EUR_USD", "H1", 500)
        df = parseData(jsonData)

        instances = cfg["detector"].detect(df, **cfg["detector_kwargs"])
        active = [inst for inst in instances if inst["index"] >= len(df) - cfg["n_active"]]

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
        result = predictCnnLstm(name, df, latest, cfg["pred_labels"])
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
        df = parseData(jsonData)
        lastCompleteCandle = df.iloc[-1]
        return CandleInfo(
            open=lastCompleteCandle["open"].item(),
            high=lastCompleteCandle["high"].item(),
            low=lastCompleteCandle["low"].item(),
            close=lastCompleteCandle["close"].item(),
            barrier=lastCompleteCandle["raw_atr"].item() * 1.5,
            timestamp=timestamp
        )
    except Exception as e:
        logger.error(f"Candle retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
