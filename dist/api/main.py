from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from pathlib import Path

from api.inference import loadModels, predictPatchTST, predictCnnLstm, PATTERN_VERSIONS, STRATEGY_VERSIONS, patchTstGateVersion
from api.models import PredictionResponse, CandleInfo, PatternResponse, StrategyResponse
from api.data_processing import getData, parseData, parseLiveCorrelated, parseDailyOhlc
import api.fair_value_gap as fvg_detector
import api.order_block as order_block_detector
import api.trend_pullback as trend_pullback_strategy
import api.liquidity_sweep as liquidity_sweep_strategy

logger = logging.getLogger(__name__)
ARTIFACTS = Path("artifacts")
N_VALUE = 6

# Registry for pattern endpoints. To add a new pattern:
#   1. Implement a detector module with detect(df, **kwargs) -> list[dict]
#      Each instance dict must have "index" (row in df) and all keys listed in get_meta.
#   2. Add an entry here with the config keys shown below.
#   3. Add the version to PATTERN_VERSIONS in inference.py.
#   4. Add the pattern to PATTERN_CONFIGS in web_interface/js/config.js.
PATTERN_REGISTRY: dict[str, dict] = {
    "fvg": {
        "detector": fvg_detector,
        "detector_kwargs": {"min_gap_atr_ratio": 0.3},
        "n_active": N_VALUE,                         # trailing bars that count as active signal
        "pred_labels": {0: "NO FILL", 1: "FILL"},   # maps class index to prediction string
        "get_meta": lambda inst: {                   # extracts human-readable metadata for the response
            "direction": "bullish" if inst["direction"] == 1 else "bearish",
            "gap_low": inst["gap_low"],
            "gap_high": inst["gap_high"],
            "gap_atr_ratio": inst["gap_atr_ratio"],
            "detection_time": str(inst["time"]),
            "tp": inst["fill_target"],
            "sl": inst["candle_high"] + 1.5 * inst["atr"]
                  if inst["direction"] == 1
                  else inst["candle_low"] - 1.5 * inst["atr"],
        },
    },

    "order_block": {
        "detector": order_block_detector,
        "detector_kwargs": {},
        "n_active": N_VALUE,
        "pred_labels": {0: "NO FILL", 1: "FILL"},
        "get_meta": lambda inst: {
            "direction": "bullish" if inst["direction"] == 1 else "bearish",
            "ob_time": str(inst["ob_time"]),
            "impulse_atr_ratio": inst["impulse_atr_ratio"],
            "zone_size_atr_ratio": inst["zone_size_atr_ratio"],
            "candles_elapsed": inst["candles_elapsed"],
            "detection_time": str(inst["time"]),
            "tp": inst["fill_target"],
            "sl": inst["ob_low"] - 0.5 * inst["atr"]
                  if inst["direction"] == 1
                  else inst["ob_high"] + 0.5 * inst["atr"],
        },
    },
}

# Per-strategy fetchers. Returns (kwargs_for_get_entries, timestamp). Each strategy
# pulls exactly the candles it needs — separating daily from H1 lets daily-EMA-based
# strategies converge without ballooning the H1 fetch.
def _fetch_h1_only():
    jsonData, timestamp = getData("EUR_USD", "H1", 500)
    return {"df": parseData(jsonData)}, timestamp

def _fetch_h1_plus_daily():
    h1_json, timestamp = getData("EUR_USD", "H1", 500)
    daily_json, _ = getData("EUR_USD", "D", 500, dailyAlignment=0, alignmentTimezone="UTC")
    return {"df": parseData(h1_json), "df_daily": parseDailyOhlc(daily_json)}, timestamp


# Registry for pure-rule strategy endpoints. To add a new strategy:
#   1. Drop a streamlined copy of the strategy module into dist/api/ that exposes
#      get_entries(df, ..., n_active) -> dict | None.
#   2. Add an entry here with a fetch fn that returns the kwargs that get_entries needs.
#   3. Add the version to STRATEGY_VERSIONS in inference.py.
#   4. Add the strategy to STRATEGY_CONFIGS in web_interface/js/config.js.
STRATEGY_REGISTRY: dict[str, dict] = {
    "trend_pullback": {
        "module": trend_pullback_strategy,
        "n_active": N_VALUE,
        "fetch": _fetch_h1_plus_daily,
    },
    "liquidity_sweep": {
        "module": liquidity_sweep_strategy,
        "n_active": N_VALUE,
        "fetch": _fetch_h1_only,
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
    version="2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://dinglebott.github.io"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

assert (ARTIFACTS / f"gate_PatchTST_EUR_USD_H1_2026_v{patchTstGateVersion}.pt").exists(), \
    f"PatchTST gate model not found for version {patchTstGateVersion}"
for _name, _version in PATTERN_VERSIONS.items():
    assert (ARTIFACTS / f"{_name}_CNN-LSTM_EUR_USD_H1_2026_v{_version}.pt").exists(), \
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


@app.get("/strategy/{name}", response_model=StrategyResponse)
def getStrategyPrediction(name: str):
    if name not in STRATEGY_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Strategy '{name}' not registered")

    cfg = STRATEGY_REGISTRY[name]
    version = STRATEGY_VERSIONS[name]

    try:
        inputs, timestamp = cfg["fetch"]()
        entry = cfg["module"].get_entries(**inputs, n_active=cfg["n_active"])

        if entry is None:
            return StrategyResponse(
                detected=False,
                version=str(version),
                meta=None,
                timestamp=timestamp,
            )
        return StrategyResponse(
            detected=True,
            version=str(version),
            meta=entry,
            timestamp=timestamp,
        )
    except Exception as e:
        logger.error(f"{name} strategy failed: {e}")
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
