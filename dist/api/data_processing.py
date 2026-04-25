# EXPORTS:
# parseData() adds features and returns DataFrame
# parseCorrelated() returns cross-pair divergence features aligned by timestamp
# splitByDate() returns specified slice of DataFrame by date
from dotenv import load_dotenv
import os
import requests
import glob
import json
import pandas as pd
import numpy as np
from pathlib import Path

# get oanda api key (from .env file)
load_dotenv()
apiKey = os.getenv("OANDA_KEY")

# global variables
headers = {"Authorization": f"Bearer {apiKey}"}
baseUrl = "https://api-fxtrade.oanda.com" # access token generated from live account (don't use fxpractice)

def getData(instr="EUR_USD", gran="H1", count=500):
    # get response
    params = {
        "granularity": gran,
        "count": count,
        "price": "M"
    }
    endpoint = f"/v3/instruments/{instr}/candles"
    response = requests.get(baseUrl + endpoint, headers=headers, params=params)
    
    # inspect response
    if response.status_code != 200:
        raise Exception(response.text)
    
    # return JSON-ified response
    data = response.json()
    timestamp = data["candles"][-1]["time"] if data["candles"][-1]["complete"] else data["candles"][-2]["time"]
    return data, timestamp


def ultSmoother(series: pd.Series, period=10):
    values = series.to_numpy(dtype=float)
    result = np.zeros(len(values))
    # coefficients
    f = (1.41421356 * np.pi) / period
    a1 = np.exp(-f)
    c2 = 2 * a1 * np.cos(f)
    c3 = -(a1 ** 2)
    c1 = (1 - c2 - c3) / 4
    # initialise
    result[0] = values[0]
    result[1] = values[1]
    # recurrence relation
    for i in range(2, len(values)):
        result[i] = (
            (1 - c1) * values[i] # weighted input
            + (2 * c1 - c2) * values[i-1] # weighted previous input
            - (c1 + c3) * values[i-2] # weighted input 2 bars ago
            + c2 * result[i-1] # feedback from previous output
            + c3 * result[i-2] # feedback from output 2 bars ago
        )
    return result


def parseData(jsonData):
    if isinstance(jsonData, (str, Path)):
        with open(jsonData, "r") as file:
            rawData = json.load(file) # rawData is a Python dict
    elif isinstance(jsonData, dict):
        rawData = jsonData
    else:
        raise TypeError(f"parseData expects a file path or dict, got {type(jsonData)}")
    
    # unpack dict into DataFrame
    records = []
    for c in rawData["candles"]:
        if c["complete"]:
            records.append({
                "time": c["time"],
                "open": float(c["mid"]["o"]), # convert from string
                "high": float(c["mid"]["h"]),
                "low": float(c["mid"]["l"]),
                "close": float(c["mid"]["c"]),
                "volume": c["volume"]
            })
    df = pd.DataFrame(records)

    # denoise volume
    df["volume"] = ultSmoother(df["volume"], 6)
    df["volume"] = df["volume"].clip(lower=1e-10)
    
    # ADD FEATURES
    # helper
    def getEma(period):
        return df["close"].ewm(span=period, adjust=False).mean()

    # Time of day (UTC hour encoded as sin/cos pair)
    hour = pd.to_datetime(df["time"]).dt.hour
    df["tod_sin"] = np.sin(2 * np.pi * hour / 24)
    df["tod_cos"] = np.cos(2 * np.pi * hour / 24)

    # Raw
    df["open_return"] = np.log(df["open"] / df["close"].shift(1))
    df["high_return"] = np.log(df["high"] / df["close"].shift(1))
    df["low_return"] = np.log(df["low"] / df["close"].shift(1))
    df["close_return"] = np.log(df["close"] / df["close"].shift(1))
    df["vol_return"] = np.log(df["volume"] / df["volume"].shift(1))
    
    # ATR
    trueRange = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs()
    ], axis=1).max(axis=1) # greatest of 3 values
    df["raw_atr"] = trueRange.ewm(alpha=1/14, adjust=False).mean()
    df["atr_14"] = df["raw_atr"] / df["close"]
    df["volatility_regime"] = df["atr_14"] / df["atr_14"].rolling(50).mean()
    
    # Bollinger bands
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["bb_width"] = (bb_upper - bb_lower) / bb_mid
    df["bb_position"] = (df["close"] - bb_lower) / (bb_upper - bb_lower)
    
    # Structure
    df["hl_spread"] = np.log(df["high"] / df["low"])
    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["raw_atr"]
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["raw_atr"]
    
    # EMAs
    df["dist_ema15"] = np.log(df["close"] / getEma(15))
    df["dist_ema50"] = np.log(df["close"] / getEma(50))
    df["ema_cross"] = np.log(getEma(12) / getEma(26))
    
    # Smoother EMA replacements
    df["dist_smooth14"] = np.log(df["close"] / ultSmoother(df["close"], 14))
    df["dist_smooth35"] = np.log(df["close"] / ultSmoother(df["close"], 35))
    df["smooth_cross"] = np.log(ultSmoother(df["close"], 8) / ultSmoother(df["close"], 18))
    
    # RSI
    def rsi(series, n=14):
        delta = series.diff()
        avgGain = delta.clip(lower=0).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
        avgLoss = (-delta.clip(upper=0)).ewm(alpha=1/n, min_periods=n, adjust=False).mean()
        relativeStrength = avgGain / avgLoss
        return 100 - (100 / (1 + relativeStrength))
    df["rsi_14"] = rsi(df["close"]) - 50
    
    # MACD histogram
    macd = getEma(12) - getEma(26)
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = (macd - macd_signal) / df["close"]
    
    # Volume
    vol_sma30 = df["volume"].rolling(30).mean()
    df["vol_ratio"] = df["volume"] / vol_sma30
    df["vol_momentum"] = df["vol_ratio"] - df["vol_ratio"].rolling(5).mean()
    
    # ADX, DIs
    def getAdx(df, period=14):
        high = df["high"]
        low = df["low"]
        # directional movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        # +DM: up move is greater than down move and positive
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        # smooth with wilder moving average (equivalent to EWM with alpha=1/period)
        plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean()
        minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean()
        atr_smooth = trueRange.ewm(alpha=1/period, adjust=False).mean()
        # directional indicators
        plus_di = 100 * plus_dm_smooth / atr_smooth
        minus_di = 100 * minus_dm_smooth / atr_smooth
        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx_line = dx.ewm(alpha=1/period, adjust=False).mean()

        return plus_di, minus_di, adx_line
    
    plus_di, minus_di, df["adx"] = getAdx(df, period=14)
    df["di_diff"] = plus_di - minus_di

    # Lagged features
    for lag in range(1, 5):
        df[f"close_lag{lag}"] = df["close_return"].shift(lag)
        df[f"vol_lag{lag}"] = df["vol_return"].shift(lag)

    # Williams %R
    fastHighest = df["high"].rolling(21).max()
    fastLowest = df["low"].rolling(21).min()
    slowHighest = df["high"].rolling(112).max()
    slowLowest = df["low"].rolling(112).min()
    fastR = (fastHighest - df["close"]) / (fastHighest - fastLowest) * -100
    slowR = (slowHighest - df["close"]) / (slowHighest - slowLowest) * -100
    df["fast_pct_R"] = fastR.ewm(span=7, adjust=False).mean() + 50
    df["slow_pct_R"] = slowR.ewm(span=3, adjust=False).mean() + 50

    # Directional features
    eps = 1e-10
    candle_range = (df["high"] - df["low"]).clip(lower=eps)
    prev_high_24 = df["high"].rolling(24).max().shift(1)
    prev_low_24 = df["low"].rolling(24).min().shift(1)
    prev_range_24 = (prev_high_24 - prev_low_24).clip(lower=eps)
    signed_return = np.sign(df["close_return"])

    # Intrabar directional pressure
    df["body_to_range"] = (df["close"] - df["open"]) / candle_range
    df["close_in_bar"] = ((df["close"] - df["low"]) / candle_range) - 0.5

    # Multi-horizon directional momentum
    df["cum_return_3"] = np.log(df["close"] / df["close"].shift(3))
    df["cum_return_6"] = np.log(df["close"] / df["close"].shift(6))
    df["cum_return_12"] = np.log(df["close"] / df["close"].shift(12))
    df["cum_return_24"] = np.log(df["close"] / df["close"].shift(24))
    df["return_accel_3_12"] = df["cum_return_3"] - (df["cum_return_12"] / 4)

    # Trend slope proxies from existing moving averages
    ema15 = getEma(15)
    ema50 = getEma(50)
    df["ema15_slope_3"] = np.log(ema15 / ema15.shift(3))
    df["ema50_slope_5"] = np.log(ema50 / ema50.shift(5))

    # Breakout and range-location context
    df["breakout_dist_high_24"] = np.log(df["close"] / prev_high_24)
    df["breakout_dist_low_24"] = np.log(df["close"] / prev_low_24)
    df["range_pos_24"] = ((df["close"] - prev_low_24) / prev_range_24) - 0.5

    # Directional persistence / impulse quality
    df["momentum_consistency_8"] = signed_return.rolling(8).mean()
    df["vol_adj_return_6"] = df["cum_return_6"] / (df["atr_14"] * np.sqrt(6) + eps)
    df["trend_pressure_8"] = df["body_to_range"].rolling(8).mean()
    df["return_zscore_24"] = (
        (df["close_return"] - df["close_return"].rolling(24).mean()) /
        (df["close_return"].rolling(24).std() + eps)
    )

    # drop empty rows and return
    df.dropna(inplace=True)
    return df


def _computeCorrFeatures(dfMain, dfCorr, prefix):
    df = dfMain.merge(dfCorr, on="time", suffixes=("_main", "_corr"))
    main_return = np.log(df["close_main"] / df["close_main"].shift(1))
    corr_return = np.log(df["close_corr"] / df["close_corr"].shift(1))
    df[f"{prefix}_close_return"] = corr_return
    df[f"{prefix}_return_spread"] = main_return - corr_return
    df[f"{prefix}_rolling_corr_20"] = main_return.rolling(20).corr(corr_return)
    cross = np.log(df["close_main"] / df["close_corr"])
    roll_mean = cross.rolling(75).mean()
    roll_std = cross.rolling(75).std()
    df[f"{prefix}_cross_zscore"] = (cross - roll_mean) / roll_std
    keep = ["time", f"{prefix}_close_return", f"{prefix}_return_spread",
            f"{prefix}_rolling_corr_20", f"{prefix}_cross_zscore"]
    return df[keep].dropna().reset_index(drop=True)


# extract only time + close from raw data
def _jsonToCloses(jsonData):
    raw = jsonData if isinstance(jsonData, dict) else json.load(open(jsonData))
    records = [
        {"time": c["time"], "close": float(c["mid"]["c"])}
        for c in raw["candles"] if c["complete"]
    ]
    return pd.DataFrame(records)


def parseCorrelated(mainPair, corrPair):
    _rawDataDir = str(Path(__file__).parent.parent / "raw_data")
    mainFiles = glob.glob(f"{_rawDataDir}/{mainPair}_*.json")
    corrFiles = glob.glob(f"{_rawDataDir}/{corrPair}_*.json")
    if len(mainFiles) != 1:
        raise FileNotFoundError(f"Expected 1 file for {mainPair} in raw_data/, found {len(mainFiles)}")
    if len(corrFiles) != 1:
        raise FileNotFoundError(f"Expected 1 file for {corrPair} in raw_data/, found {len(corrFiles)}")
    dfMain = _jsonToCloses(mainFiles[0])
    dfCorr = _jsonToCloses(corrFiles[0])
    return _computeCorrFeatures(dfMain, dfCorr, corrPair.split("_")[0].lower())


def parseLiveCorrelated(jsonMain, jsonCorr, corrPair):
    dfMain = _jsonToCloses(jsonMain)
    dfCorr = _jsonToCloses(jsonCorr)
    return _computeCorrFeatures(dfMain, dfCorr, corrPair.split("_")[0].lower())
