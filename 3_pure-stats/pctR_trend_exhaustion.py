import pandas as pd

THRESHOLD = 20 # width of the OB/OS zone
FAST_PERIOD = 21
SLOW_PERIOD = 112

def get_signals(df: pd.DataFrame):
    workingDf = pd.DataFrame()
    # Compute %R, range from -100 to 0
    fastHighest = df["high"].rolling(FAST_PERIOD).max()
    fastLowest = df["low"].rolling(FAST_PERIOD).min()
    slowHighest = df["high"].rolling(SLOW_PERIOD).max()
    slowLowest = df["low"].rolling(SLOW_PERIOD).min()
    fastR = (fastHighest - df["close"]) / (fastHighest - fastLowest) * -100
    slowR = (slowHighest - df["close"]) / (slowHighest - slowLowest) * -100
    # save to working dataframe
    workingDf["fast_pct_R"] = fastR.ewm(span=7, adjust=False).mean()
    workingDf["slow_pct_R"] = slowR.ewm(span=3, adjust=False).mean()
    # drop warmup period
    workingDf = workingDf[SLOW_PERIOD:]
    workingDf.dropna(inplace=True)

    # Boolean threshold logic
    obMask = (workingDf["fast_pct_R"] >= -THRESHOLD) & (workingDf["slow_pct_R"] >= -THRESHOLD)
    osMask = (workingDf["fast_pct_R"] <= -100 + THRESHOLD) & (workingDf["slow_pct_R"] <= -100 + THRESHOLD)
    # where is not OB/OS, but previous candle is OB/OS
    obReversal = ~obMask & obMask.shift(1).fillna(False)
    osReversal = ~osMask & osMask.shift(1).fillna(False)
    # convert to signal
    workingDf["signal"] = 0
    workingDf.loc[obReversal, "signal"] = -1
    workingDf.loc[osReversal, "signal"] = 1

    return workingDf["signal"]