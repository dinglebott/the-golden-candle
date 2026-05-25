import numpy as np
import pandas as pd

# === EMD cycle model (H4) ===
EMD_PERIOD = 20             # bandpass centre, in H4 bars
EMD_DELTA = 0.3             # bandpass bandwidth (dimensionless)
EMD_SMOOTH_N = 50           # peak/valley smoothing window, in H4 bars
EMD_THRESHOLD_ATR = 0.5     # avg peak/valley magnitude must exceed this * H4 ATR to count as cyclic

# === TP midline (H4) ===
MIDLINE_EMA_PERIOD = 20     # H4 EMA used as the mean-reversion target

# === ATR (shared) ===
ATR_PERIOD = 14

# === Setup -> trigger window (H1) ===
SETUP_EXPIRY_BARS = 6       # H1 bars after H4 setup to wait for a pin

# === Pin bar definition (H1) ===
PIN_WICK_BODY_RATIO = 1.5       # rejection wick >= this * body
PIN_WICK_OPPOSING_RATIO = 1.5   # rejection wick >= this * opposing wick
PIN_CLOSE_FRACTION = 0.5        # close in upper/lower half of bar (bull/bear pin)

# === Risk ===
SL_BUFFER_ATR = 0.5         # SL placed this many H1 ATRs beyond the pin wick

# === Direction toggles ===
TRADE_LONGS = True
TRADE_SHORTS = True


def _wilder(s, period):
    return s.ewm(alpha=1/period, adjust=False).mean()


def _atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return _wilder(tr, period)


def _emd_bandpass(price, period, delta):
    beta = np.cos(2 * np.pi / period)
    gamma = 1 / np.cos(4 * np.pi * delta / period)
    alpha = gamma - np.sqrt(gamma ** 2 - 1)
    x = np.asarray(price, dtype=float)
    bp = np.zeros(len(x))
    for i in range(2, len(x)):
        bp[i] = (
            0.5 * (1 - alpha) * (x[i] - x[i-2])
            + beta * (1 + alpha) * bp[i-1]
            - alpha * bp[i-2]
        )
    return bp


def _emd_components(high, low, atr_arr):
    hl2 = (np.asarray(high, dtype=float) + np.asarray(low, dtype=float)) / 2
    bp = _emd_bandpass(hl2, EMD_PERIOD, EMD_DELTA)
    peak_series = np.zeros(len(bp))
    valley_series = np.zeros(len(bp))
    last_peak = 0.0
    last_valley = 0.0
    for i in range(2, len(bp)):
        if bp[i-1] > bp[i] and bp[i-1] > bp[i-2]:
            last_peak = bp[i-1]
        if bp[i-1] < bp[i] and bp[i-1] < bp[i-2]:
            last_valley = bp[i-1]
        peak_series[i] = last_peak
        valley_series[i] = last_valley
    avg_peak = pd.Series(peak_series).rolling(EMD_SMOOTH_N).mean().to_numpy()
    avg_valley = pd.Series(valley_series).rolling(EMD_SMOOTH_N).mean().to_numpy()
    threshold = EMD_THRESHOLD_ATR * np.asarray(atr_arr, dtype=float)
    with np.errstate(invalid="ignore"):
        cycle = (avg_peak > threshold) & (avg_valley < -threshold)
    cycle = np.where(np.isnan(avg_peak) | np.isnan(avg_valley) | np.isnan(threshold), False, cycle)
    return bp, avg_peak, avg_valley, cycle.astype(bool)


def _is_bullish_pin(o, h, l, c):
    rng = h - l
    if rng <= 0:
        return False
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    if body == 0:
        return lower_wick > upper_wick and (c - l) / rng >= PIN_CLOSE_FRACTION
    if lower_wick < PIN_WICK_BODY_RATIO * body:
        return False
    if lower_wick < PIN_WICK_OPPOSING_RATIO * upper_wick:
        return False
    if (c - l) / rng < PIN_CLOSE_FRACTION:
        return False
    return True


def _is_bearish_pin(o, h, l, c):
    rng = h - l
    if rng <= 0:
        return False
    body = abs(c - o)
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    if body == 0:
        return upper_wick > lower_wick and (h - c) / rng >= PIN_CLOSE_FRACTION
    if upper_wick < PIN_WICK_BODY_RATIO * body:
        return False
    if upper_wick < PIN_WICK_OPPOSING_RATIO * lower_wick:
        return False
    if (h - c) / rng < PIN_CLOSE_FRACTION:
        return False
    return True


def get_entries(df):
    n = len(df)
    entry_arr = np.full(n, np.nan)
    tp_arr = np.full(n, np.nan)
    sl_arr = np.full(n, np.nan)
    tp_type_arr = np.empty(n, dtype=object)

    dt_utc = pd.to_datetime(df["time"], utc=True)
    h4_floor = dt_utc.dt.floor("4h")
    h4_floor_np = h4_floor.to_numpy()

    # === H4 OHLC + features ===
    h4 = pd.DataFrame({
        "open":  df.groupby(h4_floor)["open"].first(),
        "high":  df.groupby(h4_floor)["high"].max(),
        "low":   df.groupby(h4_floor)["low"].min(),
        "close": df.groupby(h4_floor)["close"].last(),
    }).sort_index()
    h4_atr = _atr(h4["high"], h4["low"], h4["close"], ATR_PERIOD)
    h4_ema = h4["close"].ewm(span=MIDLINE_EMA_PERIOD, adjust=False).mean()
    bp, avg_peak, avg_valley, h4_regime = _emd_components(
        h4["high"], h4["low"], h4_atr.to_numpy()
    )

    h4_ema_np = h4_ema.to_numpy()
    h4_idx_np = h4.index.to_numpy()
    h4_count = len(h4)

    # === H1 features ===
    h1_atr_np = _atr(df["high"], df["low"], df["close"], ATR_PERIOD).shift(1).to_numpy()
    o_arr = df["open"].to_numpy()
    h_arr = df["high"].to_numpy()
    l_arr = df["low"].to_numpy()
    c_arr = df["close"].to_numpy()

    # Map each H4 bucket to its final H1 index (the bar that closes the bucket)
    same_as_next = h4_floor_np[:-1] == h4_floor_np[1:]
    h4_close_idx_h1 = np.where(~same_as_next)[0]
    h4_close_idx_h1 = np.append(h4_close_idx_h1, n - 1)
    h4_pos_at_close = np.array([
        np.searchsorted(h4_idx_np, h4_floor_np[i]) for i in h4_close_idx_h1
    ])

    # === State machine: walk H4 closes, scan intervening H1 bars for a pin trigger ===
    armed = False
    is_long = False
    setup_tp = np.nan        # snapshot of H4 EMA at arming time, used as TP
    setup_idx = -1           # H1 index where setup was armed
    last_scan_pos = -1
    # Per-direction lock: blocks re-arming on the same cycle extreme until bp
    # crosses to the opposite half (half-cycle has elapsed)
    bull_locked = False
    bear_locked = False

    def scan_pin(start_i, end_i):
        nonlocal armed, is_long, setup_tp, setup_idx
        if not armed:
            return
        for ii in range(start_i, end_i):
            if ii - setup_idx > SETUP_EXPIRY_BARS:
                armed = False
                return
            o, h, l, c = o_arr[ii], h_arr[ii], l_arr[ii], c_arr[ii]
            atr_h1 = h1_atr_np[ii]
            if np.isnan(atr_h1):
                continue
            if is_long:
                if _is_bullish_pin(o, h, l, c):
                    sl = l - SL_BUFFER_ATR * atr_h1
                    risk = c - sl
                    if risk > 0 and setup_tp > c:
                        entry_arr[ii] = c
                        tp_arr[ii] = setup_tp
                        sl_arr[ii] = sl
                        tp_type_arr[ii] = "h4_midline"
                    armed = False
                    return
            else:
                if _is_bearish_pin(o, h, l, c):
                    sl = h + SL_BUFFER_ATR * atr_h1
                    risk = sl - c
                    if risk > 0 and setup_tp < c:
                        entry_arr[ii] = c
                        tp_arr[ii] = setup_tp
                        sl_arr[ii] = sl
                        tp_type_arr[ii] = "h4_midline"
                    armed = False
                    return

    for ev_i in range(len(h4_close_idx_h1)):
        h1_close_i = h4_close_idx_h1[ev_i]
        h4_pos = h4_pos_at_close[ev_i]
        if h4_pos < 0 or h4_pos >= h4_count:
            continue

        # Scan H1 bars from previous H4 close (exclusive) up to and including this one
        if armed and last_scan_pos < h1_close_i:
            scan_pin(max(last_scan_pos + 1, setup_idx + 1), h1_close_i + 1)
        last_scan_pos = h1_close_i

        bp_v = bp[h4_pos]
        bp_prev = bp[h4_pos - 1] if h4_pos > 0 else bp_v
        # Release per-direction lock once bp has traversed to the opposite half
        if bull_locked and bp_v > 0:
            bull_locked = False
        if bear_locked and bp_v < 0:
            bear_locked = False

        # Setup checks at the H4 close
        if not h4_regime[h4_pos]:
            continue
        ema_v = h4_ema_np[h4_pos]
        avg_peak_v = avg_peak[h4_pos]
        avg_valley_v = avg_valley[h4_pos]
        if np.isnan(ema_v) or np.isnan(avg_peak_v) or np.isnan(avg_valley_v):
            continue

        # Require bp to have turned away from the extreme on this H4 close
        # (rising for longs, falling for shorts) — fade exhaustion, not extension.
        bull_setup = (TRADE_LONGS and not bull_locked
                      and bp_v <= avg_valley_v
                      and bp_v > bp_prev)
        bear_setup = (TRADE_SHORTS and not bear_locked
                      and bp_v >= avg_peak_v
                      and bp_v < bp_prev)

        if bull_setup and bear_setup:
            # avg_peak <= 0 <= avg_valley shouldn't happen under the regime gate, but guard anyway
            continue
        if bull_setup:
            armed = True
            is_long = True
            setup_tp = ema_v
            setup_idx = h1_close_i
            bull_locked = True
        elif bear_setup:
            armed = True
            is_long = False
            setup_tp = ema_v
            setup_idx = h1_close_i
            bear_locked = True

    # Final tail scan for any setup still armed after the last H4 close
    if armed:
        scan_pin(max(last_scan_pos + 1, setup_idx + 1), n)

    entries = pd.DataFrame({
        "entry": entry_arr,
        "tp": tp_arr,
        "sl": sl_arr,
        "tp_type": tp_type_arr,
    }, index=df.index)
    return entries
