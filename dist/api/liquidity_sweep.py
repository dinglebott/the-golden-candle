# Deployment copy of 3_algo-backtest/liquidity_sweep.py.
# Returns at most one entry dict, fired within the last `n_active` H1 bars, else None.
import numpy as np
import pandas as pd

# === Sweep detection (H4 candle wicks beyond prior H4 swing, closes back inside) ===
SWEEP_MIN_DEPTH_ATR = 0.40
SWEEP_LOOKBACK = 20
SWEEP_CLOSE_PCT = 0.75

# === MSS trigger ===
MSS_TIMEOUT_BARS = 30
SL_BUFFER_ATR = 0.50

# === TP ===
TP_MIN_DIST = 1.2
TP_MAX_DIST = 6.0
TP_FALLBACK = 4.0

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


def _select_tp_bullish(entry, risk, trade_time, sh_times_np, sh_vals_np):
    valid_until = trade_time - np.timedelta64(8, 'h')
    mask = (sh_times_np <= valid_until) & (sh_vals_np > entry)
    if not mask.any():
        return entry + TP_FALLBACK * risk, "fallback"
    vals = sh_vals_np[mask]
    dists = vals - entry
    rng = (dists >= TP_MIN_DIST * risk) & (dists <= TP_MAX_DIST * risk)
    if not rng.any():
        return entry + TP_FALLBACK * risk, "fallback"
    return float(vals[rng].min()), "h4_swing"


def _select_tp_bearish(entry, risk, trade_time, sl_times_np, sl_vals_np):
    valid_until = trade_time - np.timedelta64(8, 'h')
    mask = (sl_times_np <= valid_until) & (sl_vals_np < entry)
    if not mask.any():
        return entry - TP_FALLBACK * risk, "fallback"
    vals = sl_vals_np[mask]
    dists = entry - vals
    rng = (dists >= TP_MIN_DIST * risk) & (dists <= TP_MAX_DIST * risk)
    if not rng.any():
        return entry - TP_FALLBACK * risk, "fallback"
    return float(vals[rng].max()), "h4_swing"


def get_entries(df, n_active=6):
    n = len(df)
    if n == 0:
        return None

    dt_utc = pd.to_datetime(df["time"], utc=True)
    h4_floor = dt_utc.dt.floor("4h")
    h4_floor_np = h4_floor.to_numpy()

    # === H4 OHLC + ATR ===
    h4 = pd.DataFrame({
        "open":  df.groupby(h4_floor)["open"].first(),
        "high":  df.groupby(h4_floor)["high"].max(),
        "low":   df.groupby(h4_floor)["low"].min(),
        "close": df.groupby(h4_floor)["close"].last(),
    }).sort_index()
    h4_atr_np = _atr(h4["high"], h4["low"], h4["close"], 14).to_numpy()
    h4_high_np = h4["high"].to_numpy()
    h4_low_np = h4["low"].to_numpy()
    h4_close_np = h4["close"].to_numpy()
    h4_idx_np = h4.index.to_numpy()
    h4_count = len(h4)

    # H4 swing fractals (1-bar)
    h4_sh_pos_l, h4_sh_vals_l = [], []
    h4_sl_pos_l, h4_sl_vals_l = [], []
    for j in range(1, h4_count - 1):
        if h4_high_np[j] > h4_high_np[j-1] and h4_high_np[j] > h4_high_np[j+1]:
            h4_sh_pos_l.append(j)
            h4_sh_vals_l.append(h4_high_np[j])
        if h4_low_np[j] < h4_low_np[j-1] and h4_low_np[j] < h4_low_np[j+1]:
            h4_sl_pos_l.append(j)
            h4_sl_vals_l.append(h4_low_np[j])
    h4_sh_pos = np.array(h4_sh_pos_l, dtype=np.int64)
    h4_sl_pos = np.array(h4_sl_pos_l, dtype=np.int64)
    h4_sh_vals = np.array(h4_sh_vals_l, dtype=np.float64)
    h4_sl_vals = np.array(h4_sl_vals_l, dtype=np.float64)
    h4_sh_times = h4_idx_np[h4_sh_pos] if len(h4_sh_pos) > 0 else np.array([], dtype='datetime64[ns]')
    h4_sl_times = h4_idx_np[h4_sl_pos] if len(h4_sl_pos) > 0 else np.array([], dtype='datetime64[ns]')

    # === H1 features ===
    h1_atr = _atr(df["high"], df["low"], df["close"], 14).shift(1).to_numpy()
    h_arr = df["high"].to_numpy()
    l_arr = df["low"].to_numpy()
    c_arr = df["close"].to_numpy()
    dt_arr = dt_utc.to_numpy()

    h1_sh = np.full(n, np.nan)
    h1_sl = np.full(n, np.nan)
    if n > 2:
        is_sh = (h_arr[1:-1] > h_arr[:-2]) & (h_arr[1:-1] > h_arr[2:])
        is_sl = (l_arr[1:-1] < l_arr[:-2]) & (l_arr[1:-1] < l_arr[2:])
        h1_sh[1:-1] = np.where(is_sh, h_arr[1:-1], np.nan)
        h1_sl[1:-1] = np.where(is_sl, l_arr[1:-1], np.nan)

    # H4 close indices in H1 space. Only treat the trailing bar as an H4 close
    # if its H4 bucket is complete (last hour of the bucket).
    same_as_next = h4_floor_np[:-1] == h4_floor_np[1:]
    h4_close_idx_h1 = np.where(~same_as_next)[0]
    if (dt_utc.iat[-1] - h4_floor.iat[-1]) >= pd.Timedelta(hours=3):
        h4_close_idx_h1 = np.append(h4_close_idx_h1, n - 1)
    h4_pos_at_close = np.array([
        np.searchsorted(h4_idx_np, h4_floor_np[i]) for i in h4_close_idx_h1
    ])

    # === Walk H4 closes; MSS scanned between consecutive H4 closes ===
    # Only the most recent entry that falls inside the active window is returned.
    armed = False
    is_long = False
    sweep_wick = np.nan
    sweep_idx = -1
    last_scan_pos = -1
    window_start = max(0, n - n_active)
    last_entry = None

    def try_mss(start_i, end_i):
        nonlocal armed, is_long, sweep_wick, sweep_idx, last_entry
        if not armed:
            return
        for ii in range(start_i, end_i):
            close_ii = c_arr[ii]
            if is_long and close_ii < sweep_wick:
                armed = False
                return
            if (not is_long) and close_ii > sweep_wick:
                armed = False
                return
            if ii - sweep_idx > MSS_TIMEOUT_BARS:
                armed = False
                return
            if is_long:
                rel_idx = -1
                for jj in range(ii - 1, sweep_idx, -1):
                    if not np.isnan(h1_sh[jj]):
                        rel_idx = jj
                        break
                if rel_idx >= 0 and close_ii > h1_sh[rel_idx]:
                    atr_m = h1_atr[ii]
                    if not np.isnan(atr_m):
                        sl = sweep_wick - SL_BUFFER_ATR * atr_m
                        risk = close_ii - sl
                        if risk > 0:
                            tp, tp_type = _select_tp_bullish(
                                close_ii, risk, dt_arr[ii],
                                h4_sh_times, h4_sh_vals)
                            if ii >= window_start:
                                last_entry = {
                                    "direction": "long",
                                    "entry": float(close_ii),
                                    "tp": float(tp),
                                    "sl": float(sl),
                                    "tp_type": tp_type,
                                    "time": str(df["time"].iat[ii]),
                                }
                    armed = False
                    return
            else:
                rel_idx = -1
                for jj in range(ii - 1, sweep_idx, -1):
                    if not np.isnan(h1_sl[jj]):
                        rel_idx = jj
                        break
                if rel_idx >= 0 and close_ii < h1_sl[rel_idx]:
                    atr_m = h1_atr[ii]
                    if not np.isnan(atr_m):
                        sl = sweep_wick + SL_BUFFER_ATR * atr_m
                        risk = sl - close_ii
                        if risk > 0:
                            tp, tp_type = _select_tp_bearish(
                                close_ii, risk, dt_arr[ii],
                                h4_sl_times, h4_sl_vals)
                            if ii >= window_start:
                                last_entry = {
                                    "direction": "short",
                                    "entry": float(close_ii),
                                    "tp": float(tp),
                                    "sl": float(sl),
                                    "tp_type": tp_type,
                                    "time": str(df["time"].iat[ii]),
                                }
                    armed = False
                    return

    for ev_i in range(len(h4_close_idx_h1)):
        h1_close_i = int(h4_close_idx_h1[ev_i])
        h4_pos = int(h4_pos_at_close[ev_i])
        if h4_pos < 0 or h4_pos >= h4_count:
            continue

        if armed and last_scan_pos < h1_close_i:
            try_mss(max(last_scan_pos + 1, sweep_idx + 1), h1_close_i + 1)
        last_scan_pos = h1_close_i

        atr_v = h4_atr_np[h4_pos]
        if np.isnan(atr_v) or atr_v == 0:
            continue
        h4_high_v = h4_high_np[h4_pos]
        h4_low_v = h4_low_np[h4_pos]
        h4_close_v = h4_close_np[h4_pos]
        candle_range = h4_high_v - h4_low_v
        if candle_range <= 0:
            continue

        lookback_pos_min = h4_pos - SWEEP_LOOKBACK

        recent_sl_val = np.nan
        if len(h4_sl_pos) > 0:
            hi = np.searchsorted(h4_sl_pos, h4_pos - 1)
            for k in range(hi - 1, -1, -1):
                sp = h4_sl_pos[k]
                if sp < lookback_pos_min:
                    break
                sv = h4_sl_vals[k]
                start = sp + 2
                if start < h4_pos:
                    if (h4_close_np[start:h4_pos] < sv).any():
                        continue
                recent_sl_val = sv
                break

        recent_sh_val = np.nan
        if len(h4_sh_pos) > 0:
            hi = np.searchsorted(h4_sh_pos, h4_pos - 1)
            for k in range(hi - 1, -1, -1):
                sp = h4_sh_pos[k]
                if sp < lookback_pos_min:
                    break
                sv = h4_sh_vals[k]
                start = sp + 2
                if start < h4_pos:
                    if (h4_close_np[start:h4_pos] > sv).any():
                        continue
                recent_sh_val = sv
                break

        bull_sweep = False
        bear_sweep = False
        if TRADE_LONGS and not np.isnan(recent_sl_val):
            depth = recent_sl_val - h4_low_v
            close_above = h4_close_v > recent_sl_val
            close_in_upper = (h4_close_v - h4_low_v) / candle_range >= SWEEP_CLOSE_PCT
            if depth >= SWEEP_MIN_DEPTH_ATR * atr_v and close_above and close_in_upper:
                bull_sweep = True
        if TRADE_SHORTS and not np.isnan(recent_sh_val):
            depth = h4_high_v - recent_sh_val
            close_below = h4_close_v < recent_sh_val
            close_in_lower = (h4_high_v - h4_close_v) / candle_range >= SWEEP_CLOSE_PCT
            if depth >= SWEEP_MIN_DEPTH_ATR * atr_v and close_below and close_in_lower:
                bear_sweep = True

        if bull_sweep and bear_sweep:
            bull_depth = (recent_sl_val - h4_low_v) / atr_v
            bear_depth = (h4_high_v - recent_sh_val) / atr_v
            if bull_depth >= bear_depth:
                bear_sweep = False
            else:
                bull_sweep = False

        if bull_sweep:
            armed = True
            is_long = True
            sweep_wick = h4_low_v
            sweep_idx = h1_close_i
        elif bear_sweep:
            armed = True
            is_long = False
            sweep_wick = h4_high_v
            sweep_idx = h1_close_i

    if armed:
        try_mss(max(last_scan_pos + 1, sweep_idx + 1), n)

    return last_entry
