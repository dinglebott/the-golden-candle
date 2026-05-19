import numpy as np
import pandas as pd
from tqdm import tqdm

# === Sweep detection (H4 candle wicks beyond prior H4 swing, closes back inside) ===
SWEEP_MIN_DEPTH_ATR = 0.40  # min wick beyond swing, in H4 ATR14
SWEEP_LOOKBACK = 20         # max H4 bars back to consider for the swept swing
SWEEP_CLOSE_PCT = 0.75      # close must be in this fraction of the candle's range on the recovery side

# === MSS trigger ===
MSS_TIMEOUT_BARS = 30       # H1 bars after sweep to wait for MSS before disarming
SL_BUFFER_ATR = 0.50        # extra SL buffer beyond sweep wick, in H1 ATR14

# === TP ===
TP_MIN_DIST = 1.2           # min TP distance, in multiples of risk
TP_MAX_DIST = 6.0           # max TP distance, in multiples of risk
TP_FALLBACK = 4.0           # used if no valid H4 swing target

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


def _select_tp_bullish(entry, risk, trade_time, sh_times_np, sh_vals_np):
    # sh_times_np is np.datetime64[ns]; require swing confirmed before entry (+ 8h)
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


def get_entries(df):
    n = len(df)
    entry_arr = np.full(n, np.nan)
    tp_arr = np.full(n, np.nan)
    sl_arr = np.full(n, np.nan)
    tp_type_arr = np.empty(n, dtype=object)

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
    h4_open_np = h4["open"].to_numpy()
    h4_high_np = h4["high"].to_numpy()
    h4_low_np = h4["low"].to_numpy()
    h4_close_np = h4["close"].to_numpy()
    h4_idx_np = h4.index.to_numpy()
    h4_count = len(h4)

    # H4 swing fractals (1-bar)
    h4_sh_pos = []
    h4_sh_vals_l = []
    h4_sl_pos = []
    h4_sl_vals_l = []
    for j in range(1, h4_count - 1):
        if h4_high_np[j] > h4_high_np[j-1] and h4_high_np[j] > h4_high_np[j+1]:
            h4_sh_pos.append(j)
            h4_sh_vals_l.append(h4_high_np[j])
        if h4_low_np[j] < h4_low_np[j-1] and h4_low_np[j] < h4_low_np[j+1]:
            h4_sl_pos.append(j)
            h4_sl_vals_l.append(h4_low_np[j])
    h4_sh_pos = np.array(h4_sh_pos, dtype=np.int64)
    h4_sl_pos = np.array(h4_sl_pos, dtype=np.int64)
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

    # Map each H4 bucket to last H1 index (close index)
    # h4_floor strictly increases across the series; find each transition
    same_as_next = h4_floor_np[:-1] == h4_floor_np[1:]
    h4_close_idx_h1 = np.where(~same_as_next)[0]  # H1 positions where the next bar starts a new H4 bucket
    # last bar always closes the last H4 bucket
    h4_close_idx_h1 = np.append(h4_close_idx_h1, n - 1)
    # corresponding h4 bucket position for each close-of-bucket H1 index
    h4_pos_at_close = np.array([
        np.searchsorted(h4_idx_np, h4_floor_np[i]) for i in h4_close_idx_h1
    ])

    # === Walk through H4 closes only ===
    # Maintain a single "armed" state across H4 closes. Between consecutive H4 closes, scan H1
    # bars for MSS. At each H4 close, detect sweep — if armed in opposite direction or sweep is
    # newer, replace the armed state (this matches the prior bar-by-bar implementation).

    iterator = range(len(h4_close_idx_h1))

    armed = False
    is_long = False
    sweep_wick = np.nan
    sweep_idx = -1

    last_scan_pos = -1  # H1 index up through which we've already scanned for MSS

    def try_mss(start_i, end_i):
        # scan H1 bars (start_i .. end_i exclusive) for MSS; return True if entered or invalidated
        nonlocal armed, is_long, sweep_wick, sweep_idx
        if not armed:
            return
        for ii in range(start_i, end_i):
            close_ii = c_arr[ii]
            # Invalidation: close back through the sweep wick
            if is_long and close_ii < sweep_wick:
                armed = False
                return
            if (not is_long) and close_ii > sweep_wick:
                armed = False
                return
            # Timeout
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
                            entry_arr[ii] = close_ii
                            tp_arr[ii] = tp
                            sl_arr[ii] = sl
                            tp_type_arr[ii] = tp_type
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
                            entry_arr[ii] = close_ii
                            tp_arr[ii] = tp
                            sl_arr[ii] = sl
                            tp_type_arr[ii] = tp_type
                    armed = False
                    return

    for ev_i in iterator:
        h1_close_i = h4_close_idx_h1[ev_i]
        h4_pos = h4_pos_at_close[ev_i]
        if h4_pos < 0 or h4_pos >= h4_count:
            continue

        # MSS scan from previous-H4-close+1 up to and including the current bar (the H4 close
        # bar itself is also an H1 bar where MSS could fire BEFORE the new sweep is detected).
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

    # final tail scan after last H4 close
    if armed:
        try_mss(max(last_scan_pos + 1, sweep_idx + 1), n)

    entries = pd.DataFrame({
        "entry": entry_arr,
        "tp": tp_arr,
        "sl": sl_arr,
        "tp_type": tp_type_arr,
    }, index=df.index)
    return entries
