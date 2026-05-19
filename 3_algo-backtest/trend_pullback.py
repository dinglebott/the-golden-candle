import numpy as np
import pandas as pd
from tqdm import tqdm

TRADE_BEAR_TRENDS = False  # shorts underperform on this strategy/pair (see tuning_log.md)

ADX_MIN = 25
RSI_LONG_MIN = 40
RSI_SHORT_MAX = 60
TOLERANCE = 0.20 # zone width around pullback levels, in H4 ATR14
SL_BUFFER = 1.5 # extra buffer below pivot low, in H1 ATR14
TP_MIN_DIST = 1.5 # min distance from entry to TP, in multiples of risk
TP_MAX_DIST = 6.0
TP_FALLBACK = 2.0 # if no valid TP candidate, use this multiple of risk
SWING_FRACTAL = 1


def _wilder(s, period):
    return s.ewm(alpha=1/period, adjust=False).mean()


def _compute_adx(high, low, close, period=14):
    up = high.diff()
    dn = -low.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=high.index)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = _wilder(tr, period)
    plus_di = 100 * _wilder(plus_dm, period) / atr
    minus_di = 100 * _wilder(minus_dm, period) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = _wilder(dx, period)
    return adx, plus_di, minus_di


def _rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def _atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return _wilder(tr, period)


def _select_tp_bullish(entry, risk, trade_time, sh_times, sh_vals, exclude_time=None):
    # H4 swing high at bar t is confirmed at t + 8h (1-bar fractal on H4).
    # exclude_time skips the impulse swing high so TPs aim for new territory.
    best = None
    for t, h in zip(sh_times, sh_vals):
        if t == exclude_time:
            continue
        if t + pd.Timedelta(hours=8) > trade_time or h <= entry:
            continue
        dist = h - entry
        if not (TP_MIN_DIST * risk <= dist <= TP_MAX_DIST * risk):
            continue
        if best is None or h < best:
            best = h
    if best is not None:
        return best
    return entry + TP_FALLBACK * risk


def _select_tp_bearish(entry, risk, trade_time, sl_times, sl_vals, exclude_time=None):
    best = None
    for t, l in zip(sl_times, sl_vals):
        if t == exclude_time:
            continue
        if t + pd.Timedelta(hours=8) > trade_time or l >= entry:
            continue
        dist = entry - l
        if not (TP_MIN_DIST * risk <= dist <= TP_MAX_DIST * risk):
            continue
        if best is None or l > best:
            best = l
    if best is not None:
        return best
    return entry - TP_FALLBACK * risk


def get_entries(df):
    entries = pd.DataFrame({"entry": np.nan, "tp": np.nan, "sl": np.nan}, index=df.index)

    dt_utc = pd.to_datetime(df["time"], utc=True)
    h4_floor = dt_utc.dt.floor("4h")
    daily_floor = dt_utc.dt.floor("D")

    # ---------- H4 OHLC + features ----------
    h4 = pd.DataFrame({
        "open":  df.groupby(h4_floor)["open"].first(),
        "high":  df.groupby(h4_floor)["high"].max(),
        "low":   df.groupby(h4_floor)["low"].min(),
        "close": df.groupby(h4_floor)["close"].last(),
    }).sort_index()
    h4_ema20 = h4["close"].ewm(span=20, adjust=False).mean()
    h4_rsi = _rsi(h4["close"], 14)
    h4_atr_h4 = _atr(h4["high"], h4["low"], h4["close"], 14)

    # H4 swing fractals (1-bar)
    hh = h4["high"].values
    ll = h4["low"].values
    h4_idx = h4.index.tolist()
    h4_sh_times, h4_sh_vals = [], []
    h4_sl_times, h4_sl_vals = [], []
    for j in range(1, len(hh) - 1):
        if hh[j] > hh[j-1] and hh[j] > hh[j+1]:
            h4_sh_times.append(h4_idx[j])
            h4_sh_vals.append(hh[j])
        if ll[j] < ll[j-1] and ll[j] < ll[j+1]:
            h4_sl_times.append(h4_idx[j])
            h4_sl_vals.append(ll[j])

    # ---------- Daily OHLC + features (shifted by 1 daily bar; daily hasn't closed yet at H4 close) ----------
    daily = pd.DataFrame({
        "open":  df.groupby(daily_floor)["open"].first(),
        "high":  df.groupby(daily_floor)["high"].max(),
        "low":   df.groupby(daily_floor)["low"].min(),
        "close": df.groupby(daily_floor)["close"].last(),
    }).sort_index()
    d_ema50 = daily["close"].ewm(span=50, adjust=False).mean().shift(1)
    d_ema200 = daily["close"].ewm(span=200, adjust=False).mean().shift(1)
    d_adx, d_pdi, d_mdi = _compute_adx(daily["high"], daily["low"], daily["close"], 14)
    d_adx = d_adx.shift(1)
    d_pdi = d_pdi.shift(1)
    d_mdi = d_mdi.shift(1)

    # ---------- H1 features ----------
    h1_atr = _atr(df["high"], df["low"], df["close"], 14).shift(1)

    # H1 swing fractals (1-bar) as positional NaN-arrays
    h_arr = df["high"].to_numpy()
    l_arr = df["low"].to_numpy()
    c_arr = df["close"].to_numpy()
    h1_sh = np.full(len(df), np.nan)
    h1_sl = np.full(len(df), np.nan)
    if len(df) > 2:
        is_sh = (h_arr[1:-1] > h_arr[:-2]) & (h_arr[1:-1] > h_arr[2:])
        is_sl = (l_arr[1:-1] < l_arr[:-2]) & (l_arr[1:-1] < l_arr[2:])
        h1_sh[1:-1] = np.where(is_sh, h_arr[1:-1], np.nan)
        h1_sl[1:-1] = np.where(is_sl, l_arr[1:-1], np.nan)

    # H4 close marker: last H1 bar of each H4 bucket
    h4_close_mask = h4_floor.ne(h4_floor.shift(-1)).to_numpy()

    # Map each H4 bucket timestamp -> first H1 index inside it
    h4_first_idx = {}
    prev = None
    for i in range(len(df)):
        ts = h4_floor.iat[i]
        if ts != prev:
            h4_first_idx[ts] = i
            prev = ts

    # ---------- Most recent H4 impulse pair ----------
    def last_impulse_bullish(h4_time):
        sh_i = None
        for k in range(len(h4_sh_times) - 1, -1, -1):
            if h4_sh_times[k] < h4_time:
                sh_i = k
                break
        if sh_i is None:
            return None
        sh_time, sh_val = h4_sh_times[sh_i], h4_sh_vals[sh_i]
        sl_time, sl_val = None, None
        for k in range(len(h4_sl_times) - 1, -1, -1):
            if h4_sl_times[k] < sh_time:
                sl_time, sl_val = h4_sl_times[k], h4_sl_vals[k]
                break
        if sl_time is None:
            return None
        prior_sh = None
        for k in range(len(h4_sh_times) - 1, -1, -1):
            if h4_sh_times[k] < sl_time:
                prior_sh = h4_sh_vals[k]
                break
        return sh_time, sh_val, sl_time, sl_val, prior_sh

    def last_impulse_bearish(h4_time):
        sl_i = None
        for k in range(len(h4_sl_times) - 1, -1, -1):
            if h4_sl_times[k] < h4_time:
                sl_i = k
                break
        if sl_i is None:
            return None
        sl_time, sl_val = h4_sl_times[sl_i], h4_sl_vals[sl_i]
        sh_time, sh_val = None, None
        for k in range(len(h4_sh_times) - 1, -1, -1):
            if h4_sh_times[k] < sl_time:
                sh_time, sh_val = h4_sh_times[k], h4_sh_vals[k]
                break
        if sh_time is None:
            return None
        prior_sl = None
        for k in range(len(h4_sl_times) - 1, -1, -1):
            if h4_sl_times[k] < sh_time:
                prior_sl = h4_sl_vals[k]
                break
        return sl_time, sl_val, sh_time, sh_val, prior_sl

    # ---------- State machine ----------
    armed = False
    direction = None
    invalidation_level = None
    impulse_high_time = None
    impulse_low_time = None
    impulse_start_idx = None  # H1 index where pullback starts
    locked_impulse = None     # (low_time, high_time, dir) — already traded

    dt_arr = dt_utc.to_numpy()

    for i in tqdm(range(len(df)), desc="trend_pullback", unit="bar"):
        # ----- Invalidation (H1 close beyond pullback extreme) -----
        if armed:
            if direction == "long" and c_arr[i] < invalidation_level:
                armed = False
            elif direction == "short" and c_arr[i] > invalidation_level:
                armed = False

        # ----- BOS trigger -----
        if armed:
            current_time = dt_arr[i]
            close_i = c_arr[i]

            if direction == "long":
                rel_idx = None
                for j in range(i - 1, impulse_start_idx - 1, -1):
                    if not np.isnan(h1_sh[j]):
                        rel_idx = j
                        break
                if rel_idx is not None and close_i > h1_sh[rel_idx]:
                    if i > rel_idx + 1:
                        pivot_low = l_arr[rel_idx + 1 : i].min()
                    else:
                        pivot_low = l_arr[rel_idx]
                    atr_m = h1_atr.iat[i]
                    if not pd.isna(atr_m):
                        sl = pivot_low - SL_BUFFER * atr_m
                        risk = close_i - sl
                        if risk > 0:
                            tp = _select_tp_bullish(close_i, risk, current_time,
                                                    h4_sh_times, h4_sh_vals,
                                                    exclude_time=impulse_high_time)
                            entries.at[i, "entry"] = close_i
                            entries.at[i, "tp"] = tp
                            entries.at[i, "sl"] = sl
                            locked_impulse = (impulse_low_time, impulse_high_time, "long")
                            armed = False
            else:
                rel_idx = None
                for j in range(i - 1, impulse_start_idx - 1, -1):
                    if not np.isnan(h1_sl[j]):
                        rel_idx = j
                        break
                if rel_idx is not None and close_i < h1_sl[rel_idx]:
                    if i > rel_idx + 1:
                        pivot_high = h_arr[rel_idx + 1 : i].max()
                    else:
                        pivot_high = h_arr[rel_idx]
                    atr_m = h1_atr.iat[i]
                    if not pd.isna(atr_m):
                        sl = pivot_high + SL_BUFFER * atr_m
                        risk = sl - close_i
                        if risk > 0:
                            tp = _select_tp_bearish(close_i, risk, current_time,
                                                    h4_sl_times, h4_sl_vals,
                                                    exclude_time=impulse_low_time)
                            entries.at[i, "entry"] = close_i
                            entries.at[i, "tp"] = tp
                            entries.at[i, "sl"] = sl
                            locked_impulse = (impulse_high_time, impulse_low_time, "short")
                            armed = False

        # ----- Arm / refresh at H4 close -----
        if not h4_close_mask[i]:
            continue

        current_h4 = h4_floor.iat[i]
        current_d = daily_floor.iat[i]

        adx_v = d_adx.get(current_d)
        pdi_v = d_pdi.get(current_d)
        mdi_v = d_mdi.get(current_d)
        e50_v = d_ema50.get(current_d)
        e200_v = d_ema200.get(current_d)
        if any(pd.isna(x) for x in [adx_v, pdi_v, mdi_v, e50_v, e200_v]):
            armed = False
            continue

        h4_close_v = h4["close"].get(current_h4)
        h4_low_v = h4["low"].get(current_h4)
        h4_high_v = h4["high"].get(current_h4)
        e20_v = h4_ema20.get(current_h4)
        rsi_v = h4_rsi.get(current_h4)
        atr_h_v = h4_atr_h4.get(current_h4)
        if any(pd.isna(x) for x in [h4_close_v, e20_v, rsi_v, atr_h_v]):
            armed = False
            continue

        bull_trend = (adx_v >= ADX_MIN and pdi_v > mdi_v
                      and e50_v > e200_v and h4_close_v > e50_v
                      and rsi_v > RSI_LONG_MIN)
        bear_trend = (TRADE_BEAR_TRENDS and adx_v >= ADX_MIN and mdi_v > pdi_v
                      and e50_v < e200_v and h4_close_v < e50_v
                      and rsi_v < RSI_SHORT_MAX)

        if not (bull_trend or bear_trend):
            armed = False
            continue

        tol = TOLERANCE * atr_h_v

        if bull_trend:
            imp = last_impulse_bullish(current_h4)
            if imp is None:
                if armed and direction == "long":
                    armed = False
                continue
            sh_time, sh_val, sl_time, sl_val, prior_sh = imp
            if armed and (direction != "long" or impulse_high_time != sh_time):
                armed = False
            if locked_impulse == (sl_time, sh_time, "long"):
                continue
            leg = sh_val - sl_val
            if leg <= 0:
                continue
            fib_618 = sh_val - 0.618 * leg
            fib_382 = sh_val - 0.382 * leg
            candidates = []
            if prior_sh is not None and sl_val < prior_sh < sh_val:
                candidates.append(prior_sh)
            candidates += [fib_618, fib_382, e20_v]
            if any(lvl - tol <= h4_low_v <= lvl for lvl in candidates):
                armed = True
                direction = "long"
                invalidation_level = h4_low_v
                impulse_high_time = sh_time
                impulse_low_time = sl_time
                next_h4_start = sh_time + pd.Timedelta(hours=4)
                impulse_start_idx = h4_first_idx.get(next_h4_start, i)

        else:  # bear_trend
            imp = last_impulse_bearish(current_h4)
            if imp is None:
                if armed and direction == "short":
                    armed = False
                continue
            sl_time, sl_val, sh_time, sh_val, prior_sl = imp
            if armed and (direction != "short" or impulse_low_time != sl_time):
                armed = False
            if locked_impulse == (sh_time, sl_time, "short"):
                continue
            leg = sh_val - sl_val
            if leg <= 0:
                continue
            fib_618 = sl_val + 0.618 * leg
            fib_382 = sl_val + 0.382 * leg
            candidates = []
            if prior_sl is not None and sl_val < prior_sl < sh_val:
                candidates.append(prior_sl)
            candidates += [fib_618, fib_382, e20_v]
            if any(lvl <= h4_high_v <= lvl + tol for lvl in candidates):
                armed = True
                direction = "short"
                invalidation_level = h4_high_v
                impulse_high_time = sh_time
                impulse_low_time = sl_time
                next_h4_start = sl_time + pd.Timedelta(hours=4)
                impulse_start_idx = h4_first_idx.get(next_h4_start, i)

    return entries
