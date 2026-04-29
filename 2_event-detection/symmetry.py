"""Mirror bearish setups onto bullish geometry so the model sees one symmetry class.

Patterns whose +1/-1 directions are exact structural mirrors (FVG, order block,
liquidity sweep) benefit from this: the encoder no longer has to spend capacity
learning two reflected copies of the same setup. Apply at the model-side build
step; the dataparser features remain raw and direction-agnostic.
"""
import numpy as np

# Sign-flips when direction == -1. Magnitude/regime features are not listed.
DIRECTIONAL_FEATURES = {
    "open_return", "high_return", "low_return", "close_return", "vol_return",
    "cum_return_3", "cum_return_6", "cum_return_12", "cum_return_24",
    "vol_adj_return_6", "return_zscore_24", "return_accel_3_12",
    "di_diff", "macd_hist",
    "trend_pressure_8", "momentum_consistency_8",
    "slow_pct_R", "fast_pct_R",
    "smooth_cross", "ema_cross",
    "dist_ema15", "dist_ema50", "dist_smooth14", "dist_smooth35",
    "ema15_slope_3", "ema50_slope_5",
    "range_pos_24", "close_in_bar", "bb_position",
    "close_lag1", "close_lag2", "close_lag3", "close_lag4",
}

# Pairs that swap (not flip) when direction == -1.
SWAP_PAIRS = [("breakout_dist_high_24", "breakout_dist_low_24"),
              ("upper_wick", "lower_wick")]


def build_flip_mask(features: list[str]) -> np.ndarray:
    """Return a (len(features),) array of +1 / -1 for sign-flippable features."""
    return np.array(
        [-1.0 if f in DIRECTIONAL_FEATURES else 1.0 for f in features],
        dtype=np.float32,
    )


def build_swap_indices(features: list[str]) -> list[tuple[int, int]]:
    """Return [(i, j), ...] for SWAP_PAIRS entries where both names are present."""
    pairs = []
    for a, b in SWAP_PAIRS:
        if a in features and b in features:
            pairs.append((features.index(a), features.index(b)))
    return pairs


def apply_flip(arr: np.ndarray, direction: int, flip_mask: np.ndarray,
               swap_indices: list[tuple[int, int]]) -> np.ndarray:
    """Mirror a feature array onto bullish geometry. No-op when direction == 1.

    arr shape: (..., n_features). Last axis indexes features; works for both
    sequence (seq_len, n_features) and per-row (n_features,) inputs.
    """
    if direction == 1:
        return arr
    arr = arr * flip_mask
    for i, j in swap_indices:
        arr[..., [i, j]] = arr[..., [j, i]]
    return arr


def apply_flip_df(event_df, directions):
    """In-place mirror of an event-row DataFrame. `directions` is a 1-D array of +1/-1."""
    sign = np.asarray(directions, dtype=np.float32)
    for col in event_df.columns.intersection(list(DIRECTIONAL_FEATURES)):
        event_df[col] = event_df[col].values * sign
    mask = sign == -1
    for a, b in SWAP_PAIRS:
        if a in event_df.columns and b in event_df.columns:
            a_vals = event_df[a].values.copy()
            event_df.loc[mask, a] = event_df.loc[mask, b].values
            event_df.loc[mask, b] = a_vals[mask]
    return event_df
