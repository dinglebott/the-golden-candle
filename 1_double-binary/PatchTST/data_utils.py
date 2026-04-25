import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from data_processing import dataparser
from classes import SplitData


ROOT_DIR = Path(__file__).parent.parent.parent
RAW_DATA_DIR = ROOT_DIR / "raw_data"

def infer_available_pairs(granularity):
    pairs = []
    for file_path in sorted(RAW_DATA_DIR.glob(f"*_{granularity}_*.json")):
        pairs.append(file_path.name.split(f"_{granularity}_")[0])
    return pairs


def resolve_target_instrument(env):
    return env["instrument"]


def resolve_pretrain_pairs(env):
    pretrain_pairs = env.get("patchtst", {}).get("pretrain_pairs")
    if pretrain_pairs:
        return pretrain_pairs
    return infer_available_pairs(env["granularity"])


def get_corr_feature_names(corr_pair):
    if isinstance(corr_pair, str):
        prefix = corr_pair.split("_")[0].lower()
        return [
            f"{prefix}_close_return",
            f"{prefix}_return_spread",
            f"{prefix}_rolling_corr_20",
            f"{prefix}_cross_zscore",
        ]
    if corr_pair == 0:
        return []
    raise ValueError("corr_pair must be 0 or a valid currency pair string")


def load_raw_dataframe(instrument, granularity, year_now):
    raw_path = RAW_DATA_DIR / f"{instrument}_{granularity}_{year_now - 21}-01-01_{year_now}-04-01.json"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing raw data file: {raw_path}")
    return dataparser.parseData(raw_path)


def prepare_task_dataframe(
    instrument,
    granularity,
    year_now,
    k_value,
    n_value,
    binary,
    corr_pair=0,
    include_corr_features=False,
):
    df = load_raw_dataframe(instrument, granularity, year_now)

    if include_corr_features and isinstance(corr_pair, str):
        df_corr = dataparser.parseCorrelated(instrument, corr_pair)
        df = df.merge(df_corr, on="time", how="inner")
    elif include_corr_features and corr_pair != 0:
        raise ValueError("corr_pair must be 0 or a valid currency pair string")

    if binary == 0:
        df = dataparser.addGateTarget(df, k_value, n_value)
        df["target_mask"] = 1.0
    elif binary == 1:
        gate_df = dataparser.addGateTarget(df.copy(), k_value, n_value)
        direction_df = dataparser.addDirectionTarget(df.copy(), k_value, n_value)[["time", "target"]]
        direction_df = direction_df.rename(columns={"target": "direction_target"})

        df = gate_df.merge(direction_df, on="time", how="left")
        df["target_mask"] = df["direction_target"].notna().astype(np.float32)
        df["target"] = df["direction_target"].fillna(0).astype(np.int64)
        df.drop(columns=["direction_target"], inplace=True)
    else:
        raise ValueError("binary must be 0 or 1")

    df.reset_index(drop=True, inplace=True)
    return df


def make_split_bundle(df, core_features, corr_features, lookback, train_split, val_split, purge_gap):
    core_matrix = df[core_features].to_numpy(dtype=np.float32)
    corr_matrix = df[corr_features].to_numpy(dtype=np.float32) if corr_features else None
    targets = df["target"].to_numpy(dtype=np.float32)
    target_mask = df["target_mask"].to_numpy(dtype=np.float32)

    total_rows = len(df)
    min_target_idx = lookback - 1
    train_end = int(train_split * total_rows)
    val_end = int((train_split + val_split) * total_rows)

    if train_end <= min_target_idx:
        raise ValueError("Training split is too small for the configured lookback window")
    if val_end <= train_end or val_end >= total_rows:
        raise ValueError("Invalid train/validation split configuration")
    if train_end - purge_gap <= min_target_idx:
        raise ValueError("Training split is too small after applying the label purge gap")
    if val_end - purge_gap <= train_end:
        raise ValueError("Validation split is too small after applying the label purge gap")

    def build_indices(start_idx, end_idx):
        start_idx = max(start_idx, min_target_idx)
        if end_idx <= start_idx:
            raise ValueError("One of the dataset splits is empty after applying the lookback window")
        indices = np.arange(start_idx, end_idx, dtype=np.int64)
        masked_indices = indices[target_mask[indices] > 0.5]
        if len(masked_indices) == 0:
            raise ValueError("One of the dataset splits has no valid target samples after masking")
        return masked_indices

    return {
        "train": SplitData(core_matrix, corr_matrix, targets, build_indices(min_target_idx, train_end - purge_gap)),
        "val": SplitData(core_matrix, corr_matrix, targets, build_indices(train_end, val_end - purge_gap)),
        "test": SplitData(core_matrix, corr_matrix, targets, build_indices(val_end, total_rows)),
        "train_end": train_end,
        "val_end": val_end,
    }


def compute_train_stats(split_bundles, feature_key):
    train_segments = []
    for bundle in split_bundles:
        split = bundle["train"]
        features = split.core_features if feature_key == "core" else split.corr_features
        if features is None or features.shape[1] == 0:
            continue
        train_segments.append(features[:bundle["train_end"]])

    if not train_segments:
        empty = np.zeros((1, 0), dtype=np.float32)
        return empty, empty.copy()

    matrix = np.concatenate(train_segments, axis=0)
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)
