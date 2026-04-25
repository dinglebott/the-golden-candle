import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from classes import PatchTST, WindowedTimeSeriesDataset
from data_utils import get_corr_feature_names, make_split_bundle, prepare_task_dataframe


TASK_CONFIG = {
    0: {"mode_name": "gate"},
    1: {"mode_name": "dir"},
}


def load_env():
    with open(Path(__file__).parent.parent / "env.json", "r") as file:
        return json.load(file)


def load_task_features(version, mode_name):
    config_path = Path(__file__).parent / f"model_configs/v{version}/{mode_name}_features.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing feature config: {config_path}")
    with open(config_path, "r") as file:
        return json.load(file)["features"]


def build_test_loader(split_bundle, config, core_mean, core_std, corr_mean, corr_std):
    dataset = WindowedTimeSeriesDataset(
        split_bundle["test"],
        config["lookback"],
        core_mean,
        core_std,
        corr_mean,
        corr_std,
    )
    return DataLoader(
        dataset,
        batch_size=config["finetune_batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )


def evaluate_mcc(model, loader, device):
    model.eval()
    probabilities = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            core_inputs = batch["core_inputs"].to(device)
            corr_inputs = batch["corr_inputs"].to(device)
            logits = model(core_inputs, corr_inputs)
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(batch["target"].cpu().numpy())

    y_prob = np.concatenate(probabilities)
    y_true = np.concatenate(targets).astype(np.int64)
    y_pred = (y_prob >= 0.5).astype(np.int64)
    return float(matthews_corrcoef(y_true, y_pred))


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    env = load_env()
    patchtst_env = env["patchtst"]

    binary = env["binary"]
    if binary not in TASK_CONFIG:
        raise ValueError("binary must be 0 or 1")

    task = TASK_CONFIG[binary]
    version = patchtst_env["use_version"]
    instrument = env["instrument"]
    granularity = env["granularity"]
    year_now = env["year_now"]
    k_value = env["k_value"]
    purge_gap = env["n_value"]
    train_split = env["train_split"]
    val_split = env["val_split"]
    corr_pair = env["corr_pair"]

    feature_list = load_task_features(version, task["mode_name"])
    corr_features = get_corr_feature_names(corr_pair) if isinstance(corr_pair, str) else []

    checkpoint_path = (
        Path(__file__).parent
        / f"models/{instrument}/{task['mode_name']}_PatchTST_{instrument}_{granularity}_{year_now}_v{version}.pt"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]

    saved_features = checkpoint.get("core_features")
    if saved_features is not None and list(saved_features) != list(feature_list):
        raise ValueError(
            "Configured feature list does not match the saved model checkpoint. "
            "Use the matching use_version or retrain the model."
        )

    normalization = checkpoint["normalization"]
    core_mean = np.asarray(normalization["core_mean"], dtype=np.float32)[None, :]
    core_std = np.asarray(normalization["core_std"], dtype=np.float32)[None, :]
    corr_mean = np.asarray(normalization.get("corr_mean", []), dtype=np.float32)[None, :]
    corr_std = np.asarray(normalization.get("corr_std", []), dtype=np.float32)[None, :]

    device_name = env.get("device", "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested in env.json but not available; falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    model = PatchTST(len(feature_list), len(corr_features), config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    base_df = prepare_task_dataframe(
        instrument,
        granularity,
        year_now,
        k_value,
        purge_gap,
        binary,
        corr_pair=corr_pair,
        include_corr_features=bool(corr_features),
    )
    base_bundle = make_split_bundle(
        base_df,
        feature_list,
        corr_features,
        config["lookback"],
        train_split,
        val_split,
        purge_gap,
    )
    baseline_loader = build_test_loader(base_bundle, config, core_mean, core_std, corr_mean, corr_std)
    baseline_mcc = evaluate_mcc(model, baseline_loader, device)
    print(f"Baseline test MCC: {baseline_mcc:.6f}")

    results = {}
    test_start = base_bundle["val_end"]
    test_end = len(base_df)

    for feature in tqdm(feature_list, desc="Permuting features"):
        drops = []
        test_values = base_df.iloc[test_start:test_end][feature].to_numpy(copy=True)
        column_index = base_df.columns.get_loc(feature)

        for _ in range(10):
            permuted_df = base_df.copy()
            shuffled = np.array(test_values, copy=True)
            np.random.shuffle(shuffled)
            permuted_df.iloc[test_start:test_end, column_index] = shuffled

            permuted_bundle = make_split_bundle(
                permuted_df,
                feature_list,
                corr_features,
                config["lookback"],
                train_split,
                val_split,
                purge_gap,
            )
            permuted_loader = build_test_loader(permuted_bundle, config, core_mean, core_std, corr_mean, corr_std)
            permuted_mcc = evaluate_mcc(model, permuted_loader, device)
            drops.append(baseline_mcc - permuted_mcc)

        results[feature] = float(np.mean(drops))

    results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / f"{task['mode_name']}_feature_rankings.json"
    with open(output_path, "w") as file:
        json.dump(results, file, indent=4)

    print(f"Saved feature rankings to {output_path}")


if __name__ == "__main__":
    main()
