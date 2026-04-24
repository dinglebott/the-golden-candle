import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from classes import PatchTST, WindowedTimeSeriesDataset
from data_utils import (
    compute_train_stats,
    get_corr_feature_names,
    make_split_bundle,
    prepare_task_dataframe,
    resolve_target_instrument,
)

TASK_CONFIG = {
    0: {"mode_name": "gate", "display_name": "Gate"},
    1: {"mode_name": "dir", "display_name": "Direction"},
}

DEFAULT_PARAMS = {
    "lookback": 96,
    "patch_len": 8,
    "patch_stride": 8,
    "d_model": 128,
    "num_heads": 4,
    "mlp_ratio": 4,
    "dropout": 0.1,
    "base_encoder_blocks": 2,
    "branch_encoder_blocks": 2,
    "grad_clip": 1.0,
    "num_workers": 0,
    "pretrain_epochs": 20,
    "pretrain_learning_rate": 3e-4,
    "pretrain_weight_decay": 1e-4,
    "pretrain_patience": 5,
    "pretrain_batch_size": 256,
    "finetune_epochs": 30,
    "finetune_learning_rate": 1e-4,
    "finetune_weight_decay": 1e-4,
    "finetune_patience": 10,
    "finetune_batch_size": 256,
    "finetune_unfreeze_shared_blocks": 0,
}


def load_env():
    with open(Path(__file__).parent.parent.parent / "env.json", "r") as f:
        return json.load(f)


def load_task_features(version, mode_name):
    config_path = Path(__file__).parent / f"model_configs/v{version}/{mode_name}_features.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)["features"]
    raise FileNotFoundError(f"Feature file not found: {config_path}")


def load_base_params(version, mode_name):
    params = dict(DEFAULT_PARAMS)
    config_path = Path(__file__).parent / f"model_configs/v{version}/{mode_name}_params.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            params.update(json.load(f))
    return params


def compute_pos_weight(labels, device):
    positive_count = float(labels.sum())
    negative_count = float(len(labels) - positive_count)
    if positive_count == 0:
        return torch.tensor(1.0, dtype=torch.float32, device=device)
    return torch.tensor(max(negative_count / positive_count, 1.0), dtype=torch.float32, device=device)


def run_epoch(model, loader, device, pos_weight, optimizer=None, grad_clip=None):
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        core_inputs = batch["core_inputs"].to(device)
        corr_inputs = batch["corr_inputs"].to(device)
        target = batch["target"].to(device)

        with torch.set_grad_enabled(is_training):
            logits = model(core_inputs, corr_inputs)
            loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        batch_size = core_inputs.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def suggest_config(trial, base_config, skip_pretrain):
    config = dict(base_config)

    lookback = trial.suggest_categorical("lookback", [48, 96, 144, 192, 240])
    patch_len = trial.suggest_categorical("patch_len", [4, 8, 16])
    overlapping = trial.suggest_categorical("overlapping_patches", [True, False])
    patch_stride = max(1, patch_len // 2) if overlapping else patch_len

    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    # All d_model choices are divisible by 2, 4, and 8
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8])
    mlp_ratio = trial.suggest_categorical("mlp_ratio", [2, 4])
    dropout = trial.suggest_float("dropout", 0.05, 0.4, step=0.05)
    base_encoder_blocks = trial.suggest_int("base_encoder_blocks", 1, 4)
    branch_encoder_blocks = trial.suggest_int("branch_encoder_blocks", 1, 3)

    finetune_lr = trial.suggest_float("finetune_learning_rate", 1e-5, 5e-4, log=True)
    finetune_wd = trial.suggest_float("finetune_weight_decay", 1e-5, 1e-2, log=True)
    finetune_batch = trial.suggest_categorical("finetune_batch_size", [128, 256, 512, 1024])

    config.update({
        "lookback": lookback,
        "patch_len": patch_len,
        "patch_stride": patch_stride,
        "d_model": d_model,
        "num_heads": num_heads,
        "mlp_ratio": mlp_ratio,
        "dropout": dropout,
        "base_encoder_blocks": base_encoder_blocks,
        "branch_encoder_blocks": branch_encoder_blocks,
        "finetune_learning_rate": finetune_lr,
        "finetune_weight_decay": finetune_wd,
        "finetune_batch_size": finetune_batch,
    })

    if not skip_pretrain:
        pretrain_lr = trial.suggest_float("pretrain_learning_rate", 1e-5, 5e-4, log=True)
        pretrain_wd = trial.suggest_float("pretrain_weight_decay", 1e-5, 1e-2, log=True)
        pretrain_batch = trial.suggest_categorical("pretrain_batch_size", [128, 256, 512, 1024])
        unfreeze = trial.suggest_int("finetune_unfreeze_shared_blocks", 0, base_encoder_blocks)
        config.update({
            "pretrain_learning_rate": pretrain_lr,
            "pretrain_weight_decay": pretrain_wd,
            "pretrain_batch_size": pretrain_batch,
            "finetune_unfreeze_shared_blocks": unfreeze,
        })

    return config


def train_and_evaluate(config, target_df, core_features, corr_features, env, device, trial=None):
    train_split = env["train_split"]
    val_split = env["val_split"]
    purge_gap = env["n_value"]

    target_bundle = make_split_bundle(
        target_df,
        core_features,
        corr_features,
        config["lookback"],
        train_split,
        val_split,
        purge_gap,
    )

    core_mean, core_std = compute_train_stats([target_bundle], "core")
    corr_mean, corr_std = compute_train_stats([target_bundle], "corr")

    lookback = config["lookback"]
    batch_size = config["finetune_batch_size"]
    num_workers = config["num_workers"]

    train_loader = DataLoader(
        WindowedTimeSeriesDataset(target_bundle["train"], lookback, core_mean, core_std, corr_mean, corr_std),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        WindowedTimeSeriesDataset(target_bundle["val"], lookback, core_mean, core_std, corr_mean, corr_std),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    train_targets = target_bundle["train"].targets[target_bundle["train"].target_indices]
    pos_weight = compute_pos_weight(train_targets, device)

    model = PatchTST(len(core_features), len(corr_features), config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["finetune_learning_rate"],
        weight_decay=config["finetune_weight_decay"],
    )

    best_val_loss = math.inf
    patience_counter = 0

    for epoch in range(1, config["finetune_epochs"] + 1):
        run_epoch(model, train_loader, device, pos_weight, optimizer=optimizer, grad_clip=config["grad_clip"])
        val_loss = run_epoch(model, val_loader, device, pos_weight)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["finetune_patience"]:
                break

        if trial is not None:
            trial.report(best_val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return best_val_loss


def build_output_params(best_trial, base_config, skip_pretrain):
    params = best_trial.params

    patch_len = params["patch_len"]
    overlapping = params["overlapping_patches"]
    patch_stride = max(1, patch_len // 2) if overlapping else patch_len

    output = {
        "lookback": params["lookback"],
        "patch_len": patch_len,
        "patch_stride": patch_stride,
        "d_model": params["d_model"],
        "num_heads": params["num_heads"],
        "mlp_ratio": params["mlp_ratio"],
        "dropout": round(params["dropout"], 4),
        "base_encoder_blocks": params["base_encoder_blocks"],
        "branch_encoder_blocks": params["branch_encoder_blocks"],
        "grad_clip": base_config["grad_clip"],
        "num_workers": base_config["num_workers"],
        "finetune_epochs": base_config["finetune_epochs"],
        "finetune_learning_rate": round(params["finetune_learning_rate"], 8),
        "finetune_weight_decay": round(params["finetune_weight_decay"], 8),
        "finetune_patience": base_config["finetune_patience"],
        "finetune_batch_size": params["finetune_batch_size"],
        "finetune_unfreeze_shared_blocks": 0,
    }

    if not skip_pretrain:
        output.update({
            "pretrain_epochs": base_config["pretrain_epochs"],
            "pretrain_learning_rate": round(params["pretrain_learning_rate"], 8),
            "pretrain_weight_decay": round(params["pretrain_weight_decay"], 8),
            "pretrain_patience": base_config["pretrain_patience"],
            "pretrain_batch_size": params["pretrain_batch_size"],
            "finetune_unfreeze_shared_blocks": params["finetune_unfreeze_shared_blocks"],
        })

    return output


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for PatchTST")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials (default: 50)")
    parser.add_argument("--timeout", type=int, default=None, help="Search timeout in seconds")
    args = parser.parse_args()

    env = load_env()
    patchtst_env = env.get("patchtst", {})

    binary = env["binary"]
    if binary not in TASK_CONFIG:
        raise ValueError("binary must be 0 or 1")

    task = TASK_CONFIG[binary]
    mode_name = task["mode_name"]

    year_now = env["year_now"]
    granularity = env["granularity"]
    k_value = env["k_value"]
    purge_gap = env["n_value"]
    corr_pair = env["corr_pair"]
    version = patchtst_env["train_version"]
    skip_pretrain = patchtst_env.get("skip_pretrain", False)

    core_features = load_task_features(version, mode_name)
    base_config = load_base_params(version, mode_name)
    target_instrument = resolve_target_instrument(env)
    corr_features = get_corr_feature_names(corr_pair) if isinstance(corr_pair, str) else []

    device_name = env.get("device", "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    print(f"Task:          {task['display_name']} (binary={binary})")
    print(f"Target:        {target_instrument} {granularity}")
    print(f"Features:      {len(core_features)} core + {len(corr_features)} corr")
    print(f"Config:        v{version} | skip_pretrain={skip_pretrain}")
    print(f"Trials:        {args.trials}" + (f" | timeout={args.timeout}s" if args.timeout else ""))
    print()

    print("Preparing dataset...")
    target_df = prepare_task_dataframe(
        target_instrument,
        granularity,
        year_now,
        k_value,
        purge_gap,
        binary,
        corr_pair=corr_pair,
        include_corr_features=bool(corr_features),
    )
    print("Dataset ready.\n")

    def objective(trial):
        torch.manual_seed(trial.number)
        np.random.seed(trial.number)

        config = suggest_config(trial, base_config, skip_pretrain)

        try:
            return train_and_evaluate(config, target_df, core_features, corr_features, env, device, trial=trial)
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"  Trial {trial.number} failed: {e}")
            return float("inf")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    study.optimize(objective, n_trials=args.trials, timeout=args.timeout, show_progress_bar=True)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"\nTrials completed: {len(completed)} | pruned: {len(pruned)}")
    print(f"Best trial:  #{study.best_trial.number}  val_loss={study.best_value:.6f}")
    print("Best params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    output_params = build_output_params(study.best_trial, base_config, skip_pretrain)

    script_dir = Path(__file__).parent
    output_path = script_dir / f"results/{mode_name}_tuned_params.json"
    with open(output_path, "w") as f:
        json.dump(output_params, f, indent=4)

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
