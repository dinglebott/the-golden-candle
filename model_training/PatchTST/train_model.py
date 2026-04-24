import copy
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, log_loss, matthews_corrcoef, roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm

from classes import PatchTST, WindowedTimeSeriesDataset
from data_utils import (
    CORE_FEATURES,
    compute_train_stats,
    get_corr_feature_names,
    make_split_bundle,
    prepare_task_dataframe,
    resolve_pretrain_pairs,
    resolve_target_instrument,
)


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
    "batch_size": 256,
    "epochs": 30,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "patience": 6,
    "grad_clip": 1.0,
    "gate_loss_weight": 1.0,
    "direction_loss_weight": 1.0,
    "num_workers": 0,
    "pretrain_epochs": 20,
    "pretrain_learning_rate": 3e-4,
    "pretrain_weight_decay": 1e-4,
    "pretrain_patience": 5,
    "pretrain_batch_size": 256,
    "finetune_epochs": 20,
    "finetune_learning_rate": 1e-4,
    "finetune_weight_decay": 1e-4,
    "finetune_patience": 6,
    "finetune_batch_size": 256,
    "finetune_unfreeze_shared_blocks": 0,
}

TASK_CONFIG = {
    0: {
        "mode_name": "gate",
        "display_name": "Gate",
        "class_names": ("flat", "directional"),
    },
    1: {
        "mode_name": "dir",
        "display_name": "Direction",
        "class_names": ("down", "up"),
    },
}


def safe_mean(values):
    return float(np.mean(values)) if len(values) > 0 else float("nan")


def load_env():
    with open(Path(__file__).parent.parent.parent / "env.json", "r") as file:
        return json.load(file)


def load_training_params(version):
    params = dict(DEFAULT_PARAMS)
    config_path = Path(__file__).parent / f"model_configs/v{version}/params.json"
    if config_path.exists():
        with open(config_path, "r") as file:
            params.update(json.load(file))
    return params


def compute_pos_weight(labels):
    positive_count = float(labels.sum())
    negative_count = float(len(labels) - positive_count)
    if positive_count == 0:
        return 1.0
    return max(negative_count / positive_count, 1.0)


def compute_binary_metrics(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if len(y_true) == 0:
        return {
            "balanced_accuracy": float("nan"),
            "mcc": float("nan"),
            "log_loss": float("nan"),
            "roc_auc": float("nan"),
            "confusion_matrix": np.zeros((2, 2), dtype=np.int64),
        }

    y_pred = (y_prob >= 0.5).astype(np.int64)
    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) * 100,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "log_loss": log_loss(y_true, np.column_stack([1 - y_prob, y_prob]), labels=[0, 1]),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
    }
    metrics["roc_auc"] = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")
    return metrics


def format_confusion_matrix(cmatrix, row_labels, col_labels):
    df = pd.DataFrame(cmatrix, index=row_labels, columns=col_labels)
    df["Count"] = df.sum(axis=1)
    df.loc["Count"] = df.sum(axis=0)
    return df


def build_datasets(split_bundles, lookback, core_mean, core_std, corr_mean, corr_std, split_name):
    datasets = []
    for bundle in tqdm(
        split_bundles,
        desc=f"Building {split_name} datasets",
        total=len(split_bundles),
        leave=False,
    ):
        datasets.append(
            WindowedTimeSeriesDataset(bundle[split_name], lookback, core_mean, core_std, corr_mean, corr_std)
        )

    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)


def build_stage_loaders(split_bundles, config, stage_name, core_mean, core_std, corr_mean, corr_std):
    batch_size = config[f"{stage_name}_batch_size"]
    train_dataset = build_datasets(split_bundles, config["lookback"], core_mean, core_std, corr_mean, corr_std, "train")
    val_dataset = build_datasets(split_bundles, config["lookback"], core_mean, core_std, corr_mean, corr_std, "val")
    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=config["num_workers"]),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=config["num_workers"]),
    }


def build_eval_loaders(split_bundle, config, core_mean, core_std, corr_mean, corr_std):
    batch_size = config["finetune_batch_size"]
    return {
        "train": DataLoader(
            WindowedTimeSeriesDataset(split_bundle["train"], config["lookback"], core_mean, core_std, corr_mean, corr_std),
            batch_size=batch_size,
            shuffle=False,
            num_workers=config["num_workers"],
        ),
        "test": DataLoader(
            WindowedTimeSeriesDataset(split_bundle["test"], config["lookback"], core_mean, core_std, corr_mean, corr_std),
            batch_size=batch_size,
            shuffle=False,
            num_workers=config["num_workers"],
        ),
    }


def gather_stage_targets(split_bundles, split_name):
    return np.concatenate([
        bundle[split_name].targets[bundle[split_name].target_indices]
        for bundle in split_bundles
    ])


def run_epoch(model, loader, device, pos_weight, optimizer=None, grad_clip=None, desc=None):
    is_training = optimizer is not None
    model.train(is_training)

    probabilities = []
    targets = []
    total_loss = 0.0
    total_samples = 0

    progress = tqdm(loader, desc=desc, total=len(loader), leave=False)
    for batch in progress:
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
        progress.set_postfix(loss=f"{total_loss / max(total_samples, 1):.4f}")

        probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
        targets.append(target.detach().cpu().numpy())

    return {
        "loss": total_loss / max(total_samples, 1),
        "true": np.concatenate(targets).astype(np.int64),
        "prob": np.concatenate(probabilities),
    }


def train_stage(model, loaders, device, pos_weight, config, stage_name):
    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=config[f"{stage_name}_learning_rate"],
        weight_decay=config[f"{stage_name}_weight_decay"],
    )

    best_state = None
    best_epoch = 0
    best_val_loss = math.inf
    patience_counter = 0
    total_epochs = config[f"{stage_name}_epochs"]

    for epoch in range(1, total_epochs + 1):
        train_results = run_epoch(
            model,
            loaders["train"],
            device,
            pos_weight,
            optimizer=optimizer,
            grad_clip=config["grad_clip"],
            desc=f"{stage_name.capitalize()} {epoch}/{total_epochs} train",
        )
        val_results = run_epoch(
            model,
            loaders["val"],
            device,
            pos_weight,
            desc=f"{stage_name.capitalize()} {epoch}/{total_epochs} val",
        )

        print(
            f"{stage_name.capitalize()} epoch {epoch:02d} | "
            f"train_loss={train_results['loss']:.4f} | "
            f"val_loss={val_results['loss']:.4f}"
        )

        if val_results["loss"] < best_val_loss:
            best_val_loss = val_results["loss"]
            best_epoch = epoch
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= config[f"{stage_name}_patience"]:
                print(f"{stage_name.capitalize()} early stopping triggered after epoch {epoch}.")
                break

    if best_state is None:
        raise RuntimeError(f"{stage_name} failed to produce a valid checkpoint")

    model.load_state_dict(best_state)
    return {
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }


def evaluate_target_model(model, loaders, device, pos_weight):
    train_results = run_epoch(
        model,
        loaders["train"],
        device,
        pos_weight,
        desc="Evaluating train split",
    )
    test_results = run_epoch(
        model,
        loaders["test"],
        device,
        pos_weight,
        desc="Evaluating test split",
    )
    return train_results, test_results


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    env = load_env()
    patchtst_env = env.get("patchtst", {})

    year_now = env["year_now"]
    granularity = env["granularity"]
    train_split = env["train_split"]
    val_split = env["val_split"]
    purge_gap = env["n_value"]
    k_value = env["k_value"]
    binary = env["binary"]
    log_metrics = env["log_metrics"]
    corr_pair = env["corr_pair"]
    version = patchtst_env["train_version"]

    if binary not in TASK_CONFIG:
        raise ValueError("binary must be 0 or 1")

    task = TASK_CONFIG[binary]
    config = load_training_params(version)
    target_instrument = resolve_target_instrument(env)
    pretrain_pairs = resolve_pretrain_pairs(env)
    corr_features = get_corr_feature_names(corr_pair) if isinstance(corr_pair, str) else []

    device_name = env.get("device", "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested in env.json but not available; falling back to CPU.")
        device_name = "cpu"
    device = torch.device(device_name)

    pretrain_bundles = []
    for instrument in tqdm(pretrain_pairs, desc="Preparing pretrain datasets", total=len(pretrain_pairs)):
        df = prepare_task_dataframe(
            instrument,
            granularity,
            year_now,
            k_value,
            purge_gap,
            binary,
            corr_pair=0,
            include_corr_features=False,
        )
        pretrain_bundles.append(
            make_split_bundle(df, CORE_FEATURES, [], config["lookback"], train_split, val_split, purge_gap)
        )

    pretrain_core_mean, pretrain_core_std = compute_train_stats(pretrain_bundles, "core")
    empty_corr_mean = np.zeros((1, 0), dtype=np.float32)
    empty_corr_std = np.zeros((1, 0), dtype=np.float32)
    pretrain_loaders = build_stage_loaders(
        pretrain_bundles,
        config,
        "pretrain",
        pretrain_core_mean,
        pretrain_core_std,
        empty_corr_mean,
        empty_corr_std,
    )

    pretrain_targets = gather_stage_targets(pretrain_bundles, "train")
    pretrain_pos_weight = torch.tensor(compute_pos_weight(pretrain_targets), dtype=torch.float32, device=device)

    model = PatchTST(len(CORE_FEATURES), len(corr_features), config).to(device)
    pretrain_summary = train_stage(
        model,
        pretrain_loaders,
        device,
        pretrain_pos_weight,
        config,
        "pretrain",
    )

    print("Preparing finetune dataset...")
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
    target_bundle = make_split_bundle(
        target_df,
        CORE_FEATURES,
        corr_features,
        config["lookback"],
        train_split,
        val_split,
        purge_gap,
    )

    finetune_core_mean, finetune_core_std = compute_train_stats([target_bundle], "core")
    finetune_corr_mean, finetune_corr_std = compute_train_stats([target_bundle], "corr")
    finetune_loaders = build_stage_loaders(
        [target_bundle],
        config,
        "finetune",
        finetune_core_mean,
        finetune_core_std,
        finetune_corr_mean,
        finetune_corr_std,
    )

    model.load_state_dict(pretrain_summary["best_state"])
    model.freeze_for_finetune(config["finetune_unfreeze_shared_blocks"])

    finetune_targets = gather_stage_targets([target_bundle], "train")
    finetune_pos_weight = torch.tensor(compute_pos_weight(finetune_targets), dtype=torch.float32, device=device)

    finetune_summary = train_stage(
        model,
        finetune_loaders,
        device,
        finetune_pos_weight,
        config,
        "finetune",
    )

    eval_loaders = build_eval_loaders(
        target_bundle,
        config,
        finetune_core_mean,
        finetune_core_std,
        finetune_corr_mean,
        finetune_corr_std,
    )
    train_results, test_results = evaluate_target_model(
        model,
        eval_loaders,
        device,
        finetune_pos_weight,
    )

    train_metrics = compute_binary_metrics(train_results["true"], train_results["prob"])
    test_metrics = compute_binary_metrics(test_results["true"], test_results["prob"])

    class_names = task["class_names"]
    lines = []
    lines.append(
        f"=== v{version} | {target_instrument} {granularity} | {task['mode_name']} | "
        f"PatchTST pretrain+finetune | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ==="
    )
    lines.append(f"Pretrain pairs: {', '.join(pretrain_pairs)}")
    lines.append(f"Finetune target: {target_instrument}")
    lines.append(f"Correlated pair adapter: {corr_pair if corr_features else 'disabled'}")
    lines.append(f"Pretrain best epoch: {pretrain_summary['best_epoch']}")
    lines.append(f"Finetune best epoch: {finetune_summary['best_epoch']}")
    lines.append(f"Core features: {len(CORE_FEATURES)} | Corr features: {len(corr_features)}")
    lines.append(f"Lookback: {config['lookback']} | Patch length: {config['patch_len']} | Patch stride: {config['patch_stride']}")
    lines.append(f"Shared blocks: {config['base_encoder_blocks']} | Task blocks: {config['branch_encoder_blocks']}")
    lines.append(f"Purged split gap: {purge_gap}")
    lines.append("")
    lines.append(f"{task['display_name']} task")
    lines.append(f"Balanced accuracy: {test_metrics['balanced_accuracy']:.2f}%")
    lines.append(f"MCC: {test_metrics['mcc']:.4f}")
    lines.append(f"MCC (train set): {train_metrics['mcc']:.4f}")
    lines.append(f"Log loss: {test_metrics['log_loss']:.4f}")
    lines.append(f"ROC-AUC: {test_metrics['roc_auc']:.4f}")
    lines.append(
        f"\nConfusion matrix:\n"
        f"{format_confusion_matrix(test_metrics['confusion_matrix'], ['Real 0', 'Real 1'], ['Pred 0', 'Pred 1'])}"
    )

    negative_probs = test_results["prob"][test_results["true"] == 0]
    positive_probs = test_results["prob"][test_results["true"] == 1]
    lines.append(
        f"True={class_names[0]}: avg P({class_names[0]})={safe_mean(1 - negative_probs):.3f} "
        f"P({class_names[1]})={safe_mean(negative_probs):.3f}"
    )
    lines.append(
        f"True={class_names[1]}: avg P({class_names[0]})={safe_mean(1 - positive_probs):.3f} "
        f"P({class_names[1]})={safe_mean(positive_probs):.3f}"
    )

    output = "\n".join(lines) + "\n"
    print("\n" + output)

    script_dir = Path(__file__).parent
    if log_metrics:
        (script_dir / "results").mkdir(exist_ok=True)
        with open(script_dir / "results/test_metrics.log", "a") as log_file:
            log_file.write(output + "\n")

    pretrained_path = script_dir / f"models/pretrained/{task['mode_name']}_PatchTST_pretrained_{granularity}_{year_now}_v{version}.pt"
    pretrained_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": pretrain_summary["best_state"],
            "config": config,
            "core_features": CORE_FEATURES,
            "corr_features": [],
            "normalization": {
                "core_mean": pretrain_core_mean.squeeze(0).tolist(),
                "core_std": pretrain_core_std.squeeze(0).tolist(),
            },
            "metadata": {
                "mode": task["mode_name"],
                "binary": binary,
                "pretrain_pairs": pretrain_pairs,
                "granularity": granularity,
                "year_now": year_now,
                "best_epoch": pretrain_summary["best_epoch"],
            },
        },
        pretrained_path,
    )

    model_path = script_dir / f"models/{target_instrument}/{task['mode_name']}_PatchTST_{target_instrument}_{granularity}_{year_now}_v{version}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "core_features": CORE_FEATURES,
            "corr_features": corr_features,
            "normalization": {
                "core_mean": finetune_core_mean.squeeze(0).tolist(),
                "core_std": finetune_core_std.squeeze(0).tolist(),
                "corr_mean": finetune_corr_mean.squeeze(0).tolist(),
                "corr_std": finetune_corr_std.squeeze(0).tolist(),
            },
            "metadata": {
                "mode": task["mode_name"],
                "binary": binary,
                "target_instrument": target_instrument,
                "pretrain_pairs": pretrain_pairs,
                "corr_pair": corr_pair,
                "granularity": granularity,
                "year_now": year_now,
                "pretrain_best_epoch": pretrain_summary["best_epoch"],
                "finetune_best_epoch": finetune_summary["best_epoch"],
                "finetune_unfreeze_shared_blocks": config["finetune_unfreeze_shared_blocks"],
            },
        },
        model_path,
    )


if __name__ == "__main__":
    main()
