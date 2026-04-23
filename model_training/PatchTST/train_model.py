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

from classes import MultiTaskPatchTST, WindowedTimeSeriesDataset
from data_utils import (
    CORE_FEATURES,
    compute_train_stats,
    get_corr_feature_names,
    prepare_multitask_dataframe,
    make_split_bundle,
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


def compute_direction_pos_weight(direction_targets, direction_mask):
    valid = direction_mask > 0.5
    positives = float(direction_targets[valid].sum())
    negatives = float(valid.sum() - positives)
    if positives == 0:
        return 1.0
    return max(negatives / positives, 1.0)


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
    datasets = [
        WindowedTimeSeriesDataset(bundle[split_name], lookback, core_mean, core_std, corr_mean, corr_std)
        for bundle in split_bundles
    ]
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
    gate_labels = np.concatenate([
        bundle[split_name].gate_targets[bundle[split_name].target_indices]
        for bundle in split_bundles
    ])
    direction_targets = np.concatenate([
        bundle[split_name].direction_targets[bundle[split_name].target_indices]
        for bundle in split_bundles
    ])
    direction_masks = np.concatenate([
        bundle[split_name].direction_mask[bundle[split_name].target_indices]
        for bundle in split_bundles
    ])
    return gate_labels, direction_targets, direction_masks


def run_epoch(model, loader, device, gate_pos_weight, direction_pos_weight, loss_weights, optimizer=None, grad_clip=None):
    is_training = optimizer is not None
    model.train(is_training)

    gate_probabilities = []
    gate_targets = []
    direction_probabilities = []
    direction_targets = []
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        core_inputs = batch["core_inputs"].to(device)
        corr_inputs = batch["corr_inputs"].to(device)
        gate_target = batch["gate_target"].to(device)
        direction_target = batch["direction_target"].to(device)
        direction_mask = batch["direction_mask"].to(device) > 0.5

        with torch.set_grad_enabled(is_training):
            gate_logits, direction_logits = model(core_inputs, corr_inputs)
            gate_loss = F.binary_cross_entropy_with_logits(gate_logits, gate_target, pos_weight=gate_pos_weight)

            if direction_mask.any():
                direction_loss = F.binary_cross_entropy_with_logits(
                    direction_logits[direction_mask],
                    direction_target[direction_mask],
                    pos_weight=direction_pos_weight,
                )
            else:
                direction_loss = torch.zeros((), device=device)

            loss = loss_weights["gate"] * gate_loss + loss_weights["direction"] * direction_loss

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        batch_size = core_inputs.size(0)
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size

        gate_probabilities.append(torch.sigmoid(gate_logits).detach().cpu().numpy())
        gate_targets.append(gate_target.detach().cpu().numpy())

        if direction_mask.any():
            direction_probabilities.append(torch.sigmoid(direction_logits[direction_mask]).detach().cpu().numpy())
            direction_targets.append(direction_target[direction_mask].detach().cpu().numpy())

    gate_probs = np.concatenate(gate_probabilities)
    gate_true = np.concatenate(gate_targets).astype(np.int64)
    direction_probs = np.concatenate(direction_probabilities) if direction_probabilities else np.array([], dtype=np.float64)
    direction_true = np.concatenate(direction_targets).astype(np.int64) if direction_targets else np.array([], dtype=np.int64)

    return {
        "loss": total_loss / max(total_samples, 1),
        "gate_true": gate_true,
        "gate_prob": gate_probs,
        "direction_true": direction_true,
        "direction_prob": direction_probs,
    }


def train_stage(model, loaders, device, gate_pos_weight, direction_pos_weight, config, stage_name):
    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=config[f"{stage_name}_learning_rate"],
        weight_decay=config[f"{stage_name}_weight_decay"],
    )

    best_state = None
    best_epoch = 0
    best_val_loss = math.inf
    patience_counter = 0

    for epoch in range(1, config[f"{stage_name}_epochs"] + 1):
        train_results = run_epoch(
            model,
            loaders["train"],
            device,
            gate_pos_weight,
            direction_pos_weight,
            {"gate": config["gate_loss_weight"], "direction": config["direction_loss_weight"]},
            optimizer=optimizer,
            grad_clip=config["grad_clip"],
        )
        val_results = run_epoch(
            model,
            loaders["val"],
            device,
            gate_pos_weight,
            direction_pos_weight,
            {"gate": config["gate_loss_weight"], "direction": config["direction_loss_weight"]},
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


def evaluate_target_model(model, loaders, device, gate_pos_weight, direction_pos_weight, config):
    train_results = run_epoch(
        model,
        loaders["train"],
        device,
        gate_pos_weight,
        direction_pos_weight,
        {"gate": config["gate_loss_weight"], "direction": config["direction_loss_weight"]},
    )
    test_results = run_epoch(
        model,
        loaders["test"],
        device,
        gate_pos_weight,
        direction_pos_weight,
        {"gate": config["gate_loss_weight"], "direction": config["direction_loss_weight"]},
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
    log_metrics = env["log_metrics"]
    corr_pair = env["corr_pair"]
    version = patchtst_env["train_version"]

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
    for instrument in pretrain_pairs:
        df = prepare_multitask_dataframe(
            instrument,
            granularity,
            year_now,
            k_value,
            purge_gap,
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

    pretrain_gate_labels, pretrain_direction_targets, pretrain_direction_masks = gather_stage_targets(pretrain_bundles, "train")
    pretrain_gate_pos_weight = torch.tensor(compute_pos_weight(pretrain_gate_labels), dtype=torch.float32, device=device)
    pretrain_direction_pos_weight = torch.tensor(
        compute_direction_pos_weight(pretrain_direction_targets, pretrain_direction_masks),
        dtype=torch.float32,
        device=device,
    )

    model = MultiTaskPatchTST(len(CORE_FEATURES), len(corr_features), config).to(device)
    pretrain_summary = train_stage(
        model,
        pretrain_loaders,
        device,
        pretrain_gate_pos_weight,
        pretrain_direction_pos_weight,
        config,
        "pretrain",
    )

    target_df = prepare_multitask_dataframe(
        target_instrument,
        granularity,
        year_now,
        k_value,
        purge_gap,
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

    finetune_gate_labels, finetune_direction_targets, finetune_direction_masks = gather_stage_targets([target_bundle], "train")
    finetune_gate_pos_weight = torch.tensor(compute_pos_weight(finetune_gate_labels), dtype=torch.float32, device=device)
    finetune_direction_pos_weight = torch.tensor(
        compute_direction_pos_weight(finetune_direction_targets, finetune_direction_masks),
        dtype=torch.float32,
        device=device,
    )

    finetune_summary = train_stage(
        model,
        finetune_loaders,
        device,
        finetune_gate_pos_weight,
        finetune_direction_pos_weight,
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
        finetune_gate_pos_weight,
        finetune_direction_pos_weight,
        config,
    )

    train_gate_metrics = compute_binary_metrics(train_results["gate_true"], train_results["gate_prob"])
    test_gate_metrics = compute_binary_metrics(test_results["gate_true"], test_results["gate_prob"])
    train_direction_metrics = compute_binary_metrics(train_results["direction_true"], train_results["direction_prob"])
    test_direction_metrics = compute_binary_metrics(test_results["direction_true"], test_results["direction_prob"])

    lines = []
    lines.append(f"=== v{version} | {target_instrument} {granularity} | PatchTST pretrain+finetune | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    lines.append(f"Pretrain pairs: {', '.join(pretrain_pairs)}")
    lines.append(f"Finetune target: {target_instrument}")
    lines.append(f"Correlated pair adapter: {corr_pair if corr_features else 'disabled'}")
    lines.append(f"Pretrain best epoch: {pretrain_summary['best_epoch']}")
    lines.append(f"Finetune best epoch: {finetune_summary['best_epoch']}")
    lines.append(f"Core features: {len(CORE_FEATURES)} | Corr features: {len(corr_features)}")
    lines.append(f"Lookback: {config['lookback']} | Patch length: {config['patch_len']} | Patch stride: {config['patch_stride']}")
    lines.append(f"Shared blocks: {config['base_encoder_blocks']} | Blocks per head: {config['branch_encoder_blocks']}")
    lines.append(f"Purged split gap: {purge_gap}")
    lines.append("")
    lines.append("Gate head (flat vs directional)")
    lines.append(f"Balanced accuracy: {test_gate_metrics['balanced_accuracy']:.2f}%")
    lines.append(f"MCC: {test_gate_metrics['mcc']:.4f}")
    lines.append(f"MCC (train set): {train_gate_metrics['mcc']:.4f}")
    lines.append(f"Log loss: {test_gate_metrics['log_loss']:.4f}")
    lines.append(f"ROC-AUC: {test_gate_metrics['roc_auc']:.4f}")
    lines.append(
        f"\nConfusion matrix:\n{format_confusion_matrix(test_gate_metrics['confusion_matrix'], ['Real 0', 'Real 1'], ['Pred 0', 'Pred 1'])}"
    )
    gate_negative = test_results["gate_prob"][test_results["gate_true"] == 0]
    gate_positive = test_results["gate_prob"][test_results["gate_true"] == 1]
    lines.append(f"True=flat: avg P(directional)={safe_mean(gate_negative):.3f}")
    lines.append(f"True=directional: avg P(directional)={safe_mean(gate_positive):.3f}")
    lines.append("")
    lines.append("Direction head (down vs up | directional samples only)")
    lines.append(f"Balanced accuracy: {test_direction_metrics['balanced_accuracy']:.2f}%")
    lines.append(f"MCC: {test_direction_metrics['mcc']:.4f}")
    lines.append(f"MCC (train set): {train_direction_metrics['mcc']:.4f}")
    lines.append(f"Log loss: {test_direction_metrics['log_loss']:.4f}")
    lines.append(f"ROC-AUC: {test_direction_metrics['roc_auc']:.4f}")
    lines.append(
        f"\nConfusion matrix:\n{format_confusion_matrix(test_direction_metrics['confusion_matrix'], ['Real 0', 'Real 1'], ['Pred 0', 'Pred 1'])}"
    )
    direction_negative = test_results["direction_prob"][test_results["direction_true"] == 0]
    direction_positive = test_results["direction_prob"][test_results["direction_true"] == 1]
    lines.append(f"True=down: avg P(up)={safe_mean(direction_negative):.3f}")
    lines.append(f"True=up: avg P(up)={safe_mean(direction_positive):.3f}")

    output = "\n".join(lines) + "\n"
    print("\n" + output)

    script_dir = Path(__file__).parent
    if log_metrics:
        (script_dir / "results").mkdir(exist_ok=True)
        with open(script_dir / "results/test_metrics.log", "a") as log_file:
            log_file.write(output + "\n")

    pretrained_path = script_dir / f"models/pretrained/PatchTST_pretrained_{granularity}_{year_now}_v{version}.pt"
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
                "pretrain_pairs": pretrain_pairs,
                "granularity": granularity,
                "year_now": year_now,
                "best_epoch": pretrain_summary["best_epoch"],
            },
        },
        pretrained_path,
    )

    model_path = script_dir / f"models/{target_instrument}/PatchTST_{target_instrument}_{granularity}_{year_now}_v{version}.pt"
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
