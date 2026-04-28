import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
import optuna
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_processing import dataparser
from patterns import registry
from classes import EventDataset, CnnLstm

optuna.logging.set_verbosity(optuna.logging.WARNING)

# LOAD CONFIGS
with open(Path(__file__).parent.parent / "env.json", "r") as f:
    env = json.load(f)
year_now = env["year_now"]
instrument = env["instrument"]
granularity = env["granularity"]
n = env["n_value"]
k = env["k_value"]
train_split = env["train_split"]
val_split = env["val_split"]
device = torch.device(env["device"] if torch.cuda.is_available() else "cpu")
pattern = env["pattern"]
version = env["cnn_lstm"]["train_version"]

# LOAD EVENT DETECTOR
pattern_module = registry.load(pattern)

# LOAD FEATURES
with open(Path(__file__).parent / f"model_configs/v{version}/{pattern}_features.json") as f:
    features = json.load(f)["features"]
print(f"Tuning on {len(features)} features: {features}")

meta_features = pattern_module.METADATA_FEATURES
seq_features = [f for f in features if f not in meta_features]

# LOAD AND PARSE DATA
_raw_data_dir = Path(__file__).parent.parent.parent / "raw_data"
df = dataparser.parseData(_raw_data_dir / f"{instrument}_{granularity}_{year_now - 21}-01-01_{year_now}-04-01.json")

# DETECT AND LABEL
instances = pattern_module.detect(df)
labelled = pattern_module.label_instances(df, instances, n, k)

# Use train + val only (no test leakage)
val_end = int((train_split + val_split) * len(labelled))
instances_tuning = labelled[:val_end]


def build_sequences(df, instances, seq_len, seq_features, meta_features):
    X_seq, X_meta, y = [], [], []
    for inst in instances:
        idx = inst["index"]
        if idx < seq_len - 1:
            continue
        seq = df.iloc[idx - seq_len + 1 : idx + 1][seq_features].values.astype(np.float32)
        meta = np.array([inst[f] for f in meta_features], dtype=np.float32)
        X_seq.append(seq)
        X_meta.append(meta)
        y.append(inst["label"])
    return np.array(X_seq), np.array(X_meta), np.array(y, dtype=np.float32)


def cross_val_splits(n_samples, n_splits, val_ratio):
    for split in range(n_splits):
        train_end = int(n_samples * (1 - (n_splits - split) * val_ratio))
        val_end = train_end + int(n_samples * val_ratio)
        yield range(train_end), range(train_end, val_end)


def objective(trial):
    seq_len = trial.suggest_categorical("seq_len", [20, 25, 30, 35, 40, 45, 50])
    conv_filters = trial.suggest_categorical("conv_filters", [32, 48, 64, 96, 128])
    conv_kernel = trial.suggest_categorical("conv_kernel_size", [3, 5, 7, 9])
    lstm_hidden = trial.suggest_categorical("lstm_hidden", [64, 96, 128, 192, 256])
    lstm_layers = trial.suggest_categorical("lstm_layers", [1, 2])
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 48, 64, 96, 128])

    fold_scores = []
    for train_idxs, val_idxs in cross_val_splits(len(instances_tuning), 4, 0.1):
        train_insts = [instances_tuning[i] for i in train_idxs]
        val_insts   = [instances_tuning[i] for i in val_idxs]

        X_tr_seq, X_tr_meta, y_tr = build_sequences(df, train_insts, seq_len, seq_features, meta_features)
        X_vl_seq, X_vl_meta, y_vl = build_sequences(df, val_insts,   seq_len, seq_features, meta_features)

        if len(X_tr_seq) == 0 or len(X_vl_seq) == 0:
            continue

        # Normalize on train fold only
        seq_mean  = X_tr_seq.mean(axis=(0, 1), keepdims=True)
        seq_std   = X_tr_seq.std(axis=(0, 1),  keepdims=True) + 1e-8
        meta_mean = X_tr_meta.mean(axis=0, keepdims=True)
        meta_std  = X_tr_meta.std(axis=0,  keepdims=True) + 1e-8

        X_tr_seq  = (X_tr_seq  - seq_mean)  / seq_std
        X_vl_seq  = (X_vl_seq  - seq_mean)  / seq_std
        X_tr_meta = (X_tr_meta - meta_mean) / meta_std
        X_vl_meta = (X_vl_meta - meta_mean) / meta_std

        model = CnnLstm(
            n_seq_features=len(seq_features),
            n_meta_features=len(meta_features),
            conv_filters=conv_filters,
            conv_kernel_size=conv_kernel,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=dropout,
        ).to(device)

        pos_count = y_tr.sum()
        neg_count = len(y_tr) - pos_count
        pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        train_loader = DataLoader(EventDataset(X_tr_seq, X_tr_meta, y_tr), batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(EventDataset(X_vl_seq, X_vl_meta, y_vl), batch_size=batch_size)

        best_fold_ap = -1.0
        patience_counter = 0
        ema_ap = None

        for _ in range(50):
            model.train()
            for x_seq, x_meta, y_batch in train_loader:
                x_seq, x_meta, y_batch = x_seq.to(device), x_meta.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x_seq, x_meta), y_batch)
                loss.backward()
                optimizer.step()

            model.eval()
            val_logits, val_labels = [], []
            with torch.no_grad():
                for x_seq, x_meta, y_batch in val_loader:
                    val_logits.append(model(x_seq.to(device), x_meta.to(device)).cpu())
                    val_labels.append(y_batch)
            val_ap = average_precision_score(
                torch.cat(val_labels).numpy(),
                torch.sigmoid(torch.cat(val_logits)).numpy(),
            )

            ema_ap = val_ap if ema_ap is None else 0.3 * val_ap + 0.7 * ema_ap

            if ema_ap > best_fold_ap:
                best_fold_ap = ema_ap
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 8:
                    break

        fold_scores.append(best_fold_ap)
        torch.cuda.empty_cache()

    return np.mean(fold_scores) if fold_scores else 0.0


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=80, show_progress_bar=True)

# RECORD RESULTS
best = study.best_params
print("\nBest hyperparameters:")
print(pd.Series(best))

final_params = {
    "seq_len":          best["seq_len"],
    "conv_filters":     best["conv_filters"],
    "conv_kernel_size": best["conv_kernel_size"],
    "lstm_hidden":      best["lstm_hidden"],
    "lstm_layers":      best["lstm_layers"],
    "dropout":          round(best["dropout"], 6),
    "learning_rate":    round(best["learning_rate"], 6),
    "batch_size":       best["batch_size"],
    "epochs":           100,
    "patience":         15,
}

(Path(__file__).parent / "results").mkdir(exist_ok=True)
with open(Path(__file__).parent / f"results/{pattern}_tuned_params.json", "w") as f:
    json.dump(final_params, f, indent=4)
