import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_processing import dataparser
from patterns import registry
from symmetry import build_flip_mask, build_swap_indices, build_offset_indices, apply_flip
from classes import EventDataset, Tcn

SEED = 42
N_REPEATS = 5
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# LOAD CONFIGS
with open(Path(__file__).parent.parent / "env.json", "r") as f:
    env = json.load(f)
year_now = env["year_now"]
instrument = env["instrument"]
granularity = env["granularity"]
n = env["n_value"]
train_split = env["train_split"]
val_split = env["val_split"]
pattern = env["pattern"]
device = torch.device(env["device"] if torch.cuda.is_available() else "cpu")

# LOAD EVENT DETECTOR
pattern_module = registry.load(pattern)

# LOAD MODEL PARAMS
with open(Path(__file__).parent / f"model_configs/training_models/{pattern}_params.json") as f:
    params = json.load(f)
print(f"Loaded params from model_configs/training_models/{pattern}_params.json")
print("Hyperparameters:", params)

candidate_seq_features = [
    "open_return", "high_return", "low_return", "close_return", "vol_return",
    "atr_14", "volatility_regime",
    "bb_width", "bb_position",
    "hl_spread", "upper_wick", "lower_wick",
    "dist_ema15", "dist_ema50", "ema_cross",
    "dist_smooth14", "dist_smooth35", "smooth_cross",
    "rsi_14", "macd_hist", "vol_ratio", "vol_momentum", "adx", "di_diff",
    "fast_pct_R", "slow_pct_R",
    "tod_sin", "tod_cos",
    "body_to_range", "close_in_bar",
    "cum_return_3", "cum_return_6", "cum_return_12", "cum_return_24",
    "return_accel_3_12", "ema15_slope_3", "ema50_slope_5",
    "breakout_dist_high_24", "breakout_dist_low_24", "range_pos_24",
    "momentum_consistency_8", "vol_adj_return_6", "trend_pressure_8", "return_zscore_24",
]
meta_features = pattern_module.METADATA_FEATURES

# LOAD AND PARSE DATA
_raw_data_dir = Path(__file__).parent.parent.parent / "raw_data"
df = dataparser.parseData(_raw_data_dir / f"{instrument}_{granularity}_{year_now - 21}-01-01_{year_now}-04-01.json")

# DETECT AND LABEL
instances = pattern_module.detect(df)
labelled = pattern_module.label_instances(df, instances, n)
print(f"Detected {len(labelled)} {pattern.upper()} instances | fill rate: {sum(i['label'] for i in labelled) / len(labelled):.2%}")

# SPLIT
train_end = int(train_split * len(labelled))
val_end = int((train_split + val_split) * len(labelled))
train_instances = labelled[:train_end]
val_instances = labelled[train_end:val_end]

# BUILD SEQUENCES
flip_mask = build_flip_mask(candidate_seq_features)
swap_indices = build_swap_indices(candidate_seq_features)
offset_indices = build_offset_indices(candidate_seq_features)

def build_sequences(df, instances, seq_len, seq_features, meta_features):
    X_seq, X_meta, y = [], [], []
    for inst in instances:
        idx = inst["index"]
        if idx < seq_len - 1:
            continue
        seq = df.iloc[idx - seq_len + 1 : idx + 1][seq_features].values.astype(np.float32)
        seq = apply_flip(seq, inst["direction"], flip_mask, swap_indices, offset_indices)
        meta = np.array([inst[f] for f in meta_features], dtype=np.float32)
        X_seq.append(seq)
        X_meta.append(meta)
        y.append(inst["label"])
    return np.array(X_seq), np.array(X_meta), np.array(y, dtype=np.float32)

seq_len = params["seq_len"]
X_train_seq, X_train_meta, y_train = build_sequences(df, train_instances, seq_len, candidate_seq_features, meta_features)
X_val_seq,   X_val_meta,   y_val   = build_sequences(df, val_instances,   seq_len, candidate_seq_features, meta_features)

# NORMALIZE — fit on train only
seq_mean = X_train_seq.mean(axis=(0, 1), keepdims=True)
seq_std = X_train_seq.std(axis=(0, 1), keepdims=True) + 1e-8
meta_mean = X_train_meta.mean(axis=0, keepdims=True)
meta_std = X_train_meta.std(axis=0, keepdims=True) + 1e-8

X_train_seq = (X_train_seq - seq_mean) / seq_std
X_val_seq = (X_val_seq - seq_mean) / seq_std
X_train_meta = (X_train_meta - meta_mean) / meta_std
X_val_meta = (X_val_meta - meta_mean) / meta_std

print(f"Train: {len(X_train_seq)} | Val: {len(X_val_seq)}")

# TRAIN TEMP MODEL ON FULL CANDIDATE SET
model = Tcn(
    n_seq_features=len(candidate_seq_features),
    n_meta_features=len(meta_features),
    channels=params["channels"],
    kernel_size=params["kernel_size"],
    n_levels=params["n_levels"],
    dropout=params["dropout"],
    head_hidden=params.get("head_hidden"),
).to(device)

pos_count = y_train.sum()
neg_count = len(y_train) - pos_count
pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"], weight_decay=params.get("weight_decay", 0.0))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params["epochs"], eta_min=1e-6)

batch_size = params["batch_size"]
train_loader = DataLoader(EventDataset(X_train_seq, X_train_meta, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(EventDataset(X_val_seq, X_val_meta, y_val), batch_size=batch_size)

best_val_ap = -1.0
patience_counter = 0
best_state = None

for epoch in range(1, params["epochs"] + 1):
    model.train()
    for x_seq, x_meta, y_batch in train_loader:
        x_seq, x_meta, y_batch = x_seq.to(device), x_meta.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x_seq, x_meta), y_batch)
        loss.backward()
        optimizer.step()

    model.eval()
    val_logits_list, val_labels_list = [], []
    with torch.no_grad():
        for x_seq, x_meta, y_batch in val_loader:
            val_logits_list.append(model(x_seq.to(device), x_meta.to(device)).cpu())
            val_labels_list.append(y_batch)
    val_probs = torch.sigmoid(torch.cat(val_logits_list)).numpy()
    val_ap = average_precision_score(torch.cat(val_labels_list).numpy(), val_probs)
    scheduler.step()

    if val_ap > best_val_ap:
        best_val_ap = val_ap
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= params["patience"]:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Epoch {epoch:3d} | val AP: {val_ap:.4f} | best: {best_val_ap:.4f}")

model.load_state_dict(best_state)
model.eval()

# PERMUTATION IMPORTANCE
# For each feature, shuffle its values on the val set and measure the drop in val AP.
# Repeated N_REPEATS times per feature with different shuffles, then averaged.
def evaluate_val(X_seq, X_meta):
    loader = DataLoader(EventDataset(X_seq, X_meta, y_val), batch_size=batch_size)
    logits_list, labels_list = [], []
    with torch.no_grad():
        for x_seq, x_meta, y_batch in loader:
            logits_list.append(model(x_seq.to(device), x_meta.to(device)).cpu())
            labels_list.append(y_batch)
    probs = torch.sigmoid(torch.cat(logits_list)).numpy()
    return average_precision_score(torch.cat(labels_list).numpy(), probs)

baseline_ap = evaluate_val(X_val_seq, X_val_meta)
print(f"\nBaseline val AP (full feature set): {baseline_ap:.4f}")

print(f"\nComputing permutation importance ({N_REPEATS} repeats per feature)...")
rng = np.random.default_rng(SEED)
importances = {}

# SEQUENCE FEATURES — shuffle column across (samples × timesteps)
n_val, t_len, _ = X_val_seq.shape
for i, feat in enumerate(candidate_seq_features):
    drops = []
    for _ in range(N_REPEATS):
        X_perturb = X_val_seq.copy()
        flat = X_perturb[:, :, i].reshape(-1)
        X_perturb[:, :, i] = rng.permutation(flat).reshape(n_val, t_len)
        ap = evaluate_val(X_perturb, X_val_meta)
        drops.append(baseline_ap - ap)
    importances[feat] = float(np.mean(drops))
    print(f"  {feat:30s} ΔAP = {importances[feat]:+.5f}")

# META FEATURES — shuffle across samples
for i, feat in enumerate(meta_features):
    drops = []
    for _ in range(N_REPEATS):
        X_perturb = X_val_meta.copy()
        X_perturb[:, i] = rng.permutation(X_perturb[:, i])
        ap = evaluate_val(X_val_seq, X_perturb)
        drops.append(baseline_ap - ap)
    importances[feat] = float(np.mean(drops))
    print(f"  {feat:30s} ΔAP = {importances[feat]:+.5f}")

importances_series = pd.Series(importances).sort_values(ascending=False)
print(f"\n{importances_series}")

(Path(__file__).parent / "results").mkdir(exist_ok=True)
importances_series.to_json(Path(__file__).parent / f"results/{pattern}_feature_rankings.json", indent=4)
print(f"\nSaved rankings to results/{pattern}_feature_rankings.json")
