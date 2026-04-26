import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, precision_score, matthews_corrcoef, confusion_matrix
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from data_processing import dataparser
from patterns import registry
from classes import EventDataset, CnnLstm

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
pattern = env["pattern"]
log_metrics = env["log_metrics"]
device = torch.device(env["device"] if torch.cuda.is_available() else "cpu")
version = env["cnn_lstm"]["train_version"]

# LOAD EVENT DETECTOR
pattern_module = registry.load(pattern)

# LOAD MODEL CONFIGS
with open(Path(__file__).parent / f"model_configs/v{version}/{pattern}_features.json", "r") as f:
    best_features = json.load(f)["features"]
print(f"Features ({len(best_features)}):", best_features)

with open(Path(__file__).parent / f"model_configs/v{version}/{pattern}_params.json", "r") as f:
    params = json.load(f)
seq_len = params["seq_len"]
print("Hyperparameters:", params)

# LOAD AND PARSE DATA
_raw_data_dir = Path(__file__).parent.parent.parent / "raw_data"
df = dataparser.parseData(_raw_data_dir / f"{instrument}_{granularity}_{year_now - 21}-01-01_{year_now}-04-01.json")

# DETECT AND LABEL INSTANCES
instances = pattern_module.detect(df)
labelled = pattern_module.label_instances(df, instances, n, k)
print(f"Detected {len(labelled)} {pattern.upper()} instances | fill rate: {sum(i['label'] for i in labelled) / len(labelled):.2%}")

# SPLIT INSTANCES (temporal order preserved)
train_end = int(train_split * len(labelled))
val_end = int((train_split + val_split) * len(labelled))
train_instances = labelled[:train_end]
val_instances = labelled[train_end:val_end]
test_instances = labelled[val_end:]

# SEPARATE METADATA FEATURES (event-level) FROM SEQUENCE FEATURES (per-candle)
# Metadata features only exist at the detection candle; they are injected after the LSTM.
meta_features = pattern_module.METADATA_FEATURES
seq_features = [f for f in best_features if f not in meta_features]

# BUILD SEQUENCES
# For each instance at df index i, extract the seq_len candles ending at i.
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

X_train_seq, X_train_meta, y_train = build_sequences(df, train_instances, seq_len, seq_features, meta_features)
X_val_seq, X_val_meta, y_val = build_sequences(df, val_instances, seq_len, seq_features, meta_features)
X_test_seq, X_test_meta, y_test = build_sequences(df, test_instances, seq_len, seq_features, meta_features)

# NORMALISE — fit on train only
seq_mean = X_train_seq.mean(axis=(0, 1), keepdims=True)
seq_std = X_train_seq.std(axis=(0, 1), keepdims=True) + 1e-8
meta_mean = X_train_meta.mean(axis=0, keepdims=True)
meta_std = X_train_meta.std(axis=0, keepdims=True) + 1e-8

X_train_seq = (X_train_seq - seq_mean) / seq_std
X_val_seq = (X_val_seq - seq_mean) / seq_std
X_test_seq = (X_test_seq - seq_mean) / seq_std
X_train_meta = (X_train_meta - meta_mean) / meta_std
X_val_meta = (X_val_meta - meta_mean) / meta_std
X_test_meta = (X_test_meta - meta_mean) / meta_std

print(f"Train: {len(X_train_seq)} | Val: {len(X_val_seq)} | Test: {len(X_test_seq)}")


model = CnnLstm(
    n_seq_features=len(seq_features),
    n_meta_features=len(meta_features),
    conv_filters=params["conv_filters"],
    conv_kernel_size=params["conv_kernel_size"],
    lstm_hidden=params["lstm_hidden"],
    lstm_layers=params["lstm_layers"],
    dropout=params["dropout"],
).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model parameters: {n_params:,}")

pos_count = y_train.sum()
neg_count = len(y_train) - pos_count
pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

batch_size = params["batch_size"]
epochs = params["epochs"]
patience = params["patience"]

train_loader = DataLoader(EventDataset(X_train_seq, X_train_meta, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(EventDataset(X_val_seq, X_val_meta, y_val), batch_size=batch_size)

# TRAINING LOOP
best_val_ap = -1.0
patience_counter = 0
best_state = None

for epoch in range(1, epochs + 1):
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
    scheduler.step(-val_ap)

    if val_ap > best_val_ap:
        best_val_ap = val_ap
        patience_counter = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | val AP: {val_ap:.4f} | best: {best_val_ap:.4f}")

model.load_state_dict(best_state)

# EVALUATE
def run_inference(loader):
    logits_list, labels_list = [], []
    model.eval()
    with torch.no_grad():
        for x_seq, x_meta, y_batch in loader:
            logits_list.append(model(x_seq.to(device), x_meta.to(device)).cpu())
            labels_list.append(y_batch)
    return torch.sigmoid(torch.cat(logits_list)).numpy(), torch.cat(labels_list).numpy()

test_loader = DataLoader(EventDataset(X_test_seq, X_test_meta, y_test), batch_size=batch_size)
train_loader_eval = DataLoader(EventDataset(X_train_seq, X_train_meta, y_train), batch_size=batch_size)

y_prob, y_test_np = run_inference(test_loader)
y_prob_train, y_train_np = run_inference(train_loader_eval)

y_pred = (y_prob >= 0.5).astype(int)
y_pred_train = (y_prob_train >= 0.5).astype(int)

avgPrecision = average_precision_score(y_test_np, y_prob)
precision1 = precision_score(y_test_np, y_pred)
mcc = matthews_corrcoef(y_test_np, y_pred)
train_mcc = matthews_corrcoef(y_train_np, y_pred_train)
cmatrix = confusion_matrix(y_test_np, y_pred)
cmatrix_df = pd.DataFrame(cmatrix, index=["Real 0", "Real 1"], columns=["Pred 0", "Pred 1"])
cmatrix_df["Count"] = cmatrix_df.sum(axis=1)
cmatrix_df.loc["Count"] = cmatrix_df.sum(axis=0)

total_candles = len(df)
instance_pct = len(labelled) / total_candles * 100

lines = []
lines.append(f"=== v{version} | {instrument} {granularity} | {pattern} | CNN-LSTM | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
lines.append(f"Pattern instances: {len(labelled)} / {total_candles} candles ({instance_pct:.1f}%)")
lines.append(f"Train instances: {len(X_train_seq)} | Val: {len(X_val_seq)} | Test: {len(X_test_seq)}")
lines.append(f"\nAverage precision: {avgPrecision:.4f}")
lines.append(f"Precision (fill): {precision1:.4f}")
lines.append(f"MCC: {mcc:.4f}")
lines.append(f"MCC (train set): {train_mcc:.4f}")
lines.append(f"\nConfusion matrix:\n{cmatrix_df}")
for true_class, name in enumerate(["no_fill", "fill"]):
    mask = y_test_np == true_class
    avg_p = y_prob[mask].mean()
    lines.append(f"True={name}: avg P(no_fill)={1 - avg_p:.3f} P(fill)={avg_p:.3f}")

output = "\n".join(lines) + "\n"
print(output)

# LOG METRICS
if log_metrics:
    (Path(__file__).parent / "results").mkdir(exist_ok=True)
    with open(Path(__file__).parent / "results/test_metrics.log", "a") as f:
        f.write(output + "\n")

# SAVE MODEL
artifact = {
    "config": params,
    "seq_features": seq_features,
    "meta_features": meta_features,
    "normalization": {
        "seq_mean": seq_mean.tolist(),
        "seq_std": seq_std.tolist(),
        "meta_mean": meta_mean.tolist(),
        "meta_std": meta_std.tolist(),
    },
    "model_state_dict": best_state,
}
model_path = Path(__file__).parent / f"models/{instrument}/{pattern}_CNN-LSTM_{instrument}_{granularity}_{year_now}_v{version}.pt"
model_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(artifact, model_path)
print(f"Saved to {model_path}")
