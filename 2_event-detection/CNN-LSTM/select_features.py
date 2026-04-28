import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
import shap
import pandas as pd
from pathlib import Path

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
version = env["cnn_lstm"]["train_version"]

# LOAD EVENT DETECTOR
pattern_module = registry.load(pattern)

# LOAD MODEL PARAMS — use versioned configs if available, else fall back to defaults
_params_path = Path(__file__).parent / f"model_configs/v{version}/{pattern}_params.json"
if _params_path.exists():
    with open(_params_path) as f:
        _p = json.load(f)
    SEQ_LEN       = _p["seq_len"]
    CONV_FILTERS  = _p["conv_filters"]
    CONV_KERNEL   = _p["conv_kernel_size"]
    LSTM_HIDDEN   = _p["lstm_hidden"]
    LSTM_LAYERS   = _p["lstm_layers"]
    DROPOUT       = _p["dropout"]
    LR            = _p["learning_rate"]
    BATCH_SIZE    = _p["batch_size"]
    print(f"Loaded params from model_configs/v{version}/{pattern}_params.json")
else:
    SEQ_LEN       = 20
    CONV_FILTERS  = 64
    CONV_KERNEL   = 3
    LSTM_HIDDEN   = 128
    LSTM_LAYERS   = 1
    DROPOUT       = 0.2
    LR            = 0.001
    BATCH_SIZE    = 64
    print(f"No model_configs found for v{version} — using default params")

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
labelled = pattern_module.label_instances(df, instances, n, k)
print(f"Detected {len(labelled)} {pattern.upper()} instances | fill rate: {sum(i['label'] for i in labelled) / len(labelled):.2%}")

# SPLIT
train_end = int(train_split * len(labelled))
val_end = int((train_split + val_split) * len(labelled))
train_instances = labelled[:train_end]
val_instances = labelled[train_end:val_end]

# BUILD SEQUENCES
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

X_train_seq, X_train_meta, y_train = build_sequences(df, train_instances, SEQ_LEN, candidate_seq_features, meta_features)
X_val_seq,   X_val_meta,   y_val   = build_sequences(df, val_instances,   SEQ_LEN, candidate_seq_features, meta_features)

# NORMALIZE
seq_mean = X_train_seq.mean(axis=(0, 1), keepdims=True)
seq_std = X_train_seq.std(axis=(0, 1), keepdims=True) + 1e-8
meta_mean = X_train_meta.mean(axis=0, keepdims=True)
meta_std = X_train_meta.std(axis=0, keepdims=True) + 1e-8

X_train_seq = (X_train_seq - seq_mean) / seq_std
X_val_seq = (X_val_seq - seq_mean) / seq_std
X_train_meta = (X_train_meta - meta_mean) / meta_std
X_val_meta = (X_val_meta - meta_mean) / meta_std

print(f"Train: {len(X_train_seq)} | Val: {len(X_val_seq)}")

# TRAIN TEMP MODEL (CPU only — required for SHAP GradientExplainer)
device = torch.device("cpu")

model = CnnLstm(
    n_seq_features=len(candidate_seq_features),
    n_meta_features=len(meta_features),
    conv_filters=CONV_FILTERS,
    conv_kernel_size=CONV_KERNEL,
    lstm_hidden=LSTM_HIDDEN,
    lstm_layers=LSTM_LAYERS,
    dropout=DROPOUT,
).to(device)

pos_count = y_train.sum()
neg_count = len(y_train) - pos_count
pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

train_loader = DataLoader(EventDataset(X_train_seq, X_train_meta, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(EventDataset(X_val_seq, X_val_meta, y_val), batch_size=BATCH_SIZE)

best_val_ap = -1.0
patience_counter = 0
best_state = None
PATIENCE = 10

for epoch in range(1, 61):
    model.train()
    for x_seq, x_meta, y_batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(x_seq, x_meta), y_batch)
        loss.backward()
        optimizer.step()

    model.eval()
    val_logits_list, val_labels_list = [], []
    with torch.no_grad():
        for x_seq, x_meta, y_batch in val_loader:
            val_logits_list.append(model(x_seq, x_meta))
            val_labels_list.append(y_batch)
    val_probs = torch.sigmoid(torch.cat(val_logits_list)).numpy()
    val_ap = average_precision_score(torch.cat(val_labels_list).numpy(), val_probs)

    if val_ap > best_val_ap:
        best_val_ap = val_ap
        best_state = {key: val.clone() for key, val in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | val AP: {val_ap:.4f} | best: {best_val_ap:.4f}")

model.load_state_dict(best_state)
model.eval()

# SHAP — GradientExplainer with multi-input model
# shap_values returns [shap_seq, shap_meta]:
#   shap_seq:  (n_explain, seq_len, n_seq_features)
#   shap_meta: (n_explain, n_meta_features)
# Average |SHAP| over (samples, timesteps) for seq, over samples for meta.
N_BACKGROUND = 100
N_EXPLAIN = min(200, len(X_val_seq))

bg_seq = torch.tensor(X_train_seq[:N_BACKGROUND])
bg_meta = torch.tensor(X_train_meta[:N_BACKGROUND])
ex_seq = torch.tensor(X_val_seq[:N_EXPLAIN])
ex_meta = torch.tensor(X_val_meta[:N_EXPLAIN])

print("Computing SHAP values...")

# GradientExplainer requires (batch, n_outputs) — wrap to unsqueeze the scalar output
class _Wrapper(nn.Module):
    def forward(self, x_seq, x_meta):
        return model(x_seq, x_meta).unsqueeze(1)

explainer = shap.GradientExplainer(_Wrapper(), [bg_seq, bg_meta])
shap_values = explainer.shap_values([ex_seq, ex_meta])

# shap_values is [shap_seq, shap_meta] — one array per input
shap_seq, shap_meta = shap_values[0], shap_values[1]

shap_seq_arr = np.abs(shap_seq).mean(axis=(0, 1)).flatten()
shap_meta_arr = np.abs(shap_meta).mean(axis=0).flatten()

importances = pd.Series(
    np.concatenate([shap_seq_arr, shap_meta_arr]),
    index=candidate_seq_features + meta_features,
)
importances.sort_values(ascending=False, inplace=True)
print(f"\n{importances}")

(Path(__file__).parent / "results").mkdir(exist_ok=True)
importances.to_json(Path(__file__).parent / f"results/{pattern}_feature_rankings.json", indent=4)
