import numpy as np
import torch
import pandas as pd
from pathlib import Path

from api.patchtst import PatchTST
from api.cnn_lstm import CnnLstm

GATE_ARTIFACTS = Path("artifacts/gate")

patchTstGateVersion = 1

# Add new pattern names and their model versions here when deploying a new pattern model.
PATTERN_VERSIONS: dict[str, int] = {
    "fvg": 2,
}

_patchTstGateModel = None
_patchTstGateCheckpoint = None
_cnnLstmModels: dict[str, dict] = {}


def loadModels():
    global _patchTstGateModel, _patchTstGateCheckpoint

    checkpoint = torch.load(
        GATE_ARTIFACTS / f"PatchTST_EUR_USD_H1_2026_v{patchTstGateVersion}.pt",
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["config"]
    core_features = checkpoint["core_features"]
    corr_features = checkpoint["corr_features"]
    _patchTstGateModel = PatchTST(len(core_features), len(corr_features), config)
    _patchTstGateModel.load_state_dict(checkpoint["model_state_dict"])
    _patchTstGateModel.eval()
    _patchTstGateCheckpoint = checkpoint

    for name, version in PATTERN_VERSIONS.items():
        checkpoint = torch.load(
            Path(f"artifacts/{name}") / f"CNN-LSTM_EUR_USD_H1_2026_v{version}.pt",
            map_location="cpu",
            weights_only=False,
        )
        cfg = checkpoint["config"]
        seq_features = checkpoint["seq_features"]
        meta_features = checkpoint["meta_features"]
        model = CnnLstm(
            n_seq_features=len(seq_features),
            n_meta_features=len(meta_features),
            conv_filters=cfg["conv_filters"],
            conv_kernel_size=cfg["conv_kernel_size"],
            lstm_hidden=cfg["lstm_hidden"],
            lstm_layers=cfg["lstm_layers"],
            dropout=cfg["dropout"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        checkpoint["_model"] = model
        _cnnLstmModels[name] = checkpoint


def predictCnnLstm(name: str, df: pd.DataFrame, instance: dict, pred_labels: dict[int, str]) -> dict:
    checkpoint = _cnnLstmModels[name]
    seq_features = checkpoint["seq_features"]
    meta_features = checkpoint["meta_features"]
    norm = checkpoint["normalization"]
    seq_len = checkpoint["config"]["seq_len"]
    model = checkpoint["_model"]

    idx = instance["index"]
    seq = df.iloc[idx - seq_len + 1 : idx + 1][seq_features].values.astype(np.float32)
    seq_mean = np.array(norm["seq_mean"], dtype=np.float32).squeeze()
    seq_std  = np.array(norm["seq_std"],  dtype=np.float32).squeeze()
    seq = (seq - seq_mean) / seq_std

    meta = np.array([instance[f] for f in meta_features], dtype=np.float32)
    meta_mean = np.array(norm["meta_mean"], dtype=np.float32).squeeze()
    meta_std  = np.array(norm["meta_std"],  dtype=np.float32).squeeze()
    meta = (meta - meta_mean) / meta_std

    x_seq  = torch.tensor(seq).unsqueeze(0)
    x_meta = torch.tensor(meta).unsqueeze(0)

    with torch.no_grad():
        logit = model(x_seq, x_meta)
        prob_pos = torch.sigmoid(logit).item()

    prob_neg = 1.0 - prob_pos
    pred = pred_labels[1] if prob_pos >= 0.5 else pred_labels[0]
    return {
        "pred": pred,
        "probs": {"0": prob_neg, "1": prob_pos},
    }


def predictPatchTST(featuresDf: pd.DataFrame) -> dict:
    config = _patchTstGateCheckpoint["config"]
    core_features = _patchTstGateCheckpoint["core_features"]
    corr_features = _patchTstGateCheckpoint["corr_features"]
    norm = _patchTstGateCheckpoint["normalization"]

    lookback = config["lookback"]
    core_mean = np.array(norm["core_mean"], dtype=np.float32)
    core_std = np.array(norm["core_std"], dtype=np.float32)

    core_window = featuresDf[core_features].to_numpy(dtype=np.float32)[-lookback:]
    core_window = (core_window - core_mean) / core_std
    core_tensor = torch.tensor(core_window, dtype=torch.float32).unsqueeze(0)

    if corr_features:
        corr_mean = np.array(norm["corr_mean"], dtype=np.float32)
        corr_std = np.array(norm["corr_std"], dtype=np.float32)
        corr_window = featuresDf[corr_features].to_numpy(dtype=np.float32)[-lookback:]
        corr_window = (corr_window - corr_mean) / corr_std
        corr_tensor = torch.tensor(corr_window, dtype=torch.float32).unsqueeze(0)
    else:
        corr_tensor = torch.zeros(1, lookback, 0, dtype=torch.float32)

    with torch.no_grad():
        logit = _patchTstGateModel(core_tensor, corr_tensor)
        probDir = torch.sigmoid(logit).item()

    probFlat = 1.0 - probDir
    pred = "DIR" if probDir >= 0.5 else "FLAT"

    return {
        "patchTstGatePred": pred,
        "patchTstGateProbs": {"0": probFlat, "1": probDir},
    }
