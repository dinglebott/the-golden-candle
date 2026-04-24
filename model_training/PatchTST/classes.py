from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


@dataclass
class SplitData:
    core_features: np.ndarray
    corr_features: np.ndarray | None
    targets: np.ndarray
    target_indices: np.ndarray


class WindowedTimeSeriesDataset(Dataset):
    def __init__(self, split_data, lookback, core_mean, core_std, corr_mean=None, corr_std=None):
        self.core_features = split_data.core_features
        self.corr_features = split_data.corr_features
        self.targets = split_data.targets
        self.target_indices = split_data.target_indices
        self.lookback = lookback
        self.core_mean = core_mean
        self.core_std = core_std
        self.corr_mean = corr_mean
        self.corr_std = corr_std

    def __len__(self):
        return len(self.target_indices)

    def __getitem__(self, idx):
        end_idx = self.target_indices[idx]
        start_idx = end_idx - self.lookback + 1
        core_window = self.core_features[start_idx:end_idx + 1]
        core_window = (core_window - self.core_mean) / self.core_std

        if self.corr_features is None:
            corr_window = np.empty((self.lookback, 0), dtype=np.float32)
        else:
            corr_window = self.corr_features[start_idx:end_idx + 1]
            corr_window = (corr_window - self.corr_mean) / self.corr_std

        return {
            "core_inputs": torch.tensor(core_window, dtype=torch.float32),
            "corr_inputs": torch.tensor(corr_window, dtype=torch.float32),
            "target": torch.tensor(self.targets[end_idx], dtype=torch.float32),
        }


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, mlp_ratio, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        hidden_dim = d_model * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_input = self.norm1(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.dropout1(attn_output)
        x = x + self.mlp(self.norm2(x))
        return x


class PatchTST(nn.Module):
    def __init__(self, num_core_features, corr_input_dim, config):
        super().__init__()
        self.patch_len = config["patch_len"]
        self.patch_stride = config["patch_stride"]
        self.d_model = config["d_model"]
        self.corr_input_dim = corr_input_dim

        num_patches = 1 + (config["lookback"] - self.patch_len) // self.patch_stride
        if num_patches <= 0:
            raise ValueError("lookback must be at least as large as patch_len")

        core_patch_dim = num_core_features * self.patch_len
        self.input_projection = nn.Linear(core_patch_dim, self.d_model)
        self.positional_embedding = nn.Parameter(torch.zeros(1, num_patches, self.d_model))
        self.input_dropout = nn.Dropout(config["dropout"])

        if corr_input_dim > 0:
            corr_patch_dim = corr_input_dim * self.patch_len
            self.corr_projection = nn.Linear(corr_patch_dim, self.d_model)
            self.corr_adapter = nn.Sequential(
                nn.LayerNorm(self.d_model),
                nn.Linear(self.d_model, self.d_model),
                nn.GELU(),
                nn.Dropout(config["dropout"]),
            )
        else:
            self.corr_projection = None
            self.corr_adapter = None

        self.shared_blocks = nn.ModuleList([
            EncoderBlock(self.d_model, config["num_heads"], config["mlp_ratio"], config["dropout"])
            for _ in range(config["base_encoder_blocks"])
        ])
        self.task_blocks = nn.ModuleList([
            EncoderBlock(self.d_model, config["num_heads"], config["mlp_ratio"], config["dropout"])
            for _ in range(config["branch_encoder_blocks"])
        ])

        head_hidden = max(self.d_model // 2, 32)
        self.head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(head_hidden, 1),
        )

        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

    def patchify(self, x):
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.patch_stride)
        patches = patches.permute(0, 1, 3, 2).contiguous()
        return patches.view(x.size(0), patches.size(1), -1)

    def _set_requires_grad(self, module, flag):
        for parameter in module.parameters():
            parameter.requires_grad = flag

    def freeze_for_finetune(self, unfreeze_shared_blocks=0):
        for parameter in self.parameters():
            parameter.requires_grad = False

        if unfreeze_shared_blocks > 0:
            for block in self.shared_blocks[-unfreeze_shared_blocks:]:
                self._set_requires_grad(block, True)
            self.positional_embedding.requires_grad = True

        for block in self.task_blocks:
            self._set_requires_grad(block, True)
        self._set_requires_grad(self.head, True)

        if self.corr_projection is not None:
            self._set_requires_grad(self.corr_projection, True)
            self._set_requires_grad(self.corr_adapter, True)

    def forward(self, core_inputs, corr_inputs=None):
        x = self.patchify(core_inputs)
        x = self.input_projection(x)
        x = self.input_dropout(x + self.positional_embedding[:, :x.size(1)])

        for block in self.shared_blocks:
            x = block(x)

        corr_context = 0.0
        if (
            self.corr_projection is not None
            and corr_inputs is not None
            and corr_inputs.size(-1) > 0
        ):
            corr_context = self.patchify(corr_inputs)
            corr_context = self.corr_projection(corr_context)
            corr_context = self.corr_adapter(corr_context)

        x = x + corr_context
        for block in self.task_blocks:
            x = block(x)

        return self.head(x.mean(dim=1)).squeeze(-1)
