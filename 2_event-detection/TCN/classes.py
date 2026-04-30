import torch
import torch.nn as nn
from torch.utils.data import Dataset


class EventDataset(Dataset):
    def __init__(self, X_seq, X_meta, y):
        self.X_seq = torch.tensor(X_seq)
        self.X_meta = torch.tensor(X_meta)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X_seq[i], self.X_meta[i], self.y[i]


class _Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x


class _TemporalBlock(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(n_in, n_out, kernel_size, padding=padding, dilation=dilation),
            _Chomp1d(padding),
            nn.BatchNorm1d(n_out),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(n_out, n_out, kernel_size, padding=padding, dilation=dilation),
            _Chomp1d(padding),
            nn.BatchNorm1d(n_out),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Tcn(nn.Module):
    def __init__(self, n_seq_features, n_meta_features, channels, kernel_size, n_levels, dropout, head_hidden=None):
        super().__init__()
        blocks = []
        for i in range(n_levels):
            dilation = 2 ** i
            in_ch = n_seq_features if i == 0 else channels
            blocks.append(_TemporalBlock(in_ch, channels, kernel_size, dilation, dropout))
        self.tcn = nn.Sequential(*blocks)
        if head_hidden is None:
            self.head = nn.Linear(channels + n_meta_features, 1)
        else:
            self.head = nn.Sequential(
                nn.Linear(channels + n_meta_features, head_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),
            )

    def forward(self, x_seq, x_meta):
        # x_seq: (batch, seq_len, n_seq_features)
        x = x_seq.permute(0, 2, 1)         # (batch, n_seq_features, seq_len)
        x = self.tcn(x)                     # (batch, channels, seq_len)
        h = x[:, :, -1]                    # (batch, channels) — last (causal) timestep
        combined = torch.cat([h, x_meta], dim=1)
        return self.head(combined).squeeze(1)  # (batch,) raw logits
