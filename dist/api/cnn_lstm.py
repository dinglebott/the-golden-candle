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


class CnnLstm(nn.Module):
    def __init__(self, n_seq_features, n_meta_features, conv_filters, conv_kernel_size, lstm_hidden, lstm_layers, dropout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_seq_features, conv_filters, conv_kernel_size, padding=conv_kernel_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_filters, conv_filters, conv_kernel_size, padding=conv_kernel_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            conv_filters, lstm_hidden, lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden + n_meta_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_seq, x_meta):
        x = x_seq.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]
        combined = torch.cat([h, x_meta], dim=1)
        return self.head(combined).squeeze(1)
