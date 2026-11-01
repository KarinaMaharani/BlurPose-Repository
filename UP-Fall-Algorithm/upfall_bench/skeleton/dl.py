"""Sequence models of the 3D-skeleton benchmark: LSTM, BiLSTM, 1D-CNN and a compact ST-GCN.
Input: (batch, T, J*D) sequences of joint coordinates."""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMNet(nn.Module):
    def __init__(self, in_dim, n_cls, hidden=128, layers=2, bidirectional=False, dropout=0.2):
        super().__init__()
        self.rnn = nn.LSTM(in_dim, hidden, layers, batch_first=True, bidirectional=bidirectional,
                           dropout=dropout if layers > 1 else 0)
        self.fc = nn.Linear(hidden * (2 if bidirectional else 1), n_cls)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1])


class CNN1D(nn.Module):
    def __init__(self, in_dim, n_cls, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch * 2, 5, padding=2), nn.BatchNorm1d(ch * 2), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1), nn.Flatten(), nn.Dropout(.3), nn.Linear(ch * 2, n_cls))

    def forward(self, x):
        return self.net(x.transpose(1, 2))


class STGCNBlock(nn.Module):
    def __init__(self, cin, cout, A, t_kernel=9, stride=1):
        super().__init__()
        self.register_buffer("A", A)
        self.gcn = nn.Conv2d(cin, cout, 1)
        self.tcn = nn.Sequential(nn.BatchNorm2d(cout), nn.ReLU(),
                                 nn.Conv2d(cout, cout, (t_kernel, 1), (stride, 1), (t_kernel // 2, 0)),
                                 nn.BatchNorm2d(cout))
        self.res = (nn.Identity() if cin == cout and stride == 1 else
                    nn.Sequential(nn.Conv2d(cin, cout, 1, (stride, 1)), nn.BatchNorm2d(cout)))
        self.relu = nn.ReLU()

    def forward(self, x):                      # x: (B, C, T, V)
        y = torch.einsum("bctv,vw->bctw", self.gcn(x), self.A)
        return self.relu(self.tcn(y) + self.res(x))


class STGCN(nn.Module):
    """Yan et al. 2018, reduced to 3 blocks with a single (uniform) adjacency partition."""

    def __init__(self, n_joints, dims, n_cls, A):
        super().__init__()
        A = torch.as_tensor(A, dtype=torch.float32)
        self.J, self.D = n_joints, dims
        self.bn = nn.BatchNorm1d(n_joints * dims)
        self.blocks = nn.Sequential(STGCNBlock(dims, 32, A), STGCNBlock(32, 64, A, stride=2),
                                    STGCNBlock(64, 64, A))
        self.fc = nn.Linear(64, n_cls)

    def forward(self, x):                      # (B, T, J*D)
        B, T, _ = x.shape
        x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        x = x.view(B, T, self.J, self.D).permute(0, 3, 1, 2)
        return self.fc(self.blocks(x).mean((2, 3)))


def build(name, in_dim, n_cls, n_joints, dims, A):
    return {"LSTM": lambda: LSTMNet(in_dim, n_cls),
            "BiLSTM": lambda: LSTMNet(in_dim, n_cls, bidirectional=True),
            "CNN": lambda: CNN1D(in_dim, n_cls),
            "STGCN": lambda: STGCN(n_joints, dims, n_cls, A)}[name]()
