# models/ecg_mlp.py
import torch.nn as nn
import torch

class ECGMLP(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=[128,64], dropout=0.1):
        super().__init__()
        layers = []
        cur = in_dim
        for h in hidden:
            layers.append(nn.Linear(cur, h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            cur = h
        layers.append(nn.Linear(cur, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (N, in_dim)
        return self.net(x)

class DeepECGMLP(ECGMLP):
    def __init__(self, in_dim, num_classes):
        super().__init__(in_dim, num_classes, hidden=[256,128,64], dropout=0.2)
