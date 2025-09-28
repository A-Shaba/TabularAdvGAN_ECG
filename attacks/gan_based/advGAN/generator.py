import torch.nn as nn

class AdvGANGeneratorTabular(nn.Module):
    def __init__(self, in_dim, hidden=[256, 128, 64]):  # updated default
        super().__init__()
        layers = []
        cur = in_dim
        for h in hidden:
            layers.append(nn.Linear(cur, h))
            layers.append(nn.ReLU())
            cur = h
        layers.append(nn.Linear(cur, in_dim))  # output perturbation δ
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
