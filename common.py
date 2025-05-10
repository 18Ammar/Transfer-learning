import torch
import torch.nn as nn

class HEMaxBlock(nn.Module):
    def __init__(self, beta=1.5):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        max_val, _ = torch.max(x.view(x.size(0), x.size(1), -1), dim=2)
        max_val = max_val.view(x.size(0), x.size(1), 1, 1)
        mask = (x == max_val)
        x[mask] *= self.beta
        return x
