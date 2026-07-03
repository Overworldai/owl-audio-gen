import torch
from torch import nn
import torch.nn.functional as F

import einops as eo

from .normalization import LayerNorm

class AdaLN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc = nn.Linear(config.d_model, 2 * config.d_model)
        self.norm = LayerNorm(config.d_model)

        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, cond):
        # x : [b,n,d]
        # cond : [b,d]
        cond = F.silu(cond)
        params = self.fc(cond)
        params = eo.repeat(params, 'b d -> b n d', n = x.shape[1])
        scale, shift = params.chunk(2, dim = -1) # both [b,n,d]
        return self.norm(x) * (1. + scale) + shift

class Gate(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.fc = nn.Linear(config.d_model, config.d_model)
        
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, cond):
        cond = F.silu(cond)
        gate = self.fc(cond)
        gate = eo.repeat(gate, 'b d -> b n d', n = x.shape[1])
        return x * gate