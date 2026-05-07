import torch
from torch import nn
import torch.nn.functional as F

from .mlp import MLP
from .nocast import NoCastModule

class SinCosEmbed(nn.Module):
    def __init__(self, dim, theta=300, mult=1000):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.mult = mult

    @torch.autocast("cuda", enabled=False)
    def forward(self, x):
        orig_dtype = x.dtype
        x = x.float()
        # Handle different input types
        if isinstance(x, float):
            x = torch.tensor([x], dtype=torch.float32)
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # Ensure x is at least 1D
        if x.dim() == 0:
            x = x.unsqueeze(0)

        # Handle [b,n] inputs
        reshape_out = False
        if x.dim() == 2:
            b, n = x.shape
            x = x.view(b*n)
            reshape_out = True

        x = x * self.mult

        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(self.theta, dtype=torch.float32)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)

        # Match device and dtype of input
        emb = emb.to(device=x.device, dtype=x.dtype)

        # Compute sin/cos embeddings
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)

        # Reshape back if needed
        if reshape_out:
            emb = emb.reshape(b, n, -1)

        return emb.to(orig_dtype)

class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.sincos = SinCosEmbed(512, theta=300, mult = 1000)
        self.mlp = MLP(512, dim * 4, dim)

    def forward(self, x):
        x = self.sincos(x)
        x = self.mlp(x)
        return x
