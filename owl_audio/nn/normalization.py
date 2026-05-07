import torch
from torch import nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # small init to default to no gain
        self.gain = nn.Parameter(torch.randn(dim) * 0.02)

    def forward(self, x):
        orig_dtype = x.dtype
        b,h,n,d = x.shape
        gain = self.gain[None,None,None,:] # [1,1,1,d]

        gain = (1. + gain)
        # Keep everything in fp32 for stability
        x_f32 = x.float()
        rms = (x_f32.pow(2).mean(-1,keepdim=True)+1.0e-6).rsqrt()
        result = x_f32 * rms * gain.float()

        return result.to(orig_dtype)

def safe_normalize(x):
    orig_dtype = x.dtype
    return F.normalize(x.float(), p=2, dim=-1).to(orig_dtype)

class QKNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = RMSNorm(dim)

    def forward(self, q, k):
        return self.norm(q), self.norm(k)

def LayerNorm(dim):
    return nn.LayerNorm(dim, elementwise_affine = False)