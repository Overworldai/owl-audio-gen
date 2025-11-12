import torch
from torch import nn
import torch.nn.functional as F

from .. import nn as owl_nn
from ..nn.embeddings import TimestepEmbedding
from ..nn.attn import DiT
import einops as eo
from einops._torch_specific import allow_ops_in_compiled_graph
allow_ops_in_compiled_graph()
class AudioRFTCore(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        assert cfg.backbone == "dit"
        self.transformer = DiT(cfg)

        self.proj_in = nn.Linear(cfg.sample_size, cfg.d_model, bias=False)
        self.proj_out = nn.Linear(cfg.d_model, cfg.sample_size, bias=False)

        self.t_embed = TimestepEmbedding(cfg.d_model)
        self.uncond = cfg.uncond
    def forward(self, x, ts):
        B, N, D = x.shape
        
        x_tokens = self.proj_in(x)
        cond = self.t_embed(ts)
        # print(f"ts shape: {ts.shape}")
        # print(f"cond shape: {cond.shape}, x shape: {x.shape}")
        x = self.transformer(x_tokens, cond)
        x = F.silu(x)
        x = self.proj_out(x)
        return x 

class AudioRFT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.core = AudioRFTCore(cfg)

    # def noise(self, tensor, ts):
    #     z = torch.randn_like(tensor)
    #     lerp = tensor * (1 - ts) + z * ts
    #     return lerp, z - tensor, z

    def forward(self, x):
        with torch.no_grad():
            B, S, _ = x.shape
            ts = torch.randn(B, S, device = x.device, dtype = x.dtype).sigmoid() # logit normal
            ts_exp = ts[:, :,None]

            z = torch.randn_like(x)
            noisy = x * (1 - ts_exp) + ts_exp * z

        pred = self.core(noisy, ts)
        # print(f"pred shape: {pred.shape}, z shape: {z.shape}")
        loss = F.mse_loss(pred, z)
        return loss