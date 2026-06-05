import torch
from torch import nn
import torch.nn.functional as F

import einops as eo
from .modulation import AdaLN, Gate
from .attn import Attn, CrossAttn, CrossAttnVid
from .mlp import MLP

class DiTBlock(nn.Module):
    """
    DiT block following SD3 conventions
    """
    def __init__(self, config):
        super().__init__()

        self.attn = Attn(config)
        self.mlp = MLP(
            config.d_model,
            config.mlp_ratio * config.d_model
        )
        self.adaln1 = AdaLN(config)
        self.gate1 = Gate(config)
        self.adaln2 = AdaLN(config)
        self.gate2 = Gate(config)

        # text cross attn
        if config.d_text > 0:
            self.cross_attn_text = CrossAttn(config)
            self.adaln_cross_text = AdaLN(config)
            self.gate_cross_text = Gate(config)
        else:
            self.cross_attn_text = None
            self.adaln_cross_text = None
            self.gate_cross_text = None

        # video cross attn
        if config.video_patch_content > 0:
            self.cross_attn_video = CrossAttnVid(config)
            self.adaln_cross_video = AdaLN(config)
            self.gate_cross_video = Gate(config)
        else:
            self.cross_attn_video = None
            self.adaln_cross_video = None
            self.gate_cross_video = None
    
    def forward(self, x, cond, video_tokens, text_tokens, attn_mask):
        # x : [b,n*p,d]
        # cond : [b,n,d]
        # attn_mask : flex attn block mask
        # timestamps : [b,n]
        res1 = x.clone()
        x = self.adaln1(x, cond)
        x = self.attn(x, attn_mask)
        x = self.gate1(x, cond)
        x = res1 + x

        if video_tokens is not None and self.cross_attn_video is not None:
            res2 = x.clone()
            x = self.adaln_cross_video(x, cond)
            x = self.cross_attn_video(x, video_tokens)
            x = self.gate_cross_video(x, cond)
            x = res2 + x

        if text_tokens is not None and self.cross_attn is not None:
            res3 = x.clone()
            x = self.adaln_cross_text(x, cond)
            x = self.cross_attn_text(x, text_tokens)
            x = self.gate_cross_text(x, cond)
            x = res3 + x

        res4 = x.clone()
        x = self.adaln2(x, cond)
        x = self.mlp(x)
        x = self.gate2(x, cond)
        x = res4 + x

        return x

class DiT(nn.Module):
    """
    Multiple DiT blocks stacked together
    """
    def __init__(self, config):
        super().__init__()

        self.blocks = []
        for _ in range(config.n_layers):
            self.blocks.append(DiTBlock(config))
        self.blocks = nn.ModuleList(self.blocks)
    
    def forward(self, x, cond, video_tokens=None, text_tokens=None, attn_mask=None):
        for block in self.blocks:
            x = block(x, cond, video_tokens, text_tokens, attn_mask)
        return x

class UViT(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.n_layers % 2 == 1 # Odd layer number required

        early_layers = config.n_layers // 2
        late_layers = early_layers

        self.early = []
        for _ in range(early_layers):
            self.early.append(DiTBlock(config))
        self.early = nn.ModuleList(self.early)

        self.middle = DiTBlock(config)

        self.late = []
        for _ in range(late_layers):
            self.late.append(DiTBlock(config))
        self.late = nn.ModuleList(self.late)
    
    def forward(self, x, cond, attn_mask):
        residuals = []
        for block in self.early:
            x = block(x, cond, attn_mask)
            residuals.append(x.clone())

        x = self.middle(x, cond, attn_mask)

        for block, residual in zip(self.late, residuals):
            x = block(x+residual, cond, attn_mask)

        return x, residuals