import torch
from torch import nn
import torch.nn.functional as F

import einops as eo

from .normalization import QKNorm
from .rope import get_rope_cls

torch.backends.cuda.enable_flash_sdp(enabled=True)

class Attn(nn.Module):
    def __init__(self, config):
        super().__init__()

        d_model = config.d_model
        n_heads = config.n_heads

        self.dim = d_model // n_heads
        self.n_heads = n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias = False)
        self.out = nn.Linear(d_model, d_model)

        self.qk_norm = QKNorm(self.dim)
        self.rope = get_rope_cls(getattr(config, "rope_cls", "image"))(config)

        #nn.init.zeros_(self.out)

    def forward(self, x, attn_mask=None):
        # x is [b,n,d]
        x = self.qkv(x)
        q,k,v = eo.rearrange(x, 'b n (three h d) -> three b h n d', three = 3, d = self.dim)
        q,k = self.qk_norm(q,k)
        q,k = self.rope(q,k)
        x_out = F.scaled_dot_product_attention(q,k,v)
        x_out = eo.rearrange(x_out, 'b h n d -> b n (h d)')
        return self.out(x_out)

class CrossAttn(nn.Module):
    def __init__(self, config):
        super().__init__()

        d_model = config.d_model
        n_heads = config.n_heads

        self.dim = d_model // n_heads
        self.n_heads = n_heads

        self.q = nn.Linear(d_model, d_model, bias = False)
        self.kv = nn.Linear(d_model, 2 * d_model, bias = False)
        self.out = nn.Linear(d_model, d_model)

        self.qk_norm = QKNorm(self.dim)

    def forward(self, x, cond, attn_mask=None):
        # x: [B, T_a, d_model]
        # cond: [B, T_txt, d_model]
        # attn_mask: [B, T_txt]
        q = eo.rearrange(self.q(x), 'b n (h d) -> b h n d', d = self.dim)
        k,v = eo.rearrange(self.kv(cond), 'b n (two h d) -> two b h n d', two = 2, d = self.dim)
        q,k = self.qk_norm(q,k)
        
        if attn_mask is not None:
            attn_mask = attn_mask[:, None, None, :].bool()

        x_out = F.scaled_dot_product_attention(q,k,v, attn_mask=attn_mask)
        x_out = eo.rearrange(x_out, 'b h n d -> b n (h d)')
        return self.out(x_out)

class CrossAttnVid(nn.Module):
    """Cross attn on video tokens with RoPE temporal alignemnt"""
    def __init__(self, config):
        super().__init__()

        d_model = config.d_model
        n_heads = config.n_heads

        self.dim = d_model // n_heads
        self.n_heads = n_heads

        self.q = nn.Linear(d_model, d_model, bias = False)
        self.kv = nn.Linear(d_model, 2 * d_model, bias = False)
        self.out = nn.Linear(d_model, d_model)

        self.qk_norm = QKNorm(self.dim)
        self.rope = get_rope_cls("vid2audio")(config)

    def forward(self, x, video_tokens):
        # x is [b, T_a, d]
        # video_tokens : [b, T_v, d]
        q = self.q(x)
        q = eo.rearrange(q, 'b n (h d) -> b h n d', d = self.dim)
        k,v = eo.rearrange(self.kv(video_tokens), 'b n (two h d) -> two b h n d', two = 2, d = self.dim)
        q,k = self.qk_norm(q,k)
        q,k = self.rope(q,k)
        x_out = F.scaled_dot_product_attention(q,k,v)
        x_out = eo.rearrange(x_out, 'b h n d -> b n (h d)')
        return self.out(x_out)