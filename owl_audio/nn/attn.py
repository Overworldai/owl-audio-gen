import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

import einops as eo

from ..utils import int_to_tuple
from .normalization import QKNorm
from .rope import get_rope_cls, AudioRoPE

torch.backends.cuda.enable_flash_sdp(enabled=True)
flex_attention = torch.compile(flex_attention)

def get_attn_mask(
    config,
    batch_size = None,
    device = None
):
    """
    This will assume combined latents
    """
    #kernel_size = getattr(config, "kernel_size", None)
    kernel_size = [1,1]

    h,w = int_to_tuple(config.sample_size)
    p_y, p_x = int_to_tuple(config.patch_size)
    if kernel_size is not None:
        k_y, k_x = int_to_tuple(kernel_size)

    n_p_y = h // p_y
    n_p_x = w // p_x
    n_tokens = n_p_y * n_p_x

    max_q_len = n_p_y * n_p_x
    max_kv_len = n_p_y * n_p_x

    def row_idx(idx):
        return idx // n_p_x
    
    def col_idx(idx):
        return idx % n_p_x

    def can_attend_to(b,h,idx_i, idx_j):
        row_idx_i = row_idx(idx_i)
        col_idx_i = col_idx(idx_i)

        row_idx_j = row_idx(idx_j)
        col_idx_j = col_idx(idx_j)

        if kernel_size is None:
            return torch.ones_like(idx_j, dtype = torch.bool)
        else:
            # If i is image, it can attend to images in neighbourhood
            image_nbr_mask = (torch.abs(row_idx_j - row_idx_i) <= k_y) & (torch.abs(col_idx_j - col_idx_i) <= k_x)

            return image_nbr_mask

    return create_block_mask(
        can_attend_to,
        B=batch_size,
        H=config.n_heads,
        Q_LEN=max_q_len,
        KV_LEN=max_kv_len,
        device=device,
    )

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
        x_out = flex_attention(q,k,v, block_mask=attn_mask)
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