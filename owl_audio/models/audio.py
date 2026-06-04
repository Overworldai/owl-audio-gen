import torch
from torch import nn
import torch.nn.functional as F

import einops as eo

from ..configs import Config
from ..nn.dit import DiT
from ..nn.embeddings import TimestepEmbedding
from ..nn.modulation import AdaLN

"""
Audio diffusion transformer model
"""

class AudioDiffusionCore(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        p = config.patch_size
        self.patch_size = int(p[0]) if isinstance(p, (list, tuple)) else int(p)
        self.channels = config.channels
        self.patch_content = self.patch_size * config.channels

        self.proj_in = nn.Linear(self.patch_content, config.d_model)
        self.dit = DiT(config)
        self.norm_out = AdaLN(config)
        self.proj_out = nn.Linear(config.d_model, self.patch_content)

        self.ts_embed = TimestepEmbedding(config.d_model)

        # text conditioning 
        if config.d_text > 0:
            self.text_proj = nn.Linear(config.d_text, config.d_model)
            self.null_text = nn.Parameter(torch.randn(config.d_model) * 0.02)
        else:
            self.text_proj = None
            self.null_text = None

    def forward(self, x, ts, text_emb, text_mask=None):
        # x           : [B, C, T]
        # video       : [B, T_v, C_v, H_v, W_v] — raw, will be projected
        # video_tokens: [B, T_v, d_model]        — pre-projected, bypasses video_proj
        cond = self.ts_embed(ts)

        # Patchify audio: [B, C, n_p*p] -> [B, n_p, p*C]
        x = eo.rearrange(x, 'b c (n_p p) -> b n_p (p c)', p=self.patch_size)
        x = self.proj_in(x)  # [B, n_p, d_model]

        T_txt = 0
        if text_emb is not None:
            T_txt = text_emb.shape[1]

        text_tokens = self.text_proj(text_emb)  # [B, seq, D]
        text_tokens = text_tokens * text_mask.unsqueeze(-1).float()
        x = torch.cat([text_tokens, x], dim=1) # [B, seq+T, D]

        x = self.dit(x, cond, attn_mask=None)
        x = self.norm_out(x, cond)
        x = self.proj_out(x)

        # slice out only audio portion 
        if T_txt > 0:
            x = x[:, T_txt:, :]

        # Depatchify: [B, n_p, p*C] -> [B, C, n_p*p]
        x = eo.rearrange(x, 'b n_p (p c) -> b c (n_p p)', p=self.patch_size, c=self.channels)

        return x


class AudioDiffusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.core = AudioDiffusionCore(config)
        self.config = config

        self.ts_mu = 0.4
        self.ts_sigma = 1.0
        self.x0_mode = getattr(config, "x0_mode", False)
        self.noise_scale = getattr(config, "noise_scale", 1.0)
        self.cfg_prob = getattr(config, "cfg_prob", 0.1)

        if self.x0_mode:
            self.ts_mu = -0.8
            self.ts_sigma = 0.8

    @torch.no_grad()
    def sample_timesteps(self, b, device, dtype):
        ts = torch.randn(b, device=device, dtype=dtype)
        ts = ts * self.ts_sigma - self.ts_mu
        ts = ts.sigmoid()
        return ts

    def forward(self, x, text_emb, text_mask):
        # x    : [B, C, T]
        # video: [B, T_v, C_v, H_v, W_v] or None
        with torch.no_grad():
            ts = self.sample_timesteps(x.shape[0], x.device, x.dtype)
            eps = torch.randn_like(x) * self.noise_scale

            ts_exp = ts.view(-1, 1, 1).expand_as(x)

            if self.x0_mode:
                z = ts_exp * x + (1. - ts_exp) * eps
                den = (1. - ts_exp).clamp(min=0.05)
                target = (x - z) / den
            else:
                z = (1. - ts_exp) * x + ts_exp * eps
                target = eps - x

        text_tokens = None
        if text_emb is not None and self.core.text_proj is not None:
            text_tokens = self.core.text_proj(text_emb)    # [B, seq, d_model]
            if self.training and torch.rand(1).item() < self.cfg_prob:
                B, T_txt = text_tokens.shape[:2]
                text_tokens = self.core.null_text[None, None, :].expand(B, T_txt, -1)

        pred = self.core(z, ts, text_emb, text_mask)
        if self.x0_mode:
            pred = (pred - z) / den

        loss = F.mse_loss(pred, target)
        return loss
