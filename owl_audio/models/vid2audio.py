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

        # Video conditioning
        video_patch_content = getattr(config, "video_patch_content", None)
        if video_patch_content is not None:
            self.video_proj = nn.Linear(video_patch_content, config.d_model)
            self.null_video = nn.Parameter(torch.randn(config.d_model) * 0.02)
        else:
            self.video_proj = None
            self.null_video = None

    def project_video(self, video):
        """Raw [B, T_v, C, H, W] -> projected [B, T_v, d_model]."""
        B, T_v, *_ = video.shape
        return self.video_proj(video.reshape(B, T_v, -1))

    def forward(self, x, ts, video=None, video_tokens=None, attn_mask=None):
        # x           : [B, C, T]
        # video       : [B, T_v, C_v, H_v, W_v] — raw, will be projected
        # video_tokens: [B, T_v, d_model]        — pre-projected, bypasses video_proj
        cond = self.ts_embed(ts)

        # Patchify audio: [B, C, n_p*p] -> [B, n_p, p*C]
        x = eo.rearrange(x, 'b c (n_p p) -> b n_p (p c)', p=self.patch_size)
        x = self.proj_in(x)  # [B, n_p, d_model]

        # T_v = 0
        # if video_tokens is not None:
        #     T_v = video_tokens.shape[1]
        #     x = torch.cat([video_tokens, x], dim=1)
        # elif video is not None and self.video_proj is not None:
        #     video_tokens = self.project_video(video)          # [B, T_v, d_model]
        #     T_v = video_tokens.shape[1]
        #     x = torch.cat([video_tokens, x], dim=1)

        x = self.dit(x, cond, attn_mask, video_tokens)
        x = self.norm_out(x, cond)
        x = self.proj_out(x)

        # if T_v > 0:
        #     x = x[:, T_v:, :]                                # strip video prefix

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

    def forward(self, x, video=None):
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

        # Project video once, then optionally swap for null tokens (CFG dropout)
        video_tokens = None
        if video is not None and self.core.video_proj is not None:
            video_tokens = self.core.project_video(video)    # [B, T_v, d_model]
            if self.training and torch.rand(1).item() < self.cfg_prob:
                B, T_v = video_tokens.shape[:2]
                video_tokens = self.core.null_video[None, None, :].expand(B, T_v, -1)

        pred = self.core(z, ts, video_tokens=video_tokens)
        if self.x0_mode:
            pred = (pred - z) / den

        loss = F.mse_loss(pred, target)
        return loss
