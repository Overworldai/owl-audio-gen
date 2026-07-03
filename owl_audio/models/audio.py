import torch
from torch import nn
import torch.nn.functional as F

import einops as eo

from ..configs import Config
from ..nn.dit import DiT
from ..nn.embeddings import TimestepEmbedding
from ..nn.modulation import AdaLN
from ..utils import int_to_tuple

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

        self._init_text_cond(config)
        self._init_video_cond(config)

    def _init_text_cond(self, config):
        has_text = config.d_text > 0
        if has_text:
            self.text_proj = nn.Linear(config.d_text, config.d_model) if has_text else None
            self.null_text = nn.Parameter(torch.randn(config.d_model) * 0.02) if has_text else None
        else:
            self.text_proj = None
            self.null_text = None

    def _init_video_cond(self, config):
        has_video = config.video_channels > 0
        if has_video:
            self.v_patch_h, self.v_patch_w = config.video_patch_spatial
            patch_in = self.v_patch_h * self.v_patch_w * config.video_channels
            self.video_proj = nn.Sequential(
                nn.Linear(patch_in, 4 * config.d_model),
                nn.GELU(),
                nn.Linear(4 * config.d_model, config.d_model),
                nn.LayerNorm(config.d_model),
                nn.Dropout(0.1)
            )
            
            h_lat, w_lat = config.video_latent_hw
            self.t_video = int(config.video_sr * config.window_length)
            self.n_spatial = (h_lat // self.v_patch_h) * (w_lat // self.v_patch_w)
            n_video = self.t_video * self.n_spatial   # 75*(20/5)*(20/5) = 75*16 = 1200
            self.null_video = nn.Parameter(torch.randn(n_video, config.d_model) * 0.02)
            self.spatial_emb = nn.Parameter(torch.randn(self.n_spatial, config.d_model) * 0.02)
        else:
            self.video_proj = None
            self.null_video = None
            self.spatial_emb = None
            self.v_patch_h = self.v_patch_w = 1

    def project_video(self, video):
        """[B, T_v, C, H, W] -> [B, N_v, d_model]"""
        video = eo.rearrange(
            video, 'b t c (n_h p_h) (n_w p_w) -> b (t n_h n_w) (c p_h p_w)',
            p_h=self.v_patch_h, p_w=self.v_patch_w,
        )
        video_tokens = self.video_proj(video)
        return self.add_spatial_emb(video_tokens)

    # adding learned spatial embedding
    def add_spatial_emb(self, video_tokens):
        # video_tokens: [B, N_v, d_model]
        video_tokens = eo.rearrange(
            video_tokens, 'b (t n) d -> b t n d', 
            t=self.t_video, n=self.n_spatial            
        )
        video_tokens = video_tokens + self.spatial_emb[None, None, :, :]
        return eo.rearrange(video_tokens, 'b t n d -> b (t n) d')

    def forward(self, x, ts, video_tokens=None, text_tokens=None, attn_mask=None):
        # x            : [B, C, T]          
        # video_tokens : [B, N_v, d_model]    
        # text_tokens  : [B, T_txt, d_model]  
        cond = self.ts_embed(ts)

        x = eo.rearrange(x, 'b c (n p) -> b n (p c)', p=self.patch_size)
        x = self.proj_in(x)

        x = self.dit(x, cond, video_tokens, text_tokens, attn_mask)
        x = self.norm_out(x, cond)
        x = self.proj_out(x)

        x = eo.rearrange(x, 'b n (p c) -> b c (n p)', p=self.patch_size, c=self.channels)
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

    def forward(self, x, video=None, text_emb=None):
        # x    : [B, C, T]
        # video: [B, T_v, C_v, H_v, W_v] or None
        # text_emb: [B, T_txt, d_txt] or None
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

        # Get text tokens from text_emb
        text_tokens = None
        if text_emb is not None and self.core.text_proj is not None:
            text_tokens = self.core.text_proj(text_emb)    # [B, T_txt, d_model]

        # Project video once
        video_tokens = None
        if video is not None and self.core.video_proj is not None:
            video_tokens = self.core.project_video(video)    # [B, N_v, d_model]
      
        # Optionally swap for null tokens (CFG dropout)
        if self.training:
            if video_tokens is not None:
                B = video_tokens.shape[0]
                video_drop = torch.rand(B, 1, 1, device=video_tokens.device) < self.cfg_prob
                null_video = self.core.add_spatial_emb(self.core.null_video[None, :, :].expand(B, -1, -1))
                video_tokens = torch.where(video_drop, null_video, video_tokens)

            if text_tokens is not None:
                B, T_txt = text_tokens.shape[:2]
                text_drop = torch.rand(B, 1, 1, device=text_tokens.device) < self.cfg_prob
                null_text = self.core.null_text[None, None, :].expand(B, T_txt, -1)
                text_tokens = torch.where(text_drop, null_text, text_tokens)

        pred = self.core(z, ts, video_tokens, text_tokens)
        if self.x0_mode:
            pred = (pred - z) / den

        loss = F.mse_loss(pred, target)
        return loss
