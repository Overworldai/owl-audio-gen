import torch
from torch import nn
import torch.nn.functional as F

from .nocast import NoCastModule
from ..utils import int_to_tuple

from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

class ImageRoPE(nn.Module):
    def __init__(self, config):
        super().__init__()

        h, w = int_to_tuple(config.sample_size)
        p_y, p_x = int_to_tuple(config.patch_size)

        n_p_y = h // p_y
        n_p_x = w // p_x
        max_freq = min(n_p_y, n_p_x) * 0.8

        dim_head = config.d_model // config.n_heads
        rope_emb = RotaryEmbedding(
            dim_head // 4,
            freqs_for = 'pixel',
            max_freq = max_freq
        )
        freqs = rope_emb.get_axial_freqs(
            n_p_y,
            n_p_x
        )
        self.register_buffer('freqs', freqs, persistent=False)
        self.n_p_y = n_p_y
        self.n_p_x = n_p_x

    def apply(self, x):
        # Assume x is [b,h,n,d], must reshape to [b,h,n_p_y,n_p_x,d]
        orig_dtype = x.dtype
        b,h,n,d = x.shape
        x = x.view(b,h,self.n_p_y,self.n_p_x,d)
        # Keep in fp32 for stability, only cast back at the very end
        x = apply_rotary_emb(self.freqs.detach().float(), x.float())
        x = x.view(b,h,n,d)
        return x.to(orig_dtype)
    
    def forward(self, q, k):
        q = self.apply(q)
        k = self.apply(k)
        return q, k


class AudioRoPE(nn.Module):
    """1D RoPE for audio (and prefixed video) tokens — sequence-length agnostic."""
    def __init__(self, config):
        super().__init__()
        dim_head = config.d_model // config.n_heads
        self.rope = RotaryEmbedding(dim_head // 2, max_freq=300)

    def forward(self, q, k):
        q = self.rope.rotate_queries_or_keys(q.float()).to(q.dtype)
        k = self.rope.rotate_queries_or_keys(k.float()).to(k.dtype)
        return q, k


class VideoAudioRoPE(nn.Module):
    """1D RoPE for audio to video temporal alignment"""
    def __init__(self, config):
        super().__init__()
        dim_head = config.d_model // config.n_heads
        self.rope = RotaryEmbedding(dim_head // 2, max_freq=32)

        audio_dt = 1 / config.sample_rate
        video_dt = 1 / config.video_sr

        audio_times = torch.arange(config.window_length * config.sample_rate).float() * audio_dt
        video_times = torch.arange(config.window_length * config.video_sr).float() * video_dt

        audio_freqs = self.rope(audio_times)
        video_freqs = self.rope(video_times)

        self.register_buffer("audio_freqs", audio_freqs, persistent=False)
        self.register_buffer("video_freqs", video_freqs, persistent=False)

    def apply(self, x, freqs):
        orig_dtype = x.dtype
        x = apply_rotary_emb(freqs.detach().float(), x.float(), seq_dim=2)
        return x.to(orig_dtype)

    def forward(self, q, k):
        q = self.apply(q, self.audio_freqs)
        k = self.apply(k, self.video_freqs)
        return q, k

def get_rope_cls(name):
    return {"image": ImageRoPE, "audio": AudioRoPE, "vid2audio": VideoAudioRoPE}[name]