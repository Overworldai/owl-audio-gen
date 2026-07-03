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
    """1D RoPE for audio-video temporal alignment, computed directly in continuous seconds."""
    def __init__(self, config):
        super().__init__()
        self.dim = config.d_model // config.n_heads

        self.patch_size = getattr(config, 'patch_size', 1)
        self.sample_rate = config.sample_rate
        self.video_sr = config.video_sr
        self.window_length = config.window_length
        self.n_video_frames = int(self.window_length * self.video_sr)
        theta = 8   # theta=8 performs well for window_length = 10.0

        inv_freq = self._build_inv_freq(theta)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _build_inv_freq(self, theta):
        i = torch.arange(0, self.dim, 2).float()
        return theta ** (-i / self.dim)

    def _angles_to_tables(self, pos):
        angles = pos[:, None] * self.inv_freq[None, :]
        angles = torch.cat([angles, angles], dim=-1)
        return angles.cos(), angles.sin()

    @staticmethod
    def _rotate_half(x):
        d = x.shape[-1]
        x1, x2 = x[..., :d // 2], x[..., d // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rotation(self, x, cos, sin):
        orig_dtype = x.dtype
        x = x.float()
        cos = cos[None, None, :, :].float()
        sin = sin[None, None, :, :].float()
        x = x * cos + self._rotate_half(x) * sin
        return x.to(orig_dtype)

    def _video_positions(self, n_k, device):
        n_frames = self.n_video_frames
        t_frames = torch.arange(n_frames, device=device).float() / self.video_sr

        if n_k == n_frames:
            return t_frames
        elif n_k > n_frames:
            n_patches_per_frame = n_k // n_frames
            return t_frames.repeat_interleave(n_patches_per_frame)
        else:
            return t_frames[:n_k]

    def forward(self, q, k):
        n_q, n_k = q.shape[2], k.shape[2]
        device = q.device

        t_audio = torch.arange(n_q, device=device).float() * self.patch_size / self.sample_rate
        audio_cos, audio_sin = self._angles_to_tables(t_audio)
        q = self._apply_rotation(q, audio_cos, audio_sin)

        t_video = self._video_positions(n_k, device)
        video_cos, video_sin = self._angles_to_tables(t_video)
        k = self._apply_rotation(k, video_cos, video_sin)

        return q, k
    
def get_rope_cls(name):
    return {"image": ImageRoPE, "audio": AudioRoPE, "vid2audio": VideoAudioRoPE}[name]