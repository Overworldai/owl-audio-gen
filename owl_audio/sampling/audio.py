import torch
from tqdm import tqdm

from .schedulers import get_sd3_euler


@torch.no_grad()
def audio_sample(model, shape, steps, device, dtype, progress_bar=True):
    """
    Unconditional flow matching sampler for audio.
    model: AudioDiffusionCore
    shape: (B, C, T)
    Returns [B, C, T] audio in [-1, 1].
    """
    x = torch.randn(shape, device=device, dtype=dtype) * model.config.noise_scale
    ts = torch.ones(shape[0], device=device, dtype=dtype)

    for dt in tqdm(get_sd3_euler(steps).to(device=device, dtype=dtype), disable=not progress_bar):
        pred = model(x, ts)
        x = x - dt * pred
        ts = ts - dt

    return x.clamp(-1, 1)
