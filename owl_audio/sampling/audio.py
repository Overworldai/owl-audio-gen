import torch
from tqdm import tqdm

from .schedulers import get_sd3_euler


@torch.no_grad()
def audio_sample(model, text_emb, text_mask, cfg_scale, shape, 
                 steps, device, dtype, progress_bar=True):
    """
    Unconditional flow matching sampler for audio.
    model: AudioDiffusionCore
    shape: (B, C, T)
    Returns [B, C, T] audio in [-1, 1].
    """
    x = torch.randn(shape, device=device, dtype=dtype) * model.config.noise_scale
    ts = torch.ones(shape[0], device=device, dtype=dtype)

    text_tokens = model.text_proj(text_emb.to(device=device, dtype=dtype))  # [B, seq, d_model]
    null_tokens = model.null_video[None, None, :].expand_as(text_tokens)

    use_cfg = cfg_scale != 1.0

    for dt in tqdm(get_sd3_euler(steps).to(device=device, dtype=dtype), disable=not progress_bar):
        cond_pred = model(x, ts, text_emb, text_mask)
        if use_cfg:
            uncond_pred = model(x, ts, null_tokens, text_mask)
            pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
        else:
            pred = cond_pred
        x = x - dt * pred
        ts = ts - dt

    return x.clamp(-1, 1)
