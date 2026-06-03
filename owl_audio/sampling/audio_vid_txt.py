import torch
from tqdm import tqdm

from .schedulers import get_sd3_euler


@torch.no_grad()
def audio_video_sample(model, shape, video, text, attn_mask, steps, device, dtype,
                       cfg_scale=1.5, progress_bar=True):
    """
    CFG flow-matching sampler for video-conditioned audio.
    model : AudioDiffusionCore
    shape : (B, C, T_audio)
    video : [B, T_v, C_v, H_v, W_v] video latents
    Returns [B, C, T_audio] audio in [-1, 1].
    """
    x = torch.randn(shape, device=device, dtype=dtype) * model.config.noise_scale
    ts = torch.ones(shape[0], device=device, dtype=dtype)

    # Project video once — reused every step
    video_tokens = model.project_video(video.to(device=device, dtype=dtype))  # [B, T_v, d_model]
    null_video = model.null_video[None, None, :].expand_as(video_tokens)

    text_tokens = model.text_proj(text) # [B, L, d_model]
    null_text = model.null_text[None, None, :].expand_as(text_tokens)

    use_cfg = cfg_scale != 1.0

    for dt in tqdm(get_sd3_euler(steps).to(device=device, dtype=dtype), disable=not progress_bar):
        cond_pred = model(x, ts, video_tokens=video_tokens, text_tokens=text_tokens, attn_mask=attn_mask)
        if use_cfg:
            uncond_pred = model(x, ts, video_tokens=null_video, text_tokens=null_text, attn_mask=attn_mask)
            pred = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
        else:
            pred = cond_pred
        x = x - dt * pred
        ts = ts - dt

    return x.clamp(-1, 1)