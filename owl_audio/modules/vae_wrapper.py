import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class VAEWrapperState:
    vae_id: str
    sample_rate: int
    latent_sr: float
    window_length: float
    latent_t: int
    vae: Optional[object] = None
    vocoder: Optional[object] = None
    audio_processor: Optional[object] = None
    vocoder_sr: Optional[int] = None
    latent_ch: Optional[int] = None
    device: Optional[str] = None

def load_audio_vae(vae_id, sample_rate, latent_sr, window_length, device):
    latent_t = int(window_length * latent_sr)
    wrapper_state = VAEWrapperState(
        vae_id=vae_id,
        sample_rate=sample_rate,
        latent_sr=latent_sr,
        window_length=window_length,
        latent_t=latent_t,
        device=device
    )

    if vae_id == 'stable_audio':
        from diffusers import StableAudioPipeline
        pipe = StableAudioPipeline.from_pretrained(
            "stabilityai/stable-audio-open-1.0", 
            torch_dtype=torch.bfloat16
        )
        wrapper_state.vae = pipe.vae.to(wrapper_state.device).eval()
        wrapper_state.vae.encode = torch.compile(wrapper_state.vae.encode)
        wrapper_state.vae.decode = torch.compile(wrapper_state.vae.decode)
        del pipe
    elif vae_id == 'stable_audio_3':
        from stable_audio_3 import AutoencoderModel
        wrapper_state.vae = AutoencoderModel.from_pretrained("same-l")
        wrapper_state.vae.autoencoder.to(wrapper_state.device).bfloat16().eval()
    elif vae_id == 'mmaudio':
        from ..modules.mmaudio.features_utils import FeaturesUtils
        # the model wights must be already loaded beforehand
        wrapper_state.vae = FeaturesUtils(
            tod_vae_ckpt='/workspace/model_weights/MMAUDIO/v1-44.pth',
            bigvgan_vocoder_ckpt='/workspace/model_weights/MMAUDIO/best_netG.pt',
            mode='44k'
        ).to(wrapper_state.device).bfloat16().eval()
    elif vae_id == 'ace-step1.5':
        from diffusers import AutoencoderOobleck
        wrapper_state.vae = AutoencoderOobleck.from_pretrained(
            "ACE-Step/Ace-Step1.5", 
            subfolder="vae",
            torch_dtype=torch.bfloat16
        ).to(wrapper_state.device).eval()
    else:
        raise ValueError(f"Unknown Audio VAE id: {vae_id}")

    return wrapper_state

@torch.no_grad()
def encode_audio(wrapper_state: VAEWrapperState, raw_audio):
    vae_id = wrapper_state.vae_id
    latent_t = wrapper_state.latent_t
    sample_rate = wrapper_state.sample_rate
    vae = wrapper_state.vae

    if vae_id == 'stable_audio':
        # [B, 2, T_raw] bf16 → [B, C, latent_t] bf16
        latents = vae.encode(raw_audio).latent_dist.sample()
        return latents[..., :latent_t]
    elif vae_id == 'stable_audio_3':
        latents = vae.encode(raw_audio, sample_rate)
        return latents[..., :latent_t]
    elif vae_id == 'mmaudio':
        if raw_audio.ndim == 3: # MMAUDIO works only with mono audio
            raw_audio = raw_audio.mean(dim=1)
        latents = vae.wrapped_encode(raw_audio.float())
        return latents[..., :latent_t]
    elif vae_id == 'ace-step1.5':
        latents = vae.encode(raw_audio).latent_dist.sample()
        return latents[..., :latent_t]
    else:
        raise ValueError(f"Unknown Audio VAE id: {vae_id}")

        
@torch.no_grad()
def decode_audio(wrapper_state: VAEWrapperState, latents):
    vae_id = wrapper_state.vae_id
    vae = wrapper_state.vae

    if vae_id == 'stable_audio':
        # [B, C, latent_t] bf16 → [B, 2, T_raw] bf16
        return vae.decode(latents).sample
    elif vae_id == 'stable_audio_3':
        return vae.decode(latents)
    elif vae_id == 'mmaudio':
        return vae.wrapped_decode(latents)
    elif vae_id == 'ace-step1.5':
        return vae.decode(latents).sample
    else:
        raise ValueError(f"Unknown Audio VAE id: {vae_id}")