import torch
import torch.nn as nn
import torchaudio.functional as F
from einops import rearrange

from diffusers import AutoencoderKLLTX2Audio, StableAudioPipeline
from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
# from ltx_core.model.audio_vae.ops import AudioProcessor
# from ltx_core.types import Audio
# from stable_audio_3 import AutoencoderModel


class VAEWrapper(nn.Module):
    """Wrapper for Audio VAEs like Stable Audio and LTX2"""
    def __init__(self, vae_id, sample_rate, latent_sr, window_length):
        super().__init__()
        self.vae_id = vae_id
        self.sample_rate = sample_rate
        self.latent_sr = latent_sr
        self.window_length = window_length
        self.latent_t = int(self.window_length * self.latent_sr)

        if self.vae_id == 'stable_audio':
            pipe = StableAudioPipeline.from_pretrained(
                "stabilityai/stable-audio-open-1.0", 
                torch_dtype=torch.bfloat16
            )
            self.vae = pipe.vae 
            self.vae.encode = torch.compile(self.vae.encode)
            self.vae.decode = torch.compile(self.vae.decode)
            del pipe
        elif self.vae_id == 'stable_audio_3':
            self.vae = AutoencoderModel.from_pretrained("same-l")
            self.vae.autoencoder.bfloat16()
        elif self.vae_id == 'ltx2':
            self.vae = AutoencoderKLLTX2Audio.from_pretrained(
                "Lightricks/LTX-2", 
                subfolder="audio_vae", 
                torch_dtype=torch.bfloat16
            )
            self.vocoder = LTX2Vocoder.from_pretrained(
                "Lightricks/LTX-2", 
                subfolder="vocoder", 
                torch_dtype=torch.bfloat16,
            )
            self.audio_processor = AudioProcessor(
                target_sample_rate=self.vae.config.sample_rate,
                mel_bins=self.vae.config.mel_bins,
                mel_hop_length=self.vae.config.mel_hop_length,
                n_fft=1024
            )
            self.vae.encode = torch.compile(self.vae.encode)
            self.vae.decode = torch.compile(self.vae.decode)
            self.vocoder.forward = torch.compile(self.vocoder.forward)

            self.vocoder_sr = self.vocoder.config.output_sampling_rate
            self.latent_ch = self.vae.config.latent_channels
        else:
            raise ValueError(f"Unknown Audio VAE id: {self.vae_id}")

    @torch.no_grad()
    def encode_audio(self, raw_audio):
        if self.vae_id == 'stable_audio':
            # [B, 2, T_raw] bf16 → [B, C, latent_t] bf16
            latents = self.vae.encode(raw_audio).latent_dist.sample()
            return latents[..., :self.latent_t]
        elif self.vae_id == 'stable_audio_3':
            latents = self.vae.encode(raw_audio, self.sample_rate)
            return latents[..., :self.latent_t]
        elif self.vae_id == 'ltx2':
            self.audio_processor.float() # cuFFT supports only float32
            audio = Audio(raw_audio.float(), self.sample_rate)
            mel = self.audio_processor.waveform_to_mel(audio).bfloat16()    # [B, 2, T_mel, n_mels] bf16
            latents = self.vae.encode(mel).latent_dist.sample()             # [B, 8, T_mel/4, 16] bf16
            latents = rearrange(latents, 'b c t f -> b (c f) t')            # [B, 128, T_mel/4] bf16
            return latents[..., :self.latent_t]
        else:
            raise ValueError(f"Unknown Audio VAE id: {self.vae_id}")
        
    @torch.no_grad()
    def decode_audio(self, latents):
        if self.vae_id == 'stable_audio':
            # [B, C, latent_t] bf16 → [B, 2, T_raw] bf16
           return self.vae.decode(latents).sample
        elif self.vae_id == 'stable_audio_3':
            return self.vae.decode(latents)
        elif self.vae_id == 'ltx2':
            latents = rearrange(latents, 'b (c f) t -> b c t f', c=self.latent_ch)      
            mel_hat = self.vae.decode(latents).sample   # [B, 2, T_mel, n_mels] bf16
            wav_rec = self.vocoder(mel_hat)             # [B, 2, T_wav] bf16 @24khz
            wav_rec = F.resample(wav_rec, self.vocoder_sr, self.sample_rate)
            return wav_rec                              # [B, 2, T_raw] bf16
        else:
            raise ValueError(f"Unknown Audio VAE id: {self.vae_id}")