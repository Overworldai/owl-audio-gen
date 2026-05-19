import torch
import torch.nn as nn
import torchaudio.functional as F

from diffusers import AutoencoderKLLTX2Audio, StableAudioPipeline
from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder
from ltx_core.model.audio_vae.ops import AudioProcessor
from ltx_core.types import Audio

from owl_audio.configs import Config


class VAEWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vae_id = config.vae_id
        self.sample_rate = config.data_kwargs.sample_rate
        self.window_length = config.data_kwargs.window_length
        self.n_samples = int(self.window_length * self.sample_rate)

        if self.vae_id == 'stable_audio':
            pipe = StableAudioPipeline.from_pretrained(
                "stabilityai/stable-audio-open-1.0", 
                torch_dtype=torch.bfloat16
            )
            self.vae = pipe.vae 
            self.vae.encode = torch.compile(self.vae.encode)
            self.vae.decode = torch.compile(self.vae.decode)
            del pipe

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
        else:
            raise ValueError(f"Unknown Audio VAE id: {self.vae_id}")

    @torch.no_grad()
    def encode_audio(self, raw_audio):
        if self.vae_id == 'stable_audio':
            # [B, 2, T_raw] bf16 → [B, C, T] bf16
            latents = self.vae.encode(raw_audio).latent_dist.sample()
            return latents
        elif self.vae_id == 'ltx2':
            audio = Audio(waveform=raw_audio, sampling_rate=self.sample_rate)
            mel = self.audio_processor.waveform_to_mel(audio).bfloat16()    # [B, 2, T_mel, n_mels] bf16
            latents = self.vae.encode(mel).latent_dist.sample()             # [B, 8, T_mel/4, 16] bf16
            return latents
        else:
            raise ValueError(f"Unknown Audio VAE id: {self.vae_id}")
        
    @torch.no_grad()
    def decode_audio(self, latents):
        if self.vae_id == 'stable_audio':
            # [B, C, T] bf16 → [B, 2, T_raw] bf16
            wav_rec = self.vae.decode(latents).sample
            return wav_rec[..., :self.n_samples]
        elif self.vae_id == 'ltx2':
            mel_hat = self.vae.decode(latents).sample   # [B, 2, T_mel, n_mels] bf16
            wav_rec = self.vocoder(mel_hat)             # [B, 2, T_wav] bf16 @24khz
            wav_rec = F.resample(wav_rec, self.vocoder_sr, self.sample_rate)
            return wav_rec[..., :self.n_samples]              # [B, 2, T_raw] bf16
        else:
            raise ValueError(f"Unknown Audio VAE id: {self.vae_id}")


if __name__ == '__main__':
    cfg = Config.from_yaml('/workspace/owl-audio-gen/configs/vid2audio_baseline.yml')
    
    x = torch.randn((1, 2, 441000)).bfloat16().to('cuda')
    vae = VAEWrapper(cfg.train).to('cuda')
    lat = vae.encode_audio(x)
    y_hat = vae.decode_audio(lat)
    print(y_hat.shape)