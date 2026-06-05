"""
Video-conditioned audio latent diffusion trainer.
Audio is encoded on-the-fly with a compiled Stable Audio VAE.
Video latents are pre-encoded on disk and loaded via mmap.
"""

import os
import torch
import wandb
from ema_pytorch import EMA
from torch.nn.parallel import DistributedDataParallel as DDP
from itertools import cycle

from .base import BaseTrainer
from ..models import get_model_cls
from ..modules.vae_wrapper import VAEWrapper
from ..sampling.audio_video import audio_video_sample
from ..data import get_loader
from ..muon import init_muon
from ..utils import Timer
from ..utils.logging import LogHelper, audio_to_wandb, video_audio_to_wandb


def _video_self_pad(fn, p=1, q=4):
    """Wraps a causal video decoder: prepend p copies of first frame, drop first q output frames."""
    def wrapper(x):
        b, t, c, h, w = x.shape
        x = torch.cat([x[:, :1].repeat(1, p, 1, 1, 1), x], dim=1)
        return fn(x)[:, q:]
    return wrapper


class AudioVideoTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        model_id = getattr(self.model_cfg, "model_id", "vid2audio")
        self.model = get_model_cls(model_id)(self.model_cfg)

        if self.rank == 0:
            param_count = sum(p.numel() for p in self.model.parameters())
            print(f"Total parameters: {param_count:,}")

        self.ema = None
        self.opt = None
        self.scheduler = None
        self.scaler = None
        self.total_step_counter = 0

        # Latent length derived entirely from model config
        self.latent_t = int(self.model_cfg.window_length * self.model_cfg.sample_rate)

        vae_id = getattr(self.train_cfg, "vae_id", None)
        if self.rank == 0:
            print(f"Loading {vae_id} Audio VAE...")
        self.vae = VAEWrapper(
            vae_id=vae_id,
            sample_rate=getattr(self.train_cfg.data_kwargs, "sample_rate", None),
            latent_sr=getattr(self.model_cfg, "sample_rate", None),
            window_length=getattr(self.model_cfg, "window_length", None)
        )
        self.raw_audio_sr = self.vae.sample_rate
       
        self.video_vae_id = getattr(self.train_cfg, "video_vae_id", None)
        self.video_vae_ckpt = getattr(self.train_cfg, "video_vae_ckpt", None)
        self.video_decode_fn = None

    def save(self):
        save_dict = {
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "opt": self.opt.state_dict(),
            "scaler": self.scaler.state_dict(),
            "total_step_counter": self.total_step_counter,
        }
        if self.scheduler is not None:
            save_dict["scheduler"] = self.scheduler.state_dict()
        super().save(save_dict)

    def load(self):
        if self.train_cfg.resume_ckpt is None:
            return
        save_dict = super().load(self.train_cfg.resume_ckpt)
        self.model.load_state_dict(save_dict["model"])
        self.ema.load_state_dict(save_dict["ema"])
        self.opt.load_state_dict(save_dict["opt"])
        self.scaler.load_state_dict(save_dict["scaler"])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(save_dict["scheduler"])
        self.total_step_counter = save_dict["total_step_counter"]

    def train(self):
        torch.cuda.set_device(self.local_rank)
        self.vae = self.vae.cuda().bfloat16().eval()

        if self.video_vae_id == "taehv":
            from taehv.taehv import TAEHV
            _taehv = TAEHV(self.video_vae_ckpt).cuda().bfloat16().eval()
            self.video_encode_fn = lambda x: _taehv.encode_video(x)
            self.video_decode_fn = lambda x: _video_self_pad(_taehv.decode_video)(x).clamp(0, 1)

        self.model = self.model.cuda().train()
        if self.world_size > 1:
            self.model = DDP(self.model, find_unused_parameters=True)

        self.ema = EMA(self.model, beta=0.999, update_after_step=0, update_every=1)

        if self.train_cfg.opt.lower() == "muon":
            self.opt = init_muon(
                self.model, rank=self.rank, world_size=self.world_size,
                **self.train_cfg.opt_kwargs
            )
        else:
            self.opt = getattr(torch.optim, self.train_cfg.opt)(
                self.model.parameters(), **self.train_cfg.opt_kwargs
            )

        accum_steps = max(
            1, self.train_cfg.target_batch_size // self.train_cfg.batch_size // self.world_size
        )
        self.scaler = torch.amp.GradScaler()
        ctx = torch.amp.autocast(f"cuda:{self.local_rank}", torch.bfloat16)

        self.load()

        timer = Timer()
        timer.reset()
        metrics = LogHelper()

        if self.rank == 0:
            wandb.watch(self.get_module(), log="all")

        n_samples = getattr(self.train_cfg, "n_samples", 2)
        cfg_scale = getattr(self.train_cfg, "cfg_scale", 1.5)
        pending_video_paths = []  # temp mp4s from previous log step, safe to delete now

        train_loader = get_loader(
            self.train_cfg.data_id,
            self.train_cfg.batch_size,
            split='train',
            **self.train_cfg.data_kwargs,
        )
        
        # create the eval data loader
        eval_loader = get_loader(
            self.train_cfg.data_id,
            self.train_cfg.n_samples, 
            split='eval',
            **self.train_cfg.data_kwargs,
        )
        eval_iter = cycle(eval_loader)
        
        local_step = 0
        for _ in range(self.train_cfg.epochs):
            for raw_audio, video_latents in train_loader:
                raw_audio = raw_audio.to(self.device).bfloat16()
                video_latents = video_latents.to(self.device).bfloat16()

                audio_latents = self.vae.encode_audio(raw_audio)   # [B, C, latent_t]

                with ctx:
                    loss = self.model(audio_latents, video=video_latents) / accum_steps
                metrics.log("loss", loss)

                self.scaler.scale(loss).backward()

                local_step += 1
                if local_step % accum_steps == 0:
                    self.scaler.unscale_(self.opt)
                    if self.train_cfg.opt.lower() != "muon":
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    self.opt.zero_grad(set_to_none=True)

                    if getattr(self.train_cfg, "scheduler", None) is not None:
                        self.scheduler.step()
                    self.ema.update()

                    with torch.no_grad():
                        wandb_dict = metrics.pop()
                        wandb_dict["time"] = timer.hit()
                        timer.reset()

                        if self.total_step_counter % self.train_cfg.sample_interval == 0:
                            for p in pending_video_paths:
                                try:
                                    os.remove(p)
                                except OSError:
                                    pass
                            pending_video_paths = []
                            
                            _, cond_video = next(eval_iter)
                            cond_video = cond_video.bfloat16().cuda()
                            with ctx:
                                audio_samples = audio_video_sample(
                                    self.get_module(ema=True).core,
                                    shape=(n_samples, self.model_cfg.channels, self.latent_t),
                                    video=cond_video,
                                    steps=self.train_cfg.sampling_steps,
                                    device=self.device,
                                    dtype=torch.bfloat16,
                                    cfg_scale=cfg_scale,
                                )
                            decoded_audio = self.vae.decode_audio(audio_samples)  # [B, 2, T_raw]

                            if self.video_decode_fn is not None:
                                decoded_video = self.video_decode_fn(cond_video)  # [B, T, C, H, W] in [0,1]
                                entries, paths = video_audio_to_wandb(
                                    decoded_video, decoded_audio,
                                    self.raw_audio_sr,
                                    self.train_cfg.video_fps
                                )
                                wandb_dict["samples"] = entries
                                pending_video_paths = paths  # delete next sampling step
                            else:
                                wandb_dict["samples"] = audio_to_wandb(decoded_audio, self.raw_audio_sr)

                        if self.rank == 0:
                            wandb.log(wandb_dict)

                        self.total_step_counter += 1

                    if self.total_step_counter % self.train_cfg.save_interval == 0:
                        if self.rank == 0:
                            self.save()

                    self.barrier()