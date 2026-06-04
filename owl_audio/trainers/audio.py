"""
Audio latent diffusion trainer.
Encodes raw audio on-the-fly with a compiled Stable Audio VAE encoder.
"""

import torch
import wandb
from ema_pytorch import EMA
from torch.nn.parallel import DistributedDataParallel as DDP
from itertools import cycle

from ..models.vae_wrapper import load_audio_vae, encode_audio, decode_audio
from transformers import T5EncoderModel, T5Tokenizer

from .base import BaseTrainer
from ..models.audio import AudioDiffusionModel
from ..sampling.audio import audio_sample
from ..data import get_loader
from ..muon import init_muon
from ..utils import Timer
from ..utils.logging import LogHelper, audio_to_wandb


class AudioTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = AudioDiffusionModel(self.model_cfg)

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

        # Load Audio VAE
        self.vae_id = getattr(self.train_cfg, "vae_id", "stable_audio")
        self.raw_audio_sr = getattr(self.train_cfg.data_kwargs, "sample_rate", 44100)

        if self.rank == 0:
            print(f"Loading {self.vae_id} Audio VAE...")

        self.vae = load_audio_vae(
            vae_id=self.vae_id, 
            sample_rate=self.raw_audio_sr, 
            latent_sr=self.model_cfg.sample_rate, 
            window_length=self.model_cfg.window_length
        )

        # Optionally: load text encoder
        if self.model_cfg.d_text > 0:
            text_encoder = getattr(self.model_cfg, "text_encoder", "t5-base")
            # In distributed training, each rank keeps its own inference-only copy.
            # Warm cache on rank 0 first to avoid concurrent first-time downloads.
            if self.world_size > 1 and self.rank == 0:
                _ = T5Tokenizer.from_pretrained(text_encoder)
                _ = T5EncoderModel.from_pretrained(text_encoder)
            if self.world_size > 1:
                self.barrier()

            self.tokenizer = T5Tokenizer.from_pretrained(text_encoder)
            self.text_encoder = T5EncoderModel.from_pretrained(text_encoder).to(self.device).bfloat16().eval()

    @torch.no_grad()
    def encode(self, raw_audio):
        latents = encode_audio(self.vae, raw_audio)
        return latents[..., :self.latent_t]

    @torch.no_grad()
    def decode(self, latents):
        return decode_audio(self.vae, latents)

    @torch.no_grad()
    def encode_text(self, text):
        # text: List[str], length B
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)

        hidden = self.text_encoder(**tokens).last_hidden_state  # [B, seq, d_text]
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        return hidden, mask

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

        self.model = self.model.cuda().train()
        if self.world_size > 1:
            self.model = DDP(self.model)

        self.ema = EMA(
            self.model,
            beta=0.999,
            update_after_step=0,
            update_every=1,
        )

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

        train_loader = get_loader(
            self.train_cfg.data_id,
            self.train_cfg.batch_size,
            split='train',
            **self.train_cfg.data_kwargs,
        )
        n_samples = getattr(self.train_cfg, "n_samples", 2)

        eval_loader = get_loader(
            self.train_cfg.data_id,
            batch_size=n_samples,
            split='eval',
            **self.train_cfg.data_kwargs,
        )
        eval_iter = cycle(eval_loader)

        local_step = 0
        for _ in range(self.train_cfg.epochs):
            for raw_audio, caption in train_loader:
                # raw_audio: [B, 2, T_raw] — encode on-the-fly
                raw_audio = raw_audio.to(self.device).bfloat16()
                latents = self.encode(raw_audio)  # [B, C, latent_t]
                text_emb, text_mask = self.encode_text(caption)

                with ctx:
                    loss = self.model(latents, text_emb, text_mask) / accum_steps
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

                            cond_audio, cond_caption = next(eval_iter)
                            cond_audio = cond_audio.to(self.device).bfloat16()
                            cond_text_emb, cond_text_mask = self.encode_text(cond_caption)
                            with ctx:
                                latent_samples = audio_sample(
                                    self.get_module(ema=True).core,
                                    text_emb=cond_text_emb,
                                    text_mask=cond_text_mask,
                                    cfg_scale=self.train_cfg.cfg_scale,
                                    shape=(n_samples, self.model_cfg.channels, self.latent_t),
                                    steps=self.train_cfg.sampling_steps,
                                    device=self.device,
                                    dtype=torch.bfloat16,
                                )
                            decoded = self.decode(latent_samples)  # [B, 2, T_raw]
                            wandb_dict["samples"] = audio_to_wandb(decoded, self.raw_audio_sr)

                            # compute eval loss
                            cond_latents = self.encode(cond_audio)
                            with ctx:
                                eval_loss = self.model(cond_latents, cond_text_emb, cond_text_mask)
                            wandb_dict["eval_loss"] = eval_loss

                        if self.rank == 0:
                            wandb.log(wandb_dict)

                        self.total_step_counter += 1

                    if self.total_step_counter % self.train_cfg.save_interval == 0:
                        if self.rank == 0:
                            self.save()

                    self.barrier()
