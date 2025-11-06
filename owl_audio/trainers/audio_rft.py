from .base import BaseTrainer
from ..data import get_loader
from ..models import get_model_cls
from ..utils import freeze, unfreeze, Timer
from ..utils.logging import LogHelper, log_audio_to_wandb
from ..sampling import flow_sample

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from ema_pytorch import EMA
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import sys
sys.path.append("./owl-vaes")
from owl_vaes import from_pretrained

def load_autoencoder():
    cfg_path = "owl-vaes/configs/waypoint_1_audio/basic.yml"
    ckpt_path = "/mnt/data/shahbuland/owl-vaes/checkpoints/waypoint_1_audio_basic/step_105000.pt"

    autoencoder = from_pretrained(cfg_path, ckpt_path).bfloat16()
    freeze(autoencoder)
    return autoencoder

class AudioRFTTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        model_id = self.model_cfg.model_id
        self.model = get_model_cls(model_id)(self.model_cfg).train().to(self.device)
        self.vae = load_autoencoder().to(self.device)
        
        if self.rank == 0:
            param_count = sum(p.numel() for p in self.model.parameters())
            print(f"Total parameters: {param_count:,}")

        self.ema = None
        self.opt = None
        self.scheduler = None
        self.scaler = None

        self.total_step_counter = 0

    def save(self):
        # Save the original model state, not the compiled one
        save_dict = {
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "opt": self.opt.state_dict(),
            "scaler": self.scaler.state_dict(),
            "steps": self.total_step_counter,
        }
        super().save(save_dict)

    def load(self):
        if not hasattr(self.train_cfg, 'resume_ckpt') or self.train_cfg.resume_ckpt is None:
            return

        save_dict = super().load(self.train_cfg.resume_ckpt)
        self.model.load_state_dict(save_dict["model"])
        self.ema.load_state_dict(save_dict["ema"])
        self.opt.load_state_dict(save_dict["opt"])
        self.scaler.load_state_dict(save_dict["scaler"])
        self.total_step_counter = save_dict["steps"]
    
    def get_ema_core(self):
        return self.ema.ema_model.module.core if self.world_size > 1 else self.ema.ema_model.core

    def train(self):
        # Model setup
        self.model = self.model.to(self.device).train()

        if self.world_size > 1:
            self.model = DDP(self.model)

        # EMA, compile, optimizer
        self.ema = EMA(self.model, beta=0.9999, update_after_step=0, update_every=1)
        self.vae.encoder = torch.compile(self.vae.encoder)
        self.vae.decoder = torch.compile(self.vae.decoder)

        self.opt = getattr(torch.optim, self.train_cfg.opt)(
            self.model.parameters(), **self.train_cfg.opt_kwargs
        )

        # Data setup
        self.data_loader = get_loader(self.train_cfg.data_id, self.train_cfg.batch_size, **self.train_cfg.data_kwargs)
        self.total_step_counter = 0
        accum_steps = (
            self.train_cfg.target_batch_size
            // self.train_cfg.batch_size
            // self.world_size
        )
        accum_steps = max(1, accum_steps)
        self.scaler = torch.GradScaler()
        ctx = torch.autocast(self.device, torch.bfloat16)

        timer = Timer()
        timer.reset()
        metrics = LogHelper()
        if self.rank == 0:
            wandb.watch(self.get_module(), log="all")
    
        def vae_sample(wf):
            wf = wf.to(self.device)
            mu, logvar = self.vae.encoder(wf)
            z = torch.randn_like(mu) * torch.exp(0.5 * logvar) + mu
            return z / self.train_cfg.ldm_scale

        for epoch_idx in range(self.train_cfg.epochs):
            for batch in self.data_loader:
                batch = batch.to(device = self.device, dtype = torch.bfloat16)
                batch = vae_sample(batch) # latents

                with ctx:
                    loss = self.model(batch)
                
                metrics.log_dict({
                    "loss": loss,
                })

                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.opt.zero_grad(set_to_none=True)
                self.scaler.update()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.ema.update()

                with torch.no_grad():
                    wandb_dict = metrics.pop()
                    wandb_dict["time"] = timer.hit()
                    timer.reset()

                    if self.total_step_counter % self.train_cfg.sample_interval == 0:
                        sample = flow_sample(
                            self.get_ema_core(), 
                            batch, batch,
                            20, decoder = self.vae.decoder,
                            scaling_factor = self.train_cfg.ldm_scale
                        )
                        wandb_dict["samples"] = log_audio_to_wandb(
                            sample.detach().contiguous().bfloat16()
                        )

                    if self.rank == 0: wandb.log(wandb_dict)

                self.total_step_counter += 1
                if self.total_step_counter % self.train_cfg.save_interval == 0:
                    self.save()
                self.barrier()