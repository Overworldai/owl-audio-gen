import os
from dataclasses import dataclass

import yaml
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("env", lambda k: os.environ.get(k))

@dataclass
class TransformerConfig():
    n_layers : int = 12
    n_heads : int = 12
    d_model : int = 384

    patch_size : int = 1
    causal: bool = True

@dataclass
class TrainingConfig:
    trainer_id : str = None
    data_id : str = None
    filepath : str = None  # For audio data path

    target_batch_size : int = 128
    batch_size : int = 2

    epochs : int = 200

    opt : str = "AdamW"
    opt_kwargs : dict = None
    d_opt_kwargs : dict = None # Only for GAN

    loss_weights : dict = None

    scheduler : str = None
    scheduler_kwargs : dict = None

    checkpoint_dir : str = "checkpoints/v0" # Where checkpoints saved
    resume_ckpt : str = None

    # Distillation related
    teacher_ckpt : str = None
    teacher_cfg : str = None

    sample_interval : int = 1000
    save_interval : int = 1000

    # Adversarial realted
    delay_adv: int = 20000
    warmup_adv:int = 5000

    # Causal regularization
    warmup_crt:int = 1000

    # For distillation, if you want to renormalize latents, scale by this amount before decode
    latent_scale:float = 1.0
    lpips_id: str = "convnext"

@dataclass
class WANDBConfig:
    name : str = None
    project : str = None
    run_name : str = None

@dataclass
class Config:
    model: TransformerConfig
    train: TrainingConfig
    wandb: WANDBConfig

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            raw_cfg = yaml.safe_load(f)

        cfg = OmegaConf.create(raw_cfg)
        return OmegaConf.structured(cls(**cfg))