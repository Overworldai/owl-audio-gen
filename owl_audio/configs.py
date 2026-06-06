import os
from dataclasses import dataclass

import yaml
from omegaconf import OmegaConf

from typing import Tuple, Optional

OmegaConf.register_new_resolver("env", lambda k: os.environ.get(k))

@dataclass
class ModelConfig:
    n_layers: int
    n_heads: int
    d_model: int
    d_text: int
    video_patch_content: int

    sample_rate: float
    
    channels: int
    patch_size: Tuple[int, int]
    sample_size: Tuple[int, int]
    
    x0_mode: bool
    kernel_size: Optional[Tuple[int, int]] = None
    mlp_ratio: int = 4

@dataclass
class TrainingConfig:
    trainer_id : str = None
    data_id : str = None
    data_kwargs : dict = None

    target_batch_size : int = 256
    batch_size : int = 32

    epochs : int = 200

    opt : str = "AdamW"
    opt_kwargs : dict = None

    scheduler : str = None
    scheduler_kwargs : dict = None

    checkpoint_dir : str = "checkpoints/v0" # Where checkpoints saved
    resume_ckpt : str = None

    # Distillation related
    teacher_ckpt : str = None
    teacher_cfg : str = None

    sample_interval : int = 1000
    save_interval : int = 1000
    
    sampler_id: str = "euler"
    sampling_steps: int = 20
    cfg_scale: float = 1.5
    vae_id: Optional[str] = None
    video_size: Optional[Tuple[int, int]] = None

@dataclass
class WANDBConfig:
    name : str = None
    project : str = None
    run_name : str = None

@dataclass
class Config:
    model: ModelConfig
    train: TrainingConfig
    wandb: WANDBConfig

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            raw_cfg = yaml.safe_load(f)

        cfg = OmegaConf.create(raw_cfg)
        return OmegaConf.structured(cls(**cfg))