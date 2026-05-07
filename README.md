# Owl Audio Gen

Latent diffusion for unconditional audio generation. The model is a DiT operating
on latents from Stability AI's [Stable Audio Open 1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0)
VAE — the VAE is loaded on-the-fly via `diffusers`, so no extra checkpoints or
submodules are needed.

## Setup

```bash
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate
pip install ema-pytorch wandb omegaconf pyyaml einops av tqdm python-dotenv
```

You'll also want a HuggingFace token with access to `stabilityai/stable-audio-open-1.0`
(the model gates downloads behind a license click-through):

```bash
huggingface-cli login
```

A `.env` in the repo root is read at startup — drop `WANDB_API_KEY=...` there if
you don't want to set it in your shell.

## Data

The default loader (`audio_dir_loader`) walks a directory tree for `.mp4` files
and decodes random audio windows on-the-fly with PyAV. Point `train.data_kwargs.source`
at any folder of mp4s (it accepts a directory, a list of directories, or a glob).

## Training

Edit `configs/audio_baseline.yml` — at minimum, set:

- `train.data_kwargs.source` → your audio dataset path
- `wandb.name` → your wandb entity

Then launch:

```bash
# single GPU
python train.py --config_path configs/audio_baseline.yml

# multi-GPU (single node, 8 GPUs)
torchrun --nproc_per_node=8 train.py --config_path configs/audio_baseline.yml
```

Checkpoints land in `train.checkpoint_dir`, and samples are logged to wandb every
`train.sample_interval` steps.
