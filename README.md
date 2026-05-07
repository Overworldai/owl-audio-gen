# Owl Audio Gen

Latent diffusion for unconditional audio generation. The model is a DiT operating
on audio latents from any pre-trained audio VAE.

## Setup

```bash
git clone https://github.com/Overworldai/owl-audio-gen.git
cd owl-audio-gen
pip install -e .
```

A `.env` in the repo root is read at startup — drop `WANDB_API_KEY=...` there if
you don't want to set it in your shell.

## Audio VAE

The training loop encodes raw audio on-the-fly with a frozen VAE. The baseline
uses the VAE from
[Stable Audio Open 1.0](https://huggingface.co/stabilityai/stable-audio-open-1.0)
That model is gated, so to use it
as-is you'll need to accept the license on HuggingFace and run
`huggingface-cli login`. Support for more VAEs/custom VAEs is a WIP.

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
