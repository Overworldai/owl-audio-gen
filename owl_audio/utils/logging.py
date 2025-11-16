import wandb
import torch
import torch.distributed as dist
import numpy as np

class LogHelper:
    """
    Helps get stats across devices/grad accum steps

    Can log stats then when pop'd will get them across
    all devices (averaged out).
    For gradient accumulation, ensure you divide by accum steps beforehand.
    """
    def __init__(self):
        if dist.is_initialized():
            self.world_size = dist.get_world_size()
        else:
            self.world_size = 1

        self.data = {}

    def log(self, key, data):
        if isinstance(data, torch.Tensor):
            data = data.detach().item()
        val = data / self.world_size
        if key in self.data:
            self.data[key].append(val)
        else:
            self.data[key] = [val]

    def log_dict(self, d):
        for (k,v) in d.items():
            self.log(k,v)

    def pop(self):
        reduced = {k : sum(v) for k,v in self.data.items()}

        if self.world_size > 1:
            gathered = [None for _ in range(self.world_size)]
            dist.all_gather_object(gathered, reduced)

            final = {}
            for d in gathered:
                for k,v in d.items():
                    if k not in final:
                        final[k] = v
                    else:
                        final[k] += v
        else:
            final = reduced

        self.data = {}
        return final

def log_audio_to_wandb(
    sample: torch.Tensor,
    sample_rate: int = 44100,
    max_samples: int = 16,
) -> dict[str, wandb.Audio]:
    """
    Log audio samples to Weights & Biases.

    Args:
        original: Original audio tensor (B, N, D) where N=samples, D=channels
        reconstructed: Reconstructed audio tensor (B, N, D)
        sample_rate: Audio sample rate
        max_samples: Maximum number of samples to log

    Returns:
        Dictionary for wandb logging
    """
    batch_size = min(sample.size(0), max_samples)
    audio_logs = {}

    for i in range(batch_size):
        # Convert to numpy and ensure correct shape for wandb
        # (B, N, D) -> (N, D)
        audio = sample[i].detach().cpu().float().numpy()  # (N, D)

        # For stereo audio, mix down to mono for logging
        if audio.shape[-1] == 2:
            # Average across channels: (N, 2) -> (N,)
            audio = np.mean(audio, axis=-1)
        else:
            # Single channel: (N, 1) -> (N,)
            audio = audio.squeeze(-1)

        # Ensure audio is in correct range [-1, 1]
        audio = np.clip(audio, -1.0, 1.0)

        audio_logs[f"audio_original_{i}"] = wandb.Audio(
            audio, sample_rate=sample_rate
        )

    return audio_logs