import os
import tempfile
from fractions import Fraction

import av
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


def audio_to_wandb(t, sample_rate):
    """
    Turn [B, C, T] tensor in [-1, 1] into wandb Audio objects.
    """
    t = t.clamp(-1, 1).detach().float().cpu()
    result = []
    for audio in t:                     # audio: [C, T]
        arr = audio.numpy().T           # [T, C]
        result.append(wandb.Audio(arr, sample_rate=int(sample_rate)))
    return result


def _write_mp4(path, video_thwc, audio_ct, fps, audio_sr):
    """Write a single mp4 with audio.
    video_thwc : numpy [T, H, W, 3] uint8
    audio_ct   : numpy [2, T_audio] float32 in [-1, 1]
    """
    H, W = video_thwc.shape[1], video_thwc.shape[2]
    container = av.open(path, mode='w')

    v_stream = container.add_stream('libx264', rate=Fraction(fps).limit_denominator(1000))
    v_stream.width = W
    v_stream.height = H
    v_stream.pix_fmt = 'yuv420p'

    a_stream = container.add_stream('aac', rate=int(audio_sr))

    for frame_np in video_thwc:
        frame = av.VideoFrame.from_ndarray(frame_np, format='rgb24')
        for pkt in v_stream.encode(frame):
            container.mux(pkt)
    for pkt in v_stream.encode(None):
        container.mux(pkt)

    resampler = av.AudioResampler(format='fltp', layout='stereo', rate=int(audio_sr))
    a_frame = av.AudioFrame.from_ndarray(audio_ct, format='fltp', layout='stereo')
    a_frame.sample_rate = int(audio_sr)
    for rf in resampler.resample(a_frame):
        for pkt in a_stream.encode(rf):
            container.mux(pkt)
    for rf in resampler.resample(None):
        for pkt in a_stream.encode(rf):
            container.mux(pkt)
    for pkt in a_stream.encode(None):
        container.mux(pkt)

    container.close()


def video_audio_to_wandb(video, audio, audio_sr, fps=60):
    """
    video : [B, T, C, H, W] float in [0, 1]   (decoded video at fps)
    audio : [B, 2, T_audio]  float in [-1, 1]  (decoded audio at audio_sr)
    Audio is trimmed to match video duration so they stay in sync.
    Returns (list of wandb.Video, list of temp paths to delete after upload).
    """
    video = video.detach().float().cpu().clamp(0, 1)
    audio = audio.detach().float().cpu().clamp(-1, 1)

    T_v = video.shape[1]
    audio_samples = int(T_v / fps * audio_sr)
    audio = audio[:, :, :audio_samples]

    wandb_entries = []
    temp_paths = []
    for v, a in zip(video, audio):
        v_np = (v.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        a_np = a.numpy()  # [2, T_audio]

        fd, path = tempfile.mkstemp(suffix='.mp4')
        os.close(fd)
        _write_mp4(path, v_np, a_np, fps, audio_sr)
        wandb_entries.append(wandb.Video(path, format="mp4"))
        temp_paths.append(path)

    return wandb_entries, temp_paths


def video_audio_txt_to_wandb(video, audio, captions, audio_sr, fps=60):
    video = video.detach().float().cpu().clamp(0, 1)
    audio = audio.detach().float().cpu().clamp(-1, 1)

    T_v = video.shape[1]
    audio_samples = int(T_v / fps * audio_sr)
    audio = audio[:, :, :audio_samples]

    wandb_entries = []
    temp_paths = []
    for v, a, cap in zip(video, audio, captions):
        v_np = (v.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
        a_np = a.numpy()  # [2, T_audio]

        fd, path = tempfile.mkstemp(suffix='.mp4')
        os.close(fd)
        _write_mp4(path, v_np, a_np, fps, audio_sr)
        wandb_entries.append(wandb.Video(path, format="mp4", caption=cap))
        temp_paths.append(path)

    return wandb_entries, temp_paths

