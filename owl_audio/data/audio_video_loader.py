import os, glob, random
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import av


class RandomAudioVideoFromMP4s:
    """
    Continuous iterator yielding (audio, video_latent) pairs.
    - audio       : [C, T_audio] float32, from raw mp4
    - video_latent: [T_lat, C, H, W] float32, mmap-loaded from pre-encoded .pt

    source   : directory of mp4s  ({source}/foo/bar.mp4)
    encoded  : directory of latents ({encoded}/foo/taehv1_5/000000_latent.pt)
    Matching is by immediate parent folder name.

    Temporal alignment:
      t_start / video_duration * n_latent_frames  → latent_start_frame
    so the latent window lines up with the audio window regardless of original fps.
    """

    def __init__(self, source, encoded, seed=None,
                 window_length=10.0, sample_rate=44100,
                 video_window_frames=75, expected_hw=(16, 32)):
        self.window_length = window_length
        self.sample_rate = sample_rate
        self.window_length_samples = int(window_length * sample_rate)
        self.expected_hw = tuple(expected_hw)
        self.video_window_frames = video_window_frames
        self.encoded = Path(encoded)

        self.pairs = self._find_pairs(source, encoded)
        if not self.pairs:
            raise RuntimeError("No paired (mp4, latent) files found.")
        self.rng = random.Random(seed)
        self.meta = {}  # mp4 Path -> duration_s

    @staticmethod
    def _find_mp4s(source):
        s = os.path.expanduser(str(source))
        p = Path(s)
        registry = p / "valid_mp4s.txt"
        if p.is_dir() and registry.exists():
            print(f"[DataLoader] Using registry {registry}")
            paths = [line.strip() for line in registry.read_text().splitlines() if line.strip()]
        elif p.is_dir():
            paths = glob.glob(str(p / "**/*.mp4"), recursive=True)
        else:
            paths = glob.glob(s, recursive=True)
        filtered = [x for x in paths if x.lower().endswith(".mp4") and '_' not in Path(x).stem]
        return [Path(x) for x in sorted(set(filtered))]

    @staticmethod
    def _find_pairs(source, encoded):
        encoded = Path(encoded)
        pairs = []
        for mp4 in RandomAudioVideoFromMP4s._find_mp4s(source):
            folder = mp4.parent.name
            latent_path = encoded / folder / "taehv1_5" / "000000_latent.pt"
            if latent_path.exists():
                pairs.append((mp4, latent_path))
        print(f"[DataLoader] Found {len(pairs)} paired (mp4, latent) files.")
        return pairs

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if not self.pairs:
                raise RuntimeError("All pairs failed to load.")

            idx = self.rng.randrange(len(self.pairs))
            mp4_path, latent_path = self.pairs[idx]

            try:
                # Cache video duration
                if mp4_path not in self.meta:
                    with av.open(str(mp4_path)) as c:
                        if not any(s.type == "audio" for s in c.streams):
                            self.pairs.pop(idx)
                            continue
                        dur = (c.duration / 1e6) if c.duration is not None else 600.0
                    self.meta[mp4_path] = float(dur)

                dur = self.meta[mp4_path]
                max_t = dur - self.window_length - 0.05
                if max_t <= 0:
                    self.pairs.pop(idx)
                    continue

                t_start = self.rng.random() * max_t

                # mmap-load latent — no full tensor in memory
                latent = torch.load(str(latent_path), map_location="cpu", mmap=True)
                if tuple(latent.shape[-2:]) != self.expected_hw:
                    continue
                n_latent = len(latent)  # total latent frames for full video

                latent_start = int(t_start / dur * n_latent)
                if latent_start + self.video_window_frames > n_latent:
                    continue

                # Decode audio window
                audio = self._decode_audio(str(mp4_path), t_start)  # [C, T_audio]

                # Clone the mmap slice to get an owned tensor
                video = latent[latent_start : latent_start + self.video_window_frames].clone()

                return audio, video  # ([C, T_audio], [T_lat, C, H, W])

            except Exception:
                self.pairs.pop(idx)
                continue

    def _decode_audio(self, path, t_sec):
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=self.sample_rate)
        chunks = []
        needed = self.window_length_samples

        with av.open(path) as container:
            a_stream = next(s for s in container.streams if s.type == "audio")
            tb = a_stream.time_base
            container.seek(int(max(0.0, t_sec) / float(tb)), stream=a_stream, backward=True)

            started = False
            done = False
            for pkt in container.demux(a_stream):
                for frame in pkt.decode():
                    if not started:
                        if frame.time is not None:
                            frame_end = frame.time + frame.samples / max(1, frame.sample_rate)
                            if frame_end + 1e-3 < t_sec:
                                continue
                        started = True
                    for rf in resampler.resample(frame):
                        chunks.append(rf.to_ndarray())
                    if sum(c.shape[1] for c in chunks) >= needed:
                        done = True
                        break
                if done:
                    break
            for rf in resampler.resample(None):
                chunks.append(rf.to_ndarray())

        if not chunks:
            raise RuntimeError("No audio decoded")

        audio = np.concatenate(chunks, axis=1)
        if audio.shape[1] < needed:
            pad = np.zeros((audio.shape[0], needed - audio.shape[1]), dtype=audio.dtype)
            audio = np.concatenate([audio, pad], axis=1)
        else:
            audio = audio[:, :needed]
        return audio  # [C, T_audio] float32


class RandomAudioVideoDataset(IterableDataset):
    """Infinite stream of (audio [C, T], video [T_lat, C, H, W]) pairs in bfloat16."""
    def __init__(self, source, encoded, seed=0,
                 window_length=10.0, sample_rate=44100,
                 video_window_frames=75, expected_hw=(16, 32)):
        super().__init__()
        self.source = source
        self.encoded = encoded
        self.seed = int(seed)
        self.window_length = window_length
        self.sample_rate = sample_rate
        self.video_window_frames = video_window_frames
        self.expected_hw = expected_hw

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        wseed = (torch.initial_seed() + self.seed + wid) % (2**32)
        rng = RandomAudioVideoFromMP4s(
            self.source, self.encoded, seed=int(wseed),
            window_length=self.window_length,
            sample_rate=self.sample_rate,
            video_window_frames=self.video_window_frames,
            expected_hw=self.expected_hw,
        )
        for audio, video in rng:
            yield torch.from_numpy(audio).bfloat16(), video.bfloat16()


def get_loader(batch_size, **data_kwargs):
    if "seed" not in data_kwargs:
        data_kwargs["seed"] = 123
    ds = RandomAudioVideoDataset(**data_kwargs)
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
        multiprocessing_context="spawn",
    )


if __name__ == "__main__":
    import time

    loader = get_loader(
        2,
        source="/mnt/data/waypoint_1/owl_control/processed",
        encoded="/mnt/data/waypoint_1/owl_control/encoded",
        window_length=10.0,
        sample_rate=44100,
        video_window_frames=75,
    )

    loader_iter = iter(loader)
    total_time = 0.0
    for i in range(10):
        t0 = time.time()
        audio, video = next(loader_iter)
        t1 = time.time()
        elapsed = t1 - t0
        total_time += elapsed
        if i == 0:
            print(f"Audio shape : {tuple(audio.shape)}, dtype={audio.dtype}")
            print(f"Video shape : {tuple(video.shape)}, dtype={video.dtype}")
        print(f"Batch {i+1:2d}: {elapsed:.3f}s")

    print(f"\nAverage over 10 batches: {total_time / 10:.3f}s")
