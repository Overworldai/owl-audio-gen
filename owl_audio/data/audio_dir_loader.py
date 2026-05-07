import os, glob, random
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import av


class RandomAudioFromMP4s:
    """
    Continuous iterator yielding random audio windows as [C, T] float32 numpy arrays.
    Uniform over files; uniform over time within each file.
    Audio is resampled to target sample_rate and normalized to [-1, 1].
    """
    def __init__(self, source, seed=None, window_length=2.0, sample_rate=16000):
        if isinstance(source, str):
            source = [source]
        self.window_length = window_length          # seconds
        self.sample_rate = sample_rate              # samples/sec
        self.window_length_samples = int(window_length * sample_rate)
        self.paths = self._find_mp4s(source)
        if not self.paths:
            raise RuntimeError("No MP4s found in the supplied source.")
        self.rng = random.Random(seed)
        self.meta = {}  # path -> duration_s

    @staticmethod
    def _find_mp4s(spec):
        specs = [spec] if isinstance(spec, (str, Path)) else list(spec)
        out = []
        for s in specs:
            s = os.path.expanduser(str(s))
            p = Path(s)
            registry_file = p / "valid_mp4s.txt"
            if p.is_dir() and registry_file.exists():
                print(f"[DataLoader] Found registry {registry_file}, using pre-filtered paths.")
                with open(registry_file, 'r') as f:
                    out.extend([line.strip() for line in f if line.strip()])
                continue
            if p.exists() and p.is_dir():
                out += glob.glob(str(p / "**/*.mp4"), recursive=True)
            elif p.exists() and p.is_file() and p.suffix.lower() == ".mp4":
                out.append(str(p))
            else:
                out += glob.glob(s, recursive=True)
        filtered = [x for x in out if x.lower().endswith(".mp4") and '_' not in Path(x).stem]
        return [Path(x) for x in sorted(set(filtered))]

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            if not self.paths:
                raise RuntimeError("All videos in this worker failed to load audio.")

            idx = self.rng.randrange(len(self.paths))
            p = self.paths[idx]
            try:
                if p not in self.meta:
                    with av.open(str(p)) as c:
                        a = next((s for s in c.streams if s.type == "audio"), None)
                        if a is None:
                            self.paths.pop(idx)
                            continue
                        dur = (c.duration / 1e6) if c.duration is not None else 600.0
                    self.meta[p] = float(dur)

                dur = self.meta[p]
                max_t = dur - self.window_length - 0.05
                if max_t <= 0:
                    self.paths.pop(idx)
                    continue

                t = self.rng.random() * max_t
                return self._decode_audio(str(p), t)

            except Exception:
                self.paths.pop(idx)
                continue

    def _decode_audio(self, path, t_sec):
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=self.sample_rate)
        chunks = []
        needed = self.window_length_samples

        with av.open(path) as container:
            a_stream = next(s for s in container.streams if s.type == "audio")
            tb = a_stream.time_base
            seek_ts = int(max(0.0, t_sec) / float(tb))
            container.seek(seek_ts, stream=a_stream, backward=True)

            started = False
            done = False
            for pkt in container.demux(a_stream):
                for frame in pkt.decode():
                    if not started:
                        # skip frames that end before t_sec
                        if frame.time is not None:
                            frame_end = frame.time + frame.samples / max(1, frame.sample_rate)
                            if frame_end + 1e-3 < t_sec:
                                continue
                        started = True

                    for rf in resampler.resample(frame):
                        chunks.append(rf.to_ndarray())  # [C, samples] for fltp

                    if sum(c.shape[1] for c in chunks) >= needed:
                        done = True
                        break
                if done:
                    break

            for rf in resampler.resample(None):  # flush
                chunks.append(rf.to_ndarray())

        if not chunks:
            raise RuntimeError("No audio decoded")

        audio = np.concatenate(chunks, axis=1)  # [C, total_samples]

        if audio.shape[1] < needed:
            pad = np.zeros((audio.shape[0], needed - audio.shape[1]), dtype=audio.dtype)
            audio = np.concatenate([audio, pad], axis=1)
        else:
            audio = audio[:, :needed]

        return audio  # [C, window_length_samples], float32 in [-1, 1]


class RandomAudioDataset(IterableDataset):
    """Infinite stream of [C, T] bfloat16 audio windows in [-1, 1]."""
    def __init__(self, source, seed=0, window_length=2.0, sample_rate=16000):
        super().__init__()
        self.source = source
        self.seed = int(seed)
        self.window_length = window_length
        self.sample_rate = sample_rate

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        wseed = (torch.initial_seed() + self.seed + wid) % (2**32)
        rng = RandomAudioFromMP4s(
            self.source, seed=int(wseed),
            window_length=self.window_length,
            sample_rate=self.sample_rate,
        )
        for audio in rng:
            yield torch.from_numpy(audio).bfloat16()  # [C, T]


def get_loader(batch_size, **data_kwargs):
    if "seed" not in data_kwargs:
        data_kwargs["seed"] = 123
    ds = RandomAudioDataset(**data_kwargs)
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
        4,
        source="/mnt/data/waypoint_1/owl_control/processed",
        window_length=2.0,
        sample_rate=16000,
    )

    loader_iter = iter(loader)
    total_time = 0.0
    for i in range(10):
        t0 = time.time()
        batch = next(loader_iter)
        t1 = time.time()
        elapsed = t1 - t0
        total_time += elapsed
        if i == 0:
            print(f"Batch shape: {tuple(batch.shape)}, dtype: {batch.dtype}")
            print(f"Value range: [{batch.min().item():.4f}, {batch.max().item():.4f}]")
        print(f"Batch {i+1:2d}: {elapsed:.3f}s")

    print(f"\nAverage over 10 batches: {total_time / 10:.3f}s")
